#!/usr/bin/env python3
# encoding: utf-8
"""
@paper: A GENERAL SCENARIO-AGNOSTIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL
@file: eval_v2
"""
import pickle
import os
import sys
from eval_config import config
from functools import partial
import copy
from tqdm.contrib.concurrent import process_map
from sumo_files.env.sim_env import TSCSimulator
from model import NN_Model
from utils import batch
import torch


def get_action(prediction):
    return prediction['a'].cpu().numpy()


def get_reward(reward, all_tls):
    ans = []
    for i in all_tls:
        ans.append(sum(reward[i].values()))
    return ans


def worker(config, port, checkpoints, p):
    c = checkpoints[p]
    port = copy.deepcopy(port)
    port += p
    env_config = config['environment']
    model = NN_Model(8, 4, state_keys=env_config['state_key'], device=device).to(device)
    model.load_state_dict(
        torch.load(os.path.join(config['model_save']['path'], c))['model_state_dict'])
    model.eval()
    print("Load {} model done!".format(os.path.join(config['model_save']['path'], c)))
    all_reward_list = {}
    for k in config['environment']['reward_type'] + ['all']:
        all_reward_list[k] = []
    for i in range(5):
        # try:
        env_config['step_num'] = i + 1
        env = TSCSimulator(env_config, port)
        unava_phase_index = []
        for j in env.all_tls:
            unava_phase_index.append(env._crosses[j].unava_index)
        state = env.reset()
        next_state = state
        while True:
            state = batch(next_state, config['environment']['state_key'], env.all_tls)
            prediction = model(state, unava_phase_index, eval=True)
            action = get_action(prediction)
            tl_action_select = {}
            for tl_index in range(len(env.all_tls)):
                tl_action_select[env.all_tls[tl_index]] = \
                    (env._crosses[env.all_tls[tl_index]].green_phases)[action[tl_index]]
            next_state, reward, done, _all_reward = env.step(tl_action_select)
            # reward = get_reward(reward, env.all_tls)
            if done:
                all_reward = _all_reward
                break
        for tl in all_reward.keys():
            all_reward[tl]['all'] = sum(all_reward[tl].values())
        for k in config['environment']['reward_type']+['all']:
            tmp = 0
            for tl in all_reward.keys():
                tmp += all_reward[tl][k]
            all_reward_list[k].append(tmp/len(all_reward))
        # except Exception as e:
        #     print(e)
        #     print(p, i)

    shared_dict = {}
    shared_dict[env_config['sumocfg_file'].split("/")[-1]] = {}
    for k, v in all_reward_list.items():
        shared_dict[env_config['sumocfg_file'].split("/")[-1]][k] = sum(v) / len(v)
        print("epoch:{}, {} Model Avg {}: {}".format(p, env_config['sumocfg_file'], k,
                                                     sum(v) / len(v)))
    with open(os.path.join(config['model_save']['path'], "reward_record_2_{}_{}.pkl".format(env_config['sumocfg_file'].split("/")[-1], p)), "wb") as f:
        pickle.dump(shared_dict, f)

def model_eval(config, port):
    config = copy.deepcopy(config)
    checkpoints_ = os.listdir(config['model_save']['path'])
    checkpoints = [i for i in checkpoints_ if "model" in i]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1][:-3]))
    # worker(config, port, checkpoints,0)
    process_map(partial(worker, config, port, checkpoints), range(len(checkpoints)), max_workers=8)


if __name__ == '__main__':
    device = 'cpu'
    # config["environment"]['gui'] = True
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    env_config = config['environment']
    for i in range(len(env_config['sumocfg_files'])):
        env_config['sumocfg_file'] = env_config['sumocfg_files'][i]
        port = env_config['port_start']
        model_eval(config, port)
        # default_eval()

    # data = {}
    # try:
    #     for i in range(100):
    #         data[i] = {}
    #         for j in config['sumocfg_files']:
    #             name = j.split("/")[-1]
    #             with open(os.path.join(config['model_save']['path'], "reward_record_2_{}_{}.pkl".format(name, i)),
    #                       "rb") as f:
    #                 data_ = pickle.load(f)
    #             data[i][name] = data_
    # except Exception as e:
    #     print(e)
    #
    # with open(os.path.join(config['model_save']['path'], "reward_record_2.pkl"), "wb") as f:
    #     pickle.dump(data, f)