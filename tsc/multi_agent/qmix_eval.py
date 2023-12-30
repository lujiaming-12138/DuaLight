#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: qmix_eval
@time: 2023/1/13
"""
#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: eval_v2
@time: 2022/10/26
"""
import pickle
import os
import sys
from tsc.multi_agent.qmix_eval_config import config
from functools import partial
import copy
from tqdm.contrib.concurrent import process_map
from sumo_files.env.sim_env import TSCSimulator
from tsc.multi_agent.qmix import BodyModel, batch
import torch


def get_action(prediction):
    return prediction['a'].cpu().detach().numpy()


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
    model_body = BodyModel(config["qmix"]["input_shape"], 4, 256, config["environment"]['state_key'])  #
    model_body.load_state_dict(
        torch.load(os.path.join(config['model_save']['path'], c))['body_model_state_dict'])
    model_body.eval()
    print("Load {} model done!".format(os.path.join(config['model_save']['path'], c)))
    all_reward_list = {}
    for k in config['environment']['reward_type'] + ['all']:
        all_reward_list[k] = []
    for i in range(5):
        # try:
        env_config['step_num'] = i + 1
        env_config['p'] = 2 * p + 1
        env_config['output_path'] = env_config['output_path_head'] + f'trial_{i}/' #不同的重复实验保存到不同的文件夹
        env = TSCSimulator(env_config, port)
        unava_phase_index = []
        for j in env.all_tls:
            unava_phase_index.append(env._crosses[j].unava_index)
        state = env.reset()
        next_state = state
        eval_hidden = None
        while True:
            state = batch(next_state, config['environment']['state_key'], env.all_tls)
            q_value, eval_hidden, _ = model_body(state, eval_hidden)
            for i in range(q_value.shape[0]):
                q_value[i, unava_phase_index[i]] = -float("inf")
            action = torch.argmax(q_value, axis=1).cpu().detach().numpy()
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
    for i in range(5):
        output_path = env_config['output_path_head'] + f'trial_{i}'
        os.makedirs(output_path, exist_ok=True)
    checkpoints_ = os.listdir(config['model_save']['path'])
    checkpoints = [i for i in checkpoints_ if "model" in i]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1][:-3]))
    # worker(config, port, checkpoints,0)
    #对checkpoint进行并行
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

    data = {}
    try:
        for i in range(100):
            data[i] = {}
            for j in config['sumocfg_files']:
                name = j.split("/")[-1]
                with open(os.path.join(config['model_save']['path'], "reward_record_2_{}_{}.pkl".format(name, i)),
                          "rb") as f:
                    data_ = pickle.load(f)
                data[i][name] = data_
    except Exception as e:
        print(e)

    with open(os.path.join(config['model_save']['path'], "reward_record_2.pkl"), "wb") as f:
        pickle.dump(data, f)

