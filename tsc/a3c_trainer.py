#!/usr/bin/env python3
# encoding: utf-8
"""
@paper: A GENERAL SCENARIO-AGNOSTIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL
@file: train
"""
import copy
import os

from torch.utils.tensorboard import SummaryWriter

from PPO import PPOWorker
from model import NN_Model
from utils import *
from sumo_files.env.sim_env import TSCSimulator


def write_tensorboard(writer, loss_actor, loss_critic, loss_entropy, advantages, reward, step):
    writer.add_scalar('loss_actor', loss_actor, step)
    writer.add_scalar('loss_critic', loss_critic, step)
    writer.add_scalar('loss_entropy', loss_entropy, step)
    writer.add_scalar('advantages', advantages, step)
    if reward:
        r = {}
        for tl, rs in reward.items():
            for k, v in rs.items():
                if k not in r:
                    r[k] = []
                r[k].append(v)
        total_reward = 0
        for k, v in r.items():
            writer.add_scalar('reward/{}'.format(k), np.mean(v), step)
            total_reward += np.mean(v)
            print('Step{}, reward_{}: {}'.format(step, k, np.mean(v)))
        writer.add_scalar('reward/all', total_reward, step)
    flush_writer(writer)


# target for multiprocess
def train_worker(id, shared_NN, env_config, port, optimizer, rollout_counter, constants, device):
    set_seed(id+2022)
    sumo_envs_num = len(env_config['sumocfg_files'])
    sumo_cfg = env_config['sumocfg_files'][id % sumo_envs_num]
    env_config = copy.deepcopy(env_config)
    env_config['sumocfg_file'] = sumo_cfg
    env = TSCSimulator(env_config, port)
    unava_phase_index = []
    for i in env.all_tls:
        unava_phase_index.append(env._crosses[i].unava_index)

    # Assumes PPO worker
    local_NN = NN_Model(8, 4, state_keys=env_config['state_key'], device=device).to(device)
    worker = PPOWorker(constants, device, env, shared_NN, local_NN, optimizer, id)
    train_step = 0
    writer = SummaryWriter(comment=sumo_cfg)
    while rollout_counter.get() < constants['episode']['num_train_rollouts'] + 1:
        policy_loss, entropy_loss, value_loss, sampled_advantages, all_reward = worker.train_rollout(train_step, unava_phase_index)
        write_tensorboard(writer, policy_loss, value_loss, entropy_loss, sampled_advantages, all_reward, rollout_counter.get())
        rollout_counter.increment()
        if rollout_counter.get() % constants['model_save']['frequency'] == 0:
            checkpoint(constants['model_save']['path'], shared_NN, optimizer, rollout_counter.get())

    # Kill connection to sumo server
    worker.env.terminate()
    print('...Training worker {} done'.format(id))


def train_A3C_PPO(config, device):
    # _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    env_config = config['environment']
    shared_NN = NN_Model(8, 4, state_keys=env_config['state_key'], device=device).to(device)
    if not os.path.exists(config['model_save']['path']):
        os.mkdir(config['model_save']['path'])
    files = os.listdir(config['model_save']['path'])
    rollout_counter = Counter()
    if len(files) > 0:
        nums = []
        for i in files:
            nums.append(int(i.split("_")[-1].split(".")[0]))
        max_file = os.path.join(config['model_save']['path'], files[nums.index(max(nums))])
        shared_NN = load_checkpoint2(shared_NN, max_file)
        step = max(nums)
        rollout_counter.increment(step)
    shared_NN.share_memory()
    optimizer = torch.optim.Adam(shared_NN.parameters(), config['ppo']['learning_rate'])
      # To keep track of all the rollouts amongst agents
    processes = []

    for i in range(config['parallel']['num_workers']):
        id = i
        port = env_config['port_start'] + i
        p = mp.Process(target=train_worker, args=(id, shared_NN, env_config, port,  optimizer,
                                                  rollout_counter, config, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    from config import config

    device = 'cpu'
    env_config = config['environment']
    shared_NN = NN_Model(8, 4, state_keys=env_config['state_key'], device=device).to(device)
    shared_NN.share_memory()
    optimizer = torch.optim.Adam(shared_NN.parameters(), config['ppo']['learning_rate'])
    rollout_counter = Counter()  # To keep track of all the rollouts amongst agents
    processes = []
    # Run eval agent
    # id = 1
    # port = env_config['port_start'] - 1
    # train_worker(id, shared_NN, env_config, port,  optimizer, rollout_counter, config, device)
    for i in range(config['parallel']['num_workers']):
        id = i
        port = env_config['port_start'] + i
        p = mp.Process(target=train_worker, args=(id, shared_NN, env_config, port, optimizer,
                                                  rollout_counter, config, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()