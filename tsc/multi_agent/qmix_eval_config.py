#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: qmix_config
@time: 2023/1/11
"""
#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: config
@time: 2022/8/3
"""
name = 'qmix2'
config = {
    "episode": {
        "num_train_rollouts": 100000,
        "episode_number": 1,
        "rollout_length": 240,
        "warmup_ep_steps": 0,
        "test_num_eps": 50
    },
    "agent": {
        "agent_type": "ppo",
        # "single_agent": True
    },
    "qmix": {
        "input_shape": 1,
        "state_shape": 512,
        "hyper_hidden_dim": 32,
        "qmix_hidden_dim": 32,
        "epsilon": 0.5,
        "anneal_epsilon": 0.00064,
        "min_epsilon": 0.02,
        "epsilon_anneal_scale": 'episode',
        # "gae_tau": 0.85,
        # "entropy_weight": 0.01,
        "minibatch_size": 128,  # 512
        # "optimization_epochs": 4,
        # "ppo_ratio_clip": 0.2,
        "discount": 0.99,
        "learning_rate": 1e-4,
        "clip_grads": True,
        "gradient_clip": 2.0,
        "value_loss_coef": 1.0,
        'predict_o_loss': 0.001,
        "optimizer": "RMS",
        "is_last_action": False,
        "target_update_cycle": 20,
        "gamma": 0.99
        # 'target_kl': None
    },
    "environment": {
        'port_start': 15900, # 36
        "action_type": "select_phase",
        # "is_dis": True,
        # "is_neighbor_reward": True,
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        "is_record": True,
        'output_path': 'sumo_logs/{}/'.format(name),
        'output_path_head': 'sumo_logs/{}/'.format("eval_"+name),
        "name": name,
        "sumocfg_files": [
            "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
            'sumo_files/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
            'sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg',
            'sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg',
            'sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg'
        ],
        "state_key": ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num'],
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score']
    },
    "model_save": {
        "frequency": 100,
        "path": "tsc/{}".format(name)
    },
    "parallel": {
        "num_workers": 8
    },
    "cuda": False,
    "evaluate_cycle": 100
}


map_configs = {
    # 'exp_0': {
    #     'net': 'sumo_files/scenarios/large_grid2/exp.net.xml',
    #     'route': None,
    #     'end_time': 3600,
    # },
    # 'osm': {
    #     'net': 'sumo_files/scenarios/nanshan/osm.net.xml',
    #     'route': None,
    #     'end_time': 3600,
    # },
    # 'exp_1': {
    #     'net': 'sumo_files/scenarios/large_grid2/exp.net.xml',
    #     'route': None,
    #     'end_time': 3600,
    # },
    # 'most_0': {
    #     'net': 'sumo_files/scenarios/real_net/in/most.net.xml',
    #     'route': None,
    #     'end_time': 3600,
    # },
    'base': {
        'net': 'sumo_files/scenarios/sumo_fenglin_base_road/fenglin_y2z_t.net.xml',
        'route': 'sumo_files/scenarios/sumo_fenglin_base_road/fenglin_y2z_t.rou.xml',#读取非随机车流的文件
        'end_time': 3600,
    },
    # 'rl_wj': {
    #     'net': 'sumo_files/scenarios/sumo_wj3/rl_wj.net.xml',
    #     'route': 'sumo_files/scenarios/sumo_wj3/testflow.rou.xml',
    #     'end_time': 3600,
    # },
    'grid4x4': {
        'net': 'sumo_files/scenarios/resco_envs/grid4x4/grid4x4.net.xml',
        'route': None,
        'end_time': 3600,
    },
    'arterial4x4': {
        'net': 'sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.net.xml',
        'route': None,
        'end_time': 3600,
    },
    'ingolstadt21': {
        'net': 'sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.net.xml',
        'route': 'sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.rou.xml',
        'end_time': 57600 + 3600,
    },
    'cologne8': {
        'net': 'sumo_files/scenarios/resco_envs/cologne8/cologne8.net.xml',
        'route': 'sumo_files/scenarios/resco_envs/cologne8/cologne8.rou.xml',
        'end_time': 25200 + 3600,
    },
}

metrics = ['timeLoss', 'duration', 'waitingTime']
