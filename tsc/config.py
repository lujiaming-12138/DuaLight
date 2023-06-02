#!/usr/bin/env python3
# encoding: utf-8
name = 'test'
config = {
    "episode": {
        "num_train_rollouts": 100000,
        "rollout_length": 128,
        "warmup_ep_steps": 0,
        "test_num_eps": 50
    },
    "agent": {
        "agent_type": "ppo",
        "single_agent": True
    },
    "ppo": {
        "gae_tau": 0.85,
        "entropy_weight": 0.01,
        "minibatch_size": 128,
        "optimization_epochs": 4,
        "ppo_ratio_clip": 0.2,
        "discount": 0.99,
        "learning_rate": 1e-4,
        "clip_grads": True,
        "gradient_clip": 2.0,
        "value_loss_coef": 1.0,
        'target_kl': None
    },
    "environment": {
        'port_start': 15900, # 36
        "action_type": "select_phase",
        # "is_dis": True,
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        "is_record": False,
        'output_path': 'sumo_logs/{}/'.format(name),
        "name": name,
        "sumocfg_files": [
            "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
            "sumo_files/scenarios/nanshan/osm.sumocfg",
            "sumo_files/scenarios/large_grid2/exp_1.sumocfg",
            # # "sumo_files/scenarios/real_net/most_0.sumocfg",
            "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
            # "sumo_files/scenarios/sumo_wj3/rl_wj.sumocfg",

            'sumo_files/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
            'sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg',
            'sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg',
            'sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg'
        ],
        "state_key": ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num'],
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score']
    },
    "model_save": {
        "frequency": 500,
        "path": "tsc/{}".format(name)
    },
    "parallel": {
        "num_workers": 4
    }
}