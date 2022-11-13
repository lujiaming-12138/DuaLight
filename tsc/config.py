#!/usr/bin/env python3
# encoding: utf-8
"""
@paper: A GENERAL SCENARIO-AGNOSTIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL
@file: config
"""
name = 'test'
config = {
    "episode": {
        "num_train_rollouts": 10000,
        "rollout_length": 128,
        "warmup_ep_steps": 0,
    },
    "agent": {
        "agent_type": "ppo",
        "single_agent": True
    },
    "ppo": {
        "gae_tau": 0.85,
        "entropy_weight": 0.01,   # (0-0.01)
        "minibatch_size": 128, # maybe larger for better performance
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
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        "is_record": True,
        'output_path': 'sumo_logs/{}/'.format(name),
        "name": name,
        "sumocfg_files": [
            "sumo_files/scenarios/large_grid2/exp_0.sumocfg",

            "sumo_files/scenarios/large_grid2/exp_1.sumocfg",
            "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",

            'sumo_files/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
            'sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg',
            'sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg',
            'sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg'
            
            "sumo_files/scenarios/nanshan/osm.sumocfg",
        ],
        "state_key": ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num'],
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure']
    },
    "model_save": {
        "frequency": 500,  # You can adjust it to be smaller for save more checkpoints
        "path": "tsc/{}".format(name)
    },
    "parallel": {
        "num_workers": 8 # 8 for test run, recommended 32 for training
    }
}