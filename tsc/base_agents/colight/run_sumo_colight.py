from pipeline import Pipeline


multi_process = True
TOP_K_ADJACENCY = 5
DIC_COLIGHT_AGENT_CONF = {
    "CNN_layers":[[32,32]],#,[32,32],[32,32],[32,32]],
    "att_regularization":False,
    "rularization_rate":0.03,
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    #special care for pretrain
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,

    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
}

DIC_EXP_CONF = {
    "RUN_COUNTS": 3600,
    "MODEL_NAME": 'CoLight',
    "NUM_ROUNDS": 100,
    "NUM_GENERATORS": 5,
    "LIST_MODEL":
        ["Fixedtime", "SOTL", "Deeplight", "SimpleDQN"],
    "LIST_MODEL_NEED_TO_UPDATE":
        ["Deeplight", "SimpleDQN", "CoLight","GCN", "SimpleDQNOne","Lit"],
    "MODEL_POOL": False,
    "NUM_BEST_MODEL": 3,
    "PRETRAIN": False,
    "PRETRAIN_MODEL_NAME": "Random",
    "PRETRAIN_NUM_ROUNDS": 0,
    "PRETRAIN_NUM_GENERATORS": 10,
    "AGGREGATE": False,
    "DEBUG": False,
    "EARLY_STOP": False,

    "MULTI_TRAFFIC": False,
    "MULTI_RANDOM": False,
}

dic_traffic_env_conf = {
    "USE_LANE_ADJACENCY": True,
    "ONE_MODEL": False,
    "NUM_AGENTS": 1,
    "NUM_INTERSECTIONS": None,
    "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
    "BINARY_PHASE_EXPANSION": True,
    "FAST_COMPUTE": True,
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,
    "MIN_ACTION_TIME": 15,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 8,
    "NUM_LANES": 1,
    "ACTION_DIM": 2,
    "MEASURE_TIME": 15,
    "IF_GUI": False,
    "DEBUG": False,

    "INTERVAL": 1,
    "THREADNUM": 8,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE = (4,),
        D_LEAVING_VEHICLE = (4,),

        D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
        D_CUR_PHASE=(8,),
        D_NEXT_PHASE=(1,),
        D_TIME_THIS_PHASE=(1,),
        D_TERMINAL=(1,),
        D_LANE_SUM_WAITING_TIME=(4,),
        D_VEHICLE_POSITION_IMG=(4, 60,),
        D_VEHICLE_SPEED_IMG=(4, 60,),
        D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

        D_PRESSURE=(1,),

        D_ADJACENCY_MATRIX=(2,)
    ),

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "lane_num_vehicle",
        "adjacency_matrix",
        'id'
    ],

    "DIC_REWARD_INFO": {
        "sum_num_vehicle_been_stopped_thres1": -0.25,
    },

    "LANE_NUM": {
        "LEFT": 1,
        "RIGHT": 0,
        "STRAIGHT": 1
    },


}
                
def pipeline_wrapper(config, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(config, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path)
    global multi_process
    ppl.run(multi_process=multi_process)

    print("pipeline_wrapper end")
    return

if __name__ == '__main__':
    name = "colight_libsumo"
    config = {
        'name': name,
        "agent": "colight",
        # "sumocfg_file": "sumo_files/scenarios/nanshan/osm.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg",
        # "sumocfg_file": 'sumo_files/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
        "sumocfg_file": "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
        'output_path': 'sumo_logs/{}/'.format(name),
        "action_type": "select_phase",
        "is_record": True,
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'state_key': ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num', 'pressure'],
        "model_save": {
            "path": "tsc/{}".format(name)
                        },
        'port_start': 1234,
        'is_dis': True,
        'is_adjacency_remove': False,
        'adjacency_top_k': TOP_K_ADJACENCY 
    }
    
    map_name = config["sumocfg_file"].split("/")[-1].split(".")[0]
    DIC_PATH = {
    "PATH_TO_MODEL": f"model/{map_name}",
    "PATH_TO_WORK_DIRECTORY": f"records/{map_name}",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_ERROR": f"errors/{map_name}"
                    }
    
    pipeline_wrapper(config, DIC_EXP_CONF, DIC_COLIGHT_AGENT_CONF, dic_traffic_env_conf, DIC_PATH)
