import json
import os
import pickle
from copy import deepcopy
from sumo_files.env.sim_env import TSCSimulator
from tsc.utils import to_colight_obs, log, batch_log, bulk_log_multi_process 
from CoLight_agent import CoLightAgent
import time
import numpy as np


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def downsample(path_to_log, i):
    path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
    with open(path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(path_to_pkl)
    with open(path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)

def downsample_for_system(path_to_log,dic_traffic_env_conf):
    for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
        downsample(path_to_log,i)



# TODO test on multiple intersections
def test(model_dir, cnt_round, cnt_gen, run_cnt, _dic_traffic_env_conf, dic_agent_conf, dic_exp_conf, config, if_gui, q, eval_map_index):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d"%cnt_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)


    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []

    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
    if not os.path.exists(path_to_log):
        try:
            os.makedirs(path_to_log)
        except:
            pass
    config["sumocfg_file"] = config["eval_sumocfg_file"][eval_map_index]
    config['path_to_log'] = path_to_log
    config['p'] = 2 * cnt_round + 1
    config['output_path'] = config['output_path'] + f'trial_{cnt_gen}/'
    if not os.path.exists(config['output_path']):
        try:
            os.makedirs(config['output_path'])
        except:
            pass
    env = TSCSimulator(config, config['port_start'] + cnt_gen + 10 * eval_map_index)
    
    dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(env.all_tls)
    
    tl_unavav_index = {}
    for tl in env._crosses:
        tl_unavav_index[tl] = env._crosses[tl].unava_index
    dic_agent_conf['tl_unavav_index'] = tl_unavav_index

    done = False
    state = env.reset()
    list_inter_log = [[] for i in range(len(env.all_tls))]

    for i in range(1):
        agent = CoLightAgent(
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=cnt_round, 
            best_round=None,
            intersection_id=str(i)
        )
        agents.append(agent)
        

    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agents[i].q_network.load_weights(
            os.path.join(dic_path["PATH_TO_MODEL"], "{0}_inter_{1}.h5".format(model_round, 0)),  
            by_name=True)


    step_num = 0

    attention_dict = {}

    while not done :
        tl_action_select = {}
        step_start_time = time.time()
        for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
            before_action_obs = env._get_state()
            action, _ = agents[i].choose_action(step_num, to_colight_obs(state))  
            if env._current_time >= 57600 - 1:
                log(env.all_tls, list_inter_log, env._current_time - (57600 - 1), to_colight_obs(before_action_obs), action)
            elif env._current_time <= 57600 - 1 and env._current_time >= 25200 - 1:
                log(env.all_tls, list_inter_log, env._current_time - (25200 - 1), to_colight_obs(before_action_obs), action)
            else:
                log(env.all_tls, list_inter_log, env._current_time, to_colight_obs(before_action_obs), action)   
        for i in range(len(action)):
            tl_action_select[env.all_tls[i]] = int(action[i] * 2)
        next_state, reward, done, all_reward = env.step(tl_action_select)     

        print("time: {0}, running_time: {1}".format(env._current_time,
                                                    time.time()-step_start_time))
        state = next_state
        step_num += 1
    # print('bulk_log_multi_process')
    bulk_log_multi_process(path_to_log, env.all_tls, list_inter_log)
    q.put(all_reward)
    env.terminate()
    # if not dic_exp_conf["DEBUG"]:
    #     path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
    #                                 model_round)
    #     # print("downsample", path_to_log)
    #     downsample_for_system(path_to_log, dic_traffic_env_conf)
    #     # print("end down")