import os
import numpy as np
from sumo_files.env.sim_env import TSCSimulator
from tsc.utils import to_colight_obs, log, batch_log, bulk_log_multi_process 
from CoLight_agent import CoLightAgent
import time


class Generator:
    def __init__(self, cnt_round, cnt_gen, config, dic_traffic_env_conf, dic_path, dic_agent_conf, q):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.agents = [None]
        self.config = config
        self.config["sumocfg_file"] = config["sumocfg_file"][cnt_gen]
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.dic_agent_conf = dic_agent_conf
        self.dic_agent_conf['cnt_gen'] = cnt_gen

        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            try:
                os.makedirs(self.path_to_log)  
            except:
                pass
            
        self.config['path_to_log'] = self.path_to_log
        #self.config['p'] = 3 * self.cnt_round 
        self.config["is_record"] = False
        self.env = TSCSimulator(self.config, self.config['port_start'] + self.cnt_gen) 

        self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(self.env.all_tls)
        
        tl_unavav_index = {}
        for tl in self.env._crosses:
            tl_unavav_index[tl] = self.env._crosses[tl].unava_index
        self.dic_agent_conf['tl_unavav_index'] = tl_unavav_index
        
        q.put({cnt_gen:[len(self.env.all_tls),tl_unavav_index]})
        
        self.env.reset()
        
        start_time = time.time()

        for i in range(1):
            agent = CoLightAgent(
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round, 
                best_round=None,
                intersection_id=str(i)
            )
            self.agents[i] = agent
        print("Create intersection agent time: ", time.time()-start_time)

    def generate(self):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        list_inter_log = [[] for i in range(len(self.env.all_tls))]
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time

        running_start_time = time.time()
            
        while not done :
            tl_action_select = {}
            step_start_time = time.time()
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                before_action_obs = self.env._get_state()
                action, _ = self.agents[i].choose_action(step_num, to_colight_obs(state))  
                if self.env._current_time >= 57600 - 1:
                    log(self.env.all_tls, list_inter_log, self.env._current_time - (57600 - 1), to_colight_obs(before_action_obs), action)
                elif self.env._current_time <= 57600 - 1 and self.env._current_time >= 25200 - 1:
                    log(self.env.all_tls, list_inter_log, self.env._current_time - (25200 - 1), to_colight_obs(before_action_obs), action)
                else:
                    log(self.env.all_tls, list_inter_log, self.env._current_time, to_colight_obs(before_action_obs), action) 
            for i in range(len(action)):
                tl_action_select[self.env.all_tls[i]] = int(action[i] * 2)
            next_state, reward, done, _ = self.env.step(tl_action_select)     
                 
            # tl_action_select = {}
            # for tl in self.env.all_tls:
            #     a = np.random.choice(self.env._crosses[tl].green_phases)
            #     while a / 2 in self.env._crosses[tl].unava_index:
            #         a = np.random.choice(self.env._crosses[tl].green_phases)
            #     tl_action_select[tl] = a
            # next_state, reward, done, _ = self.env.step(tl_action_select)    
                    
            print("time: {0}, running_time: {1}".format(self.env._current_time,
                                                        time.time()-step_start_time))
            state = next_state
            step_num += 1
        running_time = time.time() - running_start_time

        log_start_time = time.time()
        print("start logging")
        bulk_log_multi_process(self.path_to_log, self.env.all_tls, list_inter_log)
        log_time = time.time() - log_start_time

        self.env.terminate()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)