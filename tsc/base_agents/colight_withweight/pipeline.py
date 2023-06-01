from sumo_files.env.sim_env import TSCSimulator

import os
from construct_sample import ConstructSample
import model_test
from updater import Updater
from generator import Generator
import time
import pickle
import traceback
from multiprocessing import Process, Pool
import multiprocessing

class Pipeline:
    def __init__(self, config, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.config = config
        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        env = TSCSimulator(self.config, self.config['port_start']) 
        
        self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(env.all_tls)
        self.dic_agent_conf['tl_unavav_index'] = env._crosses
        env.terminate()
        
        
    def generator_wrapper(self, cnt_round, cnt_gen, config, dic_traffic_env_conf, dic_path, dic_agent_conf):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              config = config,
                              dic_traffic_env_conf = dic_traffic_env_conf,
                              dic_path = dic_path,
                              dic_agent_conf = dic_agent_conf
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return
    
    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, config, best_round=None, bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            config = config,
            best_round=best_round,
            bar_round=bar_round
        ) 

        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")
        return   

    def downsample(self, path_to_log, i):

        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
        with open(path_to_pkl, "rb") as f_logging_data:
            try:
                logging_data = pickle.load(f_logging_data)
                subset_data = logging_data[::10]
                print(subset_data)
                os.remove(path_to_pkl)
                with open(path_to_pkl, "wb") as f_subset:
                    try:
                        pickle.dump(subset_data, f_subset)
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
                        print("----------------------------")

            except Exception as e:
                # print("CANNOT READ %s"%path_to_pkl)
                print("----------------------------")
                print("Error occurs when READING pickles when down sampling for inter {0}, {1}".format(i, f_logging_data))
                print('traceback.format_exc():\n%s' % traceback.format_exc())
                print("----------------------------") 

    def downsample_for_system(self, path_to_log, dic_traffic_env_conf):
        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.downsample(path_to_log, i)
            
    def make_reward_record(self, results, cnt_round):
        all_reward_list = {}
        for k in self.config['reward_type'] + ['all']:
            all_reward_list[k] = []
        for all_reward in results:
            for tl in all_reward.keys():
                all_reward[tl]['all'] = sum(all_reward[tl].values())
            for k in self.config['reward_type']+['all']:
                tmp = 0
                for tl in all_reward.keys():
                    tmp += all_reward[tl][k]
                all_reward_list[k].append(tmp/len(all_reward))
        shared_dict = {}
        shared_dict[self.config['sumocfg_file'].split("/")[-1]] = {}
        for k, v in all_reward_list.items():
            shared_dict[self.config['sumocfg_file'].split("/")[-1]][k] = sum(v) / len(v)
            print("epoch:{}, {} Model Avg {}: {}".format(cnt_round, self.config['sumocfg_file'], k,
                                                        sum(v) / len(v)))
        if not os.path.exists(self.config['model_save']['path']):
            os.makedirs(self.config['model_save']['path'])               
        with open(os.path.join(self.config['model_save']['path'], "reward_record_2_{}_{}.pkl".format(self.config['sumocfg_file'].split("/")[-1], cnt_round)), "wb") as f:
            pickle.dump(shared_dict, f)
    
    def run(self, multi_process=False):
        # trainf
        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            process_list = []

            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    p = Process(target=self.generator_wrapper,
                                args=(cnt_round, cnt_gen, self.config, self.dic_traffic_env_conf, self.dic_path, self.dic_agent_conf)
                                )
                    print("before p")
                    p.start()
                    print("end p")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):#
                    self.generator_wrapper(cnt_round, 
                                           cnt_gen, 
                                           self.config, 
                                           self.dic_traffic_env_conf, 
                                           self.dic_path, 
                                           self.dic_agent_conf
                                           )
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time
            
            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()

            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)

            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()
            
            print("==============  update network =============")
            
            if multi_process:
                p = Process(target=self.updater_wrapper,
                            args=(cnt_round,
                                    self.dic_agent_conf,
                                    self.dic_exp_conf,
                                    self.dic_traffic_env_conf,
                                    self.dic_path,
                                    self.config))
                p.start()
                print("update to join")
                p.join()
                print("update finish join")
            else:
                self.updater_wrapper(cnt_round=cnt_round,
                                        dic_agent_conf=self.dic_agent_conf,
                                        dic_exp_conf=self.dic_exp_conf,
                                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                                        dic_path=self.dic_path,
                                        config=self.config)            

            if not self.dic_exp_conf["DEBUG"]:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                               "round_" + str(cnt_round), "generator_" + str(cnt_gen))
                    try:
                        self.downsample_for_system(path_to_log,self.dic_traffic_env_conf)
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when downsampling for round {0} generator {1}".format(cnt_round, cnt_gen))
                        print("traceback.format_exc():\n%s"%traceback.format_exc())
                        print("----------------------------")

            print("==============  test evaluation =============")
            process_list = []
            q = multiprocessing.Queue()
            
            for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                p = Process(target=model_test.test,
                            args=(self.dic_path["PATH_TO_MODEL"], cnt_round, cnt_gen, self.dic_exp_conf["RUN_COUNTS"],
                                self.dic_traffic_env_conf, self.dic_agent_conf, self.dic_exp_conf, self.config, False, q))
                print("before p")
                p.start()
                print("end p")
                process_list.append(p)
            print("before join")
            for i in range(len(process_list)):
                p = process_list[i]
                print("generator %d to join" % i)
                p.join()
                print("generator %d finish join" % i)
            print("end join")
            
            #make reward
            results = [q.get() for j in process_list]
            self.make_reward_record(results, cnt_round)
                

