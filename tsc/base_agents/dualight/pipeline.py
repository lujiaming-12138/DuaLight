from sumo_files.env.sim_env import TSCSimulator
import copy
import os
from construct_sample import ConstructSample
import model_test
from updater import Updater, Updater2
from generator import Generator
import time
import pickle
import traceback
from multiprocessing import set_start_method
set_start_method("spawn", force=True)
from multiprocessing import Process, Pool
import multiprocessing
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class Pipeline:
    def __init__(self, config, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.config = config
        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        
        self.dic_exp_conf["NUM_MAPS"] = len(config["sumocfg_file"])
        self.config["is_record"] = False
        
        self.dic_agent_conf["tl_cut"] = {}
        cut_start = 0
        for i in range(len(config["sumocfg_file"])):
            new_config = copy.deepcopy(self.config)
            new_config["sumocfg_file"] = new_config["sumocfg_file"][i]
            env = TSCSimulator(new_config, new_config['port_start']) 
            cut_end = len(env.all_tls) + cut_start
            self.dic_agent_conf["tl_cut"][i] = [cut_start, cut_end]
            cut_start = cut_end
        
    def generator_wrapper(self, cnt_round, cnt_gen, config, dic_traffic_env_conf, dic_path, dic_agent_conf,q):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              config = config,
                              dic_traffic_env_conf = dic_traffic_env_conf,
                              dic_path = dic_path,
                              dic_agent_conf = dic_agent_conf,
                              q = q
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return
    
    def construct_wrapper(self, train_round, cnt_round, cnt_gen, dic_traffic_env_conf):
        cs = ConstructSample(path_to_samples = train_round,
                                      cnt_round = cnt_round,
                                      cnt_gen = cnt_gen,
                                      dic_traffic_env_conf = dic_traffic_env_conf
                                      )
        print("make ConstructSample")
        cs.make_reward_for_system()
        print("ConstructSample_wrapper end")
        return
    
    def updater_wrapper(self, cnt_round, cnt_gen, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, config, q, best_round=None, bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            cnt_gen = cnt_gen,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            config = config,
            q = q,
            best_round=best_round,
            bar_round=bar_round,
        ) 

        updater.load_sample_for_agents()
        #updater.update_network_for_agents()
        print("updater_wrapper collect data end")
        return   
    
    def test_wrapper(self, model_dir, cnt_round, cnt_gen, run_cnt, _dic_traffic_env_conf, dic_agent_conf, dic_exp_conf, config, if_gui, q, eval_map_index):
        process_list_ = []
        q = multiprocessing.Queue()
        
        for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
            p = Process(target=model_test.test,
                        args=(model_dir, cnt_round, cnt_gen, run_cnt, _dic_traffic_env_conf,
                              dic_agent_conf, dic_exp_conf, config, if_gui, q, eval_map_index))
            print("evaluation before p")
            p.start()
            print("end p")
            process_list_.append(p)
        print("evaluation before join")
        for i in range(len(process_list_)):
            p = process_list_[i]
            print("evaluation generator %d to join" % i)
            p.join()
            print("evaluation generator %d finish join" % i)
        print("end join")
        
        #make reward
        results = [q.get() for j in process_list_]
        self.make_reward_record(results, cnt_round, eval_map_index)
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

    def downsample_for_system(self, path_to_log, dic_traffic_env_conf, cnt_gen):
        for i in range(dic_traffic_env_conf['maps_tl_nums'][cnt_gen]):
            self.downsample(path_to_log, i)
            
    def make_reward_record(self, results, cnt_round, eval_map_index):
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
        shared_dict[self.config["eval_sumocfg_file"][eval_map_index].split("/")[-1]] = {}
        for k, v in all_reward_list.items():
            shared_dict[self.config["eval_sumocfg_file"][eval_map_index].split("/")[-1]][k] = sum(v) / len(v)
            print("epoch:{}, {} Model Avg {}: {}".format(cnt_round, self.config["eval_sumocfg_file"][eval_map_index], k,
                                                        sum(v) / len(v)))
        if not os.path.exists(self.config['model_save']['path']):
            os.makedirs(self.config['model_save']['path'])               
        with open(os.path.join(self.config['model_save']['path'], "reward_record_2_{}_{}.pkl".format(self.config["eval_sumocfg_file"][eval_map_index].split("/")[-1], cnt_round)), "wb") as f:
            pickle.dump(shared_dict, f)
    
    def run(self, multi_process=False):
        # trainf
        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            os.popen('cd /tmp' + '&&' + 'rm -r pymp*')
            process_list = []
            q = multiprocessing.Queue()

            print("==============  generator =============")
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_MAPS"]):
                    p = Process(target=self.generator_wrapper,
                                args=(cnt_round, cnt_gen, self.config, self.dic_traffic_env_conf, self.dic_path, self.dic_agent_conf, q)
                                )
                    print("before p map: {}".format(self.config["sumocfg_file"][cnt_gen]))
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
                results = [q.get() for j in process_list]
                self.dic_traffic_env_conf["maps_tl_nums"] = {}
                self.dic_traffic_env_conf["maps_tl_unavav_index"] = {}
                for gen_index in results:
                    gen_name = list(gen_index)[0]
                    self.dic_traffic_env_conf["maps_tl_nums"][gen_name] = gen_index[gen_name][0]
                    self.dic_traffic_env_conf["maps_tl_unavav_index"][gen_name] = gen_index[gen_name][1]
            
            print("==============  make samples =============")
            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)
                
            process_list = []    
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_MAPS"]):
                    p = Process(target=self.construct_wrapper,
                                args=(train_round, cnt_round, cnt_gen, self.dic_traffic_env_conf)
                                )
                    print("before construct_wrapper map: {}".format(self.config["sumocfg_file"][cnt_gen]))
                    p.start()
                    print("end construct_wrapper")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("construct_wrapper %d to join" % i)
                    p.join()
                    print("construct_wrapper %d finish join" % i)
                print("end join")
            
            print("==============  update network =============")
            
            process_list = []
            q = multiprocessing.Queue()
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_MAPS"]):
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                        cnt_gen,
                                        self.dic_agent_conf,
                                        self.dic_exp_conf,
                                        self.dic_traffic_env_conf,
                                        self.dic_path,
                                        self.config,
                                        q))
                    print("before collect data map: {}".format(self.config["sumocfg_file"][cnt_gen]))
                    p.start()
                    print("end collect data")
                    process_list.append(p)    
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("collect data %d to join" % i)
                    p.join()
                    print("collect data %d finish join" % i)
                print("end join")    
                
                updater = Updater2(cnt_round,
                                    cnt_gen,
                                    self.dic_agent_conf,
                                    self.dic_exp_conf,
                                    self.dic_traffic_env_conf,
                                    self.dic_path,
                                    self.config,
                                    q)
                    

            for cnt_gen in range(self.dic_exp_conf["NUM_MAPS"]):
                path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(cnt_round), "generator_" + str(cnt_gen))
                try:
                    self.downsample_for_system(path_to_log,self.dic_traffic_env_conf, cnt_gen)
                except Exception as e:
                    print("----------------------------")
                    print("Error occurs when downsampling for round {0} generator {1}".format(cnt_round, cnt_gen))
                    print("traceback.format_exc():\n%s"%traceback.format_exc())
                    print("----------------------------")

            print("==============  test evaluation =============")
            process_list = []
            for eval_map_index in range(len(self.config["eval_sumocfg_file"])):
                p = Process(target=self.test_wrapper,
                                args=(self.dic_path["PATH_TO_MODEL"], cnt_round, cnt_gen, self.dic_exp_conf["RUN_COUNTS"],
                                    self.dic_traffic_env_conf, self.dic_agent_conf, self.dic_exp_conf, self.config, False, q, eval_map_index))
                print("before test_wrapper maps:{}".format(self.config["eval_sumocfg_file"][eval_map_index]))
                p.start()
                print("end test_wrapper")
                process_list.append(p)
            print("before join")
            for i in range(len(process_list)):
                p = process_list[i]
                print("test_wrapper %d to join" % i)
                p.join()
                print("test_wrapper %d finish join" % i)
            print("end join")
                # process_list_ = []
                # q = multiprocessing.Queue()
                
                # for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                #     p = Process(target=model_test.test,
                #                 args=(self.dic_path["PATH_TO_MODEL"], cnt_round, cnt_gen, self.dic_exp_conf["RUN_COUNTS"],
                #                     self.dic_traffic_env_conf, self.dic_agent_conf, self.dic_exp_conf, self.config, False, q, eval_map_index))
                #     print("evaluation before p")
                #     p.start()
                #     print("end p")
                #     process_list_.append(p)
                # print("evaluation before join")
                # for i in range(len(process_list_)):
                #     p = process_list_[i]
                #     print("evaluation generator %d to join" % i)
                #     p.join()
                #     print("evaluation generator %d finish join" % i)
                # print("end join")
                
                # #make reward
                # results = [q.get() for j in process_list_]
                # self.make_reward_record(results, cnt_round, eval_map_index)
                

