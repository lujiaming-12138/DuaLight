import pickle
import os
import pandas as pd
import shutil
import pandas as pd
import time
import traceback
import random
import numpy as np
from sumo_files.env.sim_env import TSCSimulator
from CoLight_agent import CoLightAgent, CoLightAgent_cotrain


class Updater:

    def __init__(self, cnt_round, cnt_gen, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, config, q, best_round=None, bar_round=None):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.config = config
        self.config["sumocfg_file"] = config["sumocfg_file"][cnt_gen]
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None
        self.q = q

        self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen]
        self.dic_agent_conf['tl_unavav_index'] = self.dic_traffic_env_conf["maps_tl_unavav_index"][cnt_gen]
        self.dic_agent_conf['cnt_gen'] = cnt_gen


        for i in range(1):
            agent = CoLightAgent(
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round, 
                best_round=None,
                intersection_id=str(i)
            )
            self.agents.append(agent)

    def load_sample(self, i):
        sample_set = []
        try:
            if self.dic_exp_conf["PRETRAIN"]:
                    sample_file = open(os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"],
                                                "train_round", "total_samples" + ".pkl"), "rb")
            elif self.dic_exp_conf["AGGREGATE"]:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_AGGREGATE_SAMPLES"],
                                                "aggregate_samples.pkl"), "rb")
            else:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_samples_generator_{0}_inter_{1}".format(self.cnt_gen, i) + ".pkl"), "rb")
            try:
                while True:
                    sample_set += pickle.load(sample_file)
            except EOFError:
                sample_file.close()
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i %100 ==0:
            print("load_sample for inter {0}".format(i))
        return sample_set

    def load_hidden_states_with_forget(self): # hidden state is a list [#time, agent, # dim]
        hidden_states_set = []
        try:
            hidden_state_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_hidden_states.pkl"), "rb")
            try:
                while True:
                    hidden_states_set.append(pickle.load(hidden_state_file))
                    hidden_states_set = np.vstack(hidden_states_set)
                    ind_end = len(hidden_states_set)
                    print("hidden_state_set shape: ",hidden_states_set.shape)
                    if self.dic_exp_conf["PRETRAIN"] or self.dic_exp_conf["AGGREGATE"]:
                        pass
                    else:
                        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                        hidden_states_after_forget = hidden_states_set[ind_sta: ind_end]
                        hidden_states_set = [np.array([hidden_states_after_forget[k] for k in self.sample_indexes])]
            except EOFError:
                hidden_state_file.close()
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("Fail to load hidden_states for inter\n")
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        return hidden_states_set

    def load_hidden_states(self): # hidden state is a list [#time, agent, # dim]
        hidden_states_set = []
        try:
            hidden_state_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_hidden_states.pkl"), "rb")
            try:
                while True:
                    hidden_states_set.append(pickle.load(hidden_state_file))
            except EOFError:
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("Fail to load hidden_states for inter\n")
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        return hidden_states_set


    def load_sample_with_forget(self, i):
        '''
        Load sample for each intersection, with forget
        :param i:
        :return: a list of samples with fixed indexes
        '''

        sample_set = []
        try:
            if self.dic_exp_conf["PRETRAIN"]:
                    sample_file = open(os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"],
                                                "train_round", "total_samples" + ".pkl"), "rb")
            elif self.dic_exp_conf["AGGREGATE"]:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_AGGREGATE_SAMPLES"],
                                                "aggregate_samples.pkl"), "rb")
            else:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            try:
                while True:
                    cur_round_sample_set = pickle.load(sample_file)
                    ind_end = len(cur_round_sample_set)
                    if self.dic_exp_conf["PRETRAIN"] or self.dic_exp_conf["AGGREGATE"]:
                        pass
                    else:
                        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                        memory_after_forget = cur_round_sample_set[ind_sta: ind_end]
                        # print("memory size after forget:", len(memory_after_forget))

                        # sample the memory
                        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
                        if self.sample_indexes is None:
                            self.sample_indexes = random.sample(range(len(memory_after_forget)), sample_size)
                        sample_set = [memory_after_forget[k] for k in self.sample_indexes]
                        # print("memory samples number:", sample_size)
                        # print(self.sample_indexes)
                    sample_set += cur_round_sample_set
            except EOFError:
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i %100 == 0:
            print("load_sample for inter {0}".format(i))
        return sample_set


    def load_sample_for_agents(self):
        # TODO should be number of agents
        start_time = time.time()
        print("Start load samples at", start_time)
        samples_gcn_df = []
        print("start get samples")
        get_samples_start_time = time.time()
        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            sample_set = self.load_sample(i)
            samples_set_df = pd.DataFrame.from_records(sample_set,columns= ['state','action','next_state','inst_reward','reward','time','generator'])
            samples_set_df['input'] = samples_set_df[['state','action','next_state','inst_reward','reward']].values.tolist()
            samples_set_df.drop(['state','action','next_state','inst_reward','reward','time','generator'], axis=1, inplace=True)
            # samples_set_df['inter_id'] = i
            samples_gcn_df.append(samples_set_df['input'])
            if i%100 == 0:
                print("inter {0} samples_set_df.shape: ".format(i), samples_set_df.shape)
        samples_gcn_df = pd.concat(samples_gcn_df, axis=1)
        print("samples_gcn_df.shape :", samples_gcn_df.shape)
        print("Getting samples time: ", time.time()-get_samples_start_time)

        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            sample_set_list = samples_gcn_df.values.tolist()
            self.agents[i].prepare_Xs_Y(sample_set_list, self.dic_exp_conf)
        
        prepared_Xs_Y = [self.agents[i].Y, self.agents[i].Xs, self.agents[i].Y_total]
        with open(os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'] + str('/train_round/'), "prepared_Xs_Y_generator_{0}.pkl".format(self.cnt_gen)),"wb+") as f:
            pickle.dump(prepared_Xs_Y, f, -1)

        print("------------------Load samples time: ", time.time()-start_time)

    def sample_set_to_sample_gcn_df(self,sample_set):
        print("make results")
        samples_set_df = pd.DataFrame.from_records(sample_set,columns= ['state','action','next_state','inst_reward','reward','time','generator'])
        samples_set_df = samples_set_df.set_index(['time','generator'])
        samples_set_df['input'] = samples_set_df[['state','action','next_state','inst_reward','reward']].values.tolist()
        samples_set_df.drop(['state','action','next_state','inst_reward','reward'], axis=1, inplace=True)
        self.sample_set_list.append(samples_set_df)


    def update_network(self,i):
        print('update agent %d'%i)
        self.agents[i].train_network(self.dic_exp_conf)
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            if self.dic_exp_conf["PRETRAIN"]:
                self.agents[i].q_network.save(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                             "{0}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0]))
                                             )
                shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                         "{0}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0])),
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            elif self.dic_exp_conf["AGGREGATE"]:
                self.agents[i].q_network.save("model/initial", "aggregate.h5")
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                self.agents[i].save_network("round_{0}".format(self.cnt_round))

        else:
            if self.dic_exp_conf["PRETRAIN"]:
                self.agents[i].q_network.save(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                             "{0}_inter_{1}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0],
                                                                       self.agents[i].intersection_id))
                                             )
                shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                         "{0}_inter_{1}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0],
                                                                   self.agents[i].intersection_id)),
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            elif self.dic_exp_conf["AGGREGATE"]:
                self.agents[i].q_network.save("model/initial", "aggregate_inter_{0}.h5".format(self.agents[i].intersection_id))
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(self.agents[i].intersection_id)))
            else:
                self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round,self.agents[i].intersection_id))

    def update_network_for_agents(self):
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            self.update_network(0)
        else:
            print("update_network_for_agents", self.dic_traffic_env_conf['NUM_AGENTS'])
            for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
                self.update_network(i)

class Updater2:

    def __init__(self, cnt_round, cnt_gen, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, config, q, best_round=None, bar_round=None):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.config = config
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None
        self.q = q
        
        self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = sum(self.dic_traffic_env_conf["maps_tl_nums"].values())

        for i in range(1):
            agent = CoLightAgent_cotrain(
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round, 
                best_round=None,
                intersection_id=str(i)
            )
            self.agents.append(agent) 
            
        #load prepared_XY for network update
        prepared_Xs_Y = []
        for cnt_gen in range(self.dic_exp_conf["NUM_MAPS"]):
            f_logging_data = open(os.path.join(self.dic_path['PATH_TO_WORK_DIRECTORY'] + str('/train_round/'), "prepared_Xs_Y_generator_{0}.pkl".format(cnt_gen)), "rb")
            prepared_Xs_Y.append(pickle.load(f_logging_data))
            
        Y = prepared_Xs_Y[0][0]
        Xs = prepared_Xs_Y[0][1]
        Y_total = prepared_Xs_Y[0][2]
        for cnt_gen in range(1, self.dic_exp_conf["NUM_MAPS"]):
            Y = np.hstack((Y, prepared_Xs_Y[cnt_gen][0]))
            Xs[0] = np.hstack((Xs[0], prepared_Xs_Y[cnt_gen][1][0]))
            Xs[1] = np.hstack((Xs[1], prepared_Xs_Y[cnt_gen][1][1]))
            Y_total[0] = np.hstack((Y_total[0], prepared_Xs_Y[cnt_gen][2][0]))
            Y_total[1] = np.concatenate((Y_total[1], prepared_Xs_Y[cnt_gen][2][1]),axis=2)
        
        self.agents[0].Y, self.agents[0].Xs, self.agents[0].Y_total = Y, Xs, Y_total
        self.agents[0].train_network(self.dic_exp_conf)
        
        self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round,self.agents[i].intersection_id))
        