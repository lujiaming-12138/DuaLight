import numpy as np 
import os 
import pickle  
from agent import Agent
import random 
import time
"""
Model for CoLight in paper "CoLight: Learning Network-level Cooperation for Traffic Signal
Control", in submission. 
"""
import keras
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge, Dot
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model, model_from_json, load_model
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, TensorBoard

# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# tf.set_random_seed(SEED)

class RepeatVector3D(Layer):
    def __init__(self,times,**kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.times, input_shape[1],input_shape[2])

    def call(self, inputs):
        #[batch,agent,dim]->[batch,1,agent,dim]
        #[batch,1,agent,dim]->[batch,agent,agent,dim]

        return K.tile(K.expand_dims(inputs,1),[1,self.times,1,1])


    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CoLightAgent(Agent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None, 
        best_round=None, bar_round=None,intersection_id="0"):
        """
        #1. compute the (dynamic) static Adjacency matrix, compute for each state
        -2. #neighbors: 5 (1 itself + W,E,S,N directions)
        -3. compute len_features
        -4. self.num_actions
        """
        super(CoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
        
        self.dic_agent_conf = dic_agent_conf
        self.att_regulatization=dic_agent_conf['att_regularization']
        self.CNN_layers=dic_agent_conf['CNN_layers']
        
        #TODO: n_agents should pass as parameter
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors=min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)
        self.vec=np.zeros((1,self.num_neighbors))
        self.vec[0][0]=1

        self.num_actions = 8#len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.len_feature=self.compute_len_feature()
        self.memory = self.build_memory()

        if cnt_round == 0: 
            # initialization
            self.q_network = self.build_network()
            if not os.path.exists(self.dic_path["PATH_TO_MODEL"]):
                try:
                    os.makedirs(self.dic_path["PATH_TO_MODEL"])
                except:
                    pass
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.q_network.load_weights(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)), 
                    by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            self.q_network = self.build_network()
            # f_logging_data = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "round_{0}_inter_{1}.pkl".format(cnt_round-1, self.intersection_id)), "rb")
            # weights = pickle.load(f_logging_data)
            # for i in range(len(weights)):
            #     np.save(f'{i}.npy',weights[i]) 
            self.q_network.load_weights(
                os.path.join(self.dic_path["PATH_TO_MODEL"], "round_{0}_inter_{1}.h5".format(cnt_round-1,intersection_id)), 
                by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)


        # decay the epsilon
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
            os.path.join(
                self.dic_path["PATH_TO_MODEL"], 
                "round_-1_inter_{0}.h5".format(intersection_id))):
            #the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f'%(cnt_round,self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])
        


    def compute_len_feature(self):
        from functools import reduce
        len_feature=tuple()
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "adjacency" in feature_name:
                continue
            elif "phase" in feature_name:
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            elif feature_name=="lane_num_vehicle":
                len_feature += (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)
        return sum(len_feature)

    """
    components of the network
    1. MLP encoder of features
    2. CNN layers
    3. q network
    """
    def MLP(self,In_0,layers=[128,128]):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,feature_dim]
        -outpout: [batch,#agents,128]
        """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index,layer_size in enumerate(layers):
            if layer_index==0:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(h)

        return h

    def gather_function(self, inputs):
        In_neighbor, neighbor_repr = inputs
        return tf.batch_gather(neighbor_repr, indices=tf.cast(In_neighbor, tf.int32))
    
    def reshape_function(self, inputs):
        neighbor_repr = inputs
        return tf.reshape(neighbor_repr, (-1, neighbor_repr.shape[-3], neighbor_repr.shape[-2], neighbor_repr.shape[-1]))

    def MultiHeadsAttModel(self,In_agent,In_neighbor,l=5, d=128, dv=16, dout=128, nv = 8,suffix=-1):
        """
        input:[bacth,agent,128]
        output:
        -hidden state: [batch,agent,32]
        -attention: [batch,agent,neighbor]
        """
        """
        agent repr
        """
        print("In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv", In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv)
        #[batch,agent,dim]->[batch,agent,1,dim]
        agent_repr=Reshape((self.num_agents,1,d))(In_agent)

        """
        neighbor repr
        """
        #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
        print("neighbor_repr.shape", neighbor_repr.shape)
        #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        
        #
        #neighbor_repr=Lambda(lambda x:K.batch_dot(x[0],x[1]))([In_neighbor,neighbor_repr])

        neighbor_repr = Lambda(self.gather_function)([In_neighbor, neighbor_repr])
        #neighbor_repr = Lambda(self.reshape_function)(neighbor_repr)
        
        print("neighbor_repr.shape", neighbor_repr.shape)
        """
        attention computation
        """
        #multi-head
        #[batch,agent,1,dim]->[batch,agent,1,dv*nv]
        agent_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='agent_repr_%d'%suffix)(agent_repr)
        #[batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
        agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
        agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)
        #agent_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,1,dv,nv)),(0,1,4,2,3)))(agent_repr_head)
        #[batch,agent,neighbor,dim]->[batch,agent,neighbor,dv*nv]

        neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
        #[batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        print("DEBUG",neighbor_repr_head.shape)
        print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.num_neighbors,dv,nv)
        neighbor_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_repr_head)
        neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_repr_head)
        #neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,self.num_neighbors,dv,nv)),(0,1,4,2,3)))(neighbor_repr_head)        
        #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
        att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head])
        #[batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
        att_record=Reshape((self.num_agents,nv,self.num_neighbors))(att)


        #self embedding again
        neighbor_hidden_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_hidden_repr_%d'%suffix)(neighbor_repr)
        neighbor_hidden_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_hidden_repr_head)
        neighbor_hidden_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_hidden_repr_head)
        out=Lambda(lambda x:K.mean(K.batch_dot(x[0],x[1]),axis=2))([att,neighbor_hidden_repr_head])
        out=Reshape((self.num_agents,dv))(out)
        out = Dense(dout, activation = "relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
        return out,att_record





    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        #l = to_categorical(adjacency_index_new,num_classes=self.num_agents)
        l = adjacency_index_new
        return l

    def action_att_predict(self,state,total_features=[],total_adjs=[],bar=False):
        #state:[batch,agent,features and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
        if total_features==[] and total_adjs==[]:
            total_features,total_adjs=list(),list()
            for i in range(batch_size): 
                feature=[]
                adj=[] 
                for j in range(self.num_agents):
                    observation=[]
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name])==1:
                                #choose_action
                                observation.extend(self.dic_traffic_env_conf["phase_expansion"]
                                                            [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name=="lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            #feature:[agents,feature]
            total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))
            #adj:[agent,neighbors]   
        if bar:
            all_output= self.q_network_bar.predict([total_features,total_adjs])
        else:
            all_output= self.q_network.predict([total_features,total_adjs])
        action,attention =all_output[0],all_output[1]

        #mask of action in sumo
        action_ = np.exp(action)
        for i in range(action.shape[1]):
            tl_id = state[0][i]['id']
            #print(tl_id,self.dic_agent_conf['tl_unavav_index'][tl_id].unava_index)
            for unindex in self.dic_agent_conf['tl_unavav_index'][tl_id]:
                action_[0][i][unindex] = -1

        #out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        if len(action)>1:
            return total_features,total_adjs,action,attention

        #[batch,agent,1]
        max_action=np.expand_dims(np.argmax(action_,axis=-1),axis=-1)
        
        #mask of random_action in sumo
        random_action = []
        for j in range(action.shape[1]):
            tl_id = state[0][j]['id']
            ava_list = [i for i in range(8) if i not in self.dic_agent_conf['tl_unavav_index'][tl_id]]
            random_action.append(np.random.choice(ava_list, 1))
        random_action=np.reshape(random_action,(1,self.num_agents,1))
        
        #[batch,agent,2]
        possible_action=np.concatenate([max_action,random_action],axis=-1)
        selection=np.random.choice(
            [0,1],
            size=batch_size*self.num_agents,
            p=[1-self.dic_agent_conf["EPSILON"],self.dic_agent_conf["EPSILON"]])
        act=possible_action.reshape((batch_size*self.num_agents,2))[np.arange(batch_size*self.num_agents),selection]
        act=np.reshape(act,(batch_size,self.num_agents))
        return act,attention


    def choose_action(self, count, state):

        ''' 
        choose the best action for current state 
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        '''
        act,attention=self.action_att_predict([state])
        return act[0],attention[0] 


    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """
        
        """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)

        _state = []
        _next_state = []
        _action=[]
        _reward=[]

        for i in range(len(sample_slice)):  
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)


        #target: [#agents,#samples,#num_actions]    
        _features,_adjs,q_values,_=self.action_att_predict(_state)   
        _next_features,_next_adjs,_,attention= self.action_att_predict(_next_state)
        #target_q_values:[batch,agent,action]
        _,_,target_q_values,_= self.action_att_predict(
            _next_state,
            total_features=_next_features,
            total_adjs=_next_adjs,
            bar=True)

        for i in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[i][j][_action[i][j]] = _reward[i][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])


        #self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features,_adjs]
        self.Y=q_values.copy()
        self.Y_total = [q_values.copy()]
        self.Y_total.append(attention)
        return 

    #TODO: MLP_layers should be defined in the conf file
    #TODO: CNN_layers should be defined in the conf file
    #TODO: CNN_heads should be defined in the conf file
    #TODO: Output_layers should be degined in the conf file
    def build_network(
        self,
        MLP_layers=[32,32], 
        # CNN_layers=[[32,32]],#[[4,32],[4,32]],
        # CNN_heads=[1],#[8,8],
        Output_layers=[]):
        CNN_layers=self.CNN_layers 
        CNN_heads=[1]*len(CNN_layers)
        """
        layer definition
        """
        start_time=time.time()
        assert len(CNN_layers)==len(CNN_heads)

        In=list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,neighbors]
        In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        In.append(Input(shape=(self.num_agents,self.num_neighbors),name="adjacency_matrix"))


        Input_end_time=time.time()
        """
        Currently, the MLP layer 
        -input: [batch,agent,feature_dim]
        -outpout: [#agent,batch,128]
        """
        feature=self.MLP(In[0],MLP_layers)

        Embedding_end_time=time.time()


        #TODO: remove the dense setting
        #feature:[batch,agents,feature_dim]
        att_record_all_layers=list()
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:",CNN_heads[CNN_layer_index])
            if CNN_layer_index==0:
                h,att_record=self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            else:
                h,att_record=self.MultiHeadsAttModel(
                    h,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            att_record_all_layers.append(att_record)

        if len(CNN_layers)>1:
            att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        else:
            att_record_all_layers=att_record_all_layers[0]

        att_record_all_layers=Reshape(
            (len(CNN_layers),self.num_agents,CNN_heads[-1],self.num_neighbors)
            )(att_record_all_layers)

        
        #TODO remove dense net
        for layer_index,layer_size in enumerate(Output_layers):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        #action prediction layer
        #[batch,agent,32]->[batch,agent,action]
        out = Dense(self.num_actions,kernel_initializer='random_normal',name='action_layer')(h)
        #out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model=Model(inputs=In,outputs=[out,att_record_all_layers])

        if self.att_regulatization:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"],'kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])
        # model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
        model.summary()
        network_end=time.time()
        print('build_Input_end_time：',Input_end_time-start_time)
        print('embedding_time:',Embedding_end_time-Input_end_time)
        print('total time:',network_end-start_time)
        return model

    def build_memory(self):

        return []

    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        # hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
        #                           shuffle=False,
        #                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
        hist = self.q_network.fit(self.Xs, self.Y_total, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3,
                                  callbacks=[early_stopping,TensorBoard(log_dir='./temp.tensorboard')])

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D, 'gather_function':self.gather_function})
        network.set_weights(network_weights)

        if self.att_regulatization:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"] for i in range(self.num_agents)]+['kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])

        return network

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        #pip3 install h5py==2.10.0
        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name) 

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))



class CoLightAgent_cotrain(Agent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None, 
        best_round=None, bar_round=None,intersection_id="0"):
        super(CoLightAgent_cotrain, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
        
        self.dic_agent_conf = dic_agent_conf
        self.att_regulatization=dic_agent_conf['att_regularization']
        self.CNN_layers=dic_agent_conf['CNN_layers']
        
        #TODO: n_agents should pass as parameter
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors=min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)
        self.vec=np.zeros((1,self.num_neighbors))
        self.vec[0][0]=1

        self.num_actions = 8#len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.len_feature=self.compute_len_feature()
        self.memory = self.build_memory()
        
        self.slice_start, self.slice_end = 0, 0

        if cnt_round == 0: 
            # initialization
            self.q_network = self.build_network()
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            self.q_network = self.build_network()
            self.q_network.load_weights(
                os.path.join(self.dic_path["PATH_TO_MODEL"], "round_{0}_inter_{1}.h5".format(cnt_round-1,intersection_id)),  
                by_name=True)
            # print('init q_bar load')
            self.q_network_bar = self.build_network_from_copy(self.q_network)

        # decay the epsilon
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
            os.path.join(
                self.dic_path["PATH_TO_MODEL"], 
                "round_-1_inter_{0}.h5".format(intersection_id))):
            #the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f'%(cnt_round,self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])
        


    def compute_len_feature(self):
        from functools import reduce
        len_feature=tuple()
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "adjacency" in feature_name:
                continue
            elif "phase" in feature_name:
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            elif feature_name=="lane_num_vehicle":
                len_feature += (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)
        return sum(len_feature)

    """
    components of the network
    1. MLP encoder of features
    2. CNN layers
    3. q network
    """
    def MLP(self,In_0,layers=[128,128]):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,feature_dim]
        -outpout: [batch,#agents,128]
        """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index,layer_size in enumerate(layers):
            if layer_index==0:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(h)

        return h

    def gather_function(self, inputs):
        In_neighbor, neighbor_repr = inputs
        return tf.batch_gather(neighbor_repr, indices=tf.cast(In_neighbor, tf.int32))
    
    def slice_function1(self, inputs):
        neighbor_repr = inputs
        return neighbor_repr[:, self.slice_start:self.slice_end, self.slice_start:self.slice_end, :]
    
    def slice_function2(self, inputs):
        In_neighbor = inputs
        return In_neighbor[:, self.slice_start:self.slice_end, :]

    def MultiHeadsAttModel(self,In_agent,In_neighbor,l=5, d=128, dv=16, dout=128, nv = 8,suffix=-1):
        """
        input:[bacth,agent,128]
        output:
        -hidden state: [batch,agent,32]
        -attention: [batch,agent,neighbor]
        """
        """
        agent repr
        """
        print("In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv", In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv)
        #[batch,agent,dim]->[batch,agent,1,dim]
        agent_repr=Reshape((self.num_agents,1,d))(In_agent)

        """
        neighbor repr
        """
        #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
        print("neighbor_repr.shape", neighbor_repr.shape)
        #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]

        self.slice_start = 0
        neighbor_repr_list = []
        for cnt_gen in range(len(self.dic_traffic_env_conf["maps_tl_nums"])):
            self.slice_end = self.slice_start + self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen]
            # slice_neighbor_repr1 = Lambda(lambda x: x[:, slice_start:slice_end, :, :])(neighbor_repr)
            # slice_neighbor_repr2 = Lambda(lambda x: x[:, :, slice_start:slice_end, :])(slice_neighbor_repr1)
            # slice_In_neighbor = Lambda(lambda x: x[:, slice_start:slice_end, :])(In_neighbor)
            slice_neighbor_repr = Lambda(self.slice_function1, output_shape=(self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen],self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen], d))(neighbor_repr)
            slice_In_neighbor = Lambda(self.slice_function2, output_shape=(self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen],l))(In_neighbor)
            neighbor_repr_list.append(Lambda(self.gather_function)([slice_In_neighbor, slice_neighbor_repr]))
            self.slice_start = self.slice_end 
        neighbor_repr = Concatenate(axis=1)(neighbor_repr_list)
            
        print("neighbor_repr.shape", neighbor_repr.shape)
        """
        attention computation
        """
        #multi-head
        #[batch,agent,1,dim]->[batch,agent,1,dv*nv]
        agent_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='agent_repr_%d'%suffix)(agent_repr)
        #[batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
        agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
        agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)
        #agent_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,1,dv,nv)),(0,1,4,2,3)))(agent_repr_head)
        #[batch,agent,neighbor,dim]->[batch,agent,neighbor,dv*nv]

        neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
        #[batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        print("DEBUG",neighbor_repr_head.shape)
        print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.num_neighbors,dv,nv)
        neighbor_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_repr_head)
        neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_repr_head)
        #neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,self.num_neighbors,dv,nv)),(0,1,4,2,3)))(neighbor_repr_head)        
        #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
        att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head])
        #[batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
        att_record=Reshape((self.num_agents,nv,self.num_neighbors))(att)


        #self embedding again
        neighbor_hidden_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_hidden_repr_%d'%suffix)(neighbor_repr)
        neighbor_hidden_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_hidden_repr_head)
        neighbor_hidden_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_hidden_repr_head)
        out=Lambda(lambda x:K.mean(K.batch_dot(x[0],x[1]),axis=2))([att,neighbor_hidden_repr_head])
        out=Reshape((self.num_agents,dv))(out)
        out = Dense(dout, activation = "relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
        return out,att_record





    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        #l = to_categorical(adjacency_index_new,num_classes=self.num_agents)
        l = adjacency_index_new
        return l

    def action_att_predict(self,state,total_features=[],total_adjs=[],bar=False):
        #state:[batch,agent,features and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
        if total_features==[] and total_adjs==[]:
            total_features,total_adjs=list(),list()
            for i in range(batch_size): 
                feature=[]
                adj=[] 
                for j in range(self.num_agents):
                    observation=[]
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name])==1:
                                #choose_action
                                observation.extend(self.dic_traffic_env_conf["phase_expansion"]
                                                            [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name=="lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            #feature:[agents,feature]
            total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))
            #adj:[agent,neighbors]   
        if bar:
            all_output= self.q_network_bar.predict([total_features,total_adjs])
        else:
            all_output= self.q_network.predict([total_features,total_adjs])
        action,attention =all_output[0],all_output[1]

        #mask of action in sumo
        action_ = np.exp(action)
        for i in range(action.shape[1]):
            tl_id = state[0][i]['id']
            #print(tl_id,self.dic_agent_conf['tl_unavav_index'][tl_id].unava_index)
            for unindex in self.dic_agent_conf['tl_unavav_index'][tl_id]:
                action_[0][i][unindex] = -1

        #out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        if len(action)>1:
            return total_features,total_adjs,action,attention

        #[batch,agent,1]
        max_action=np.expand_dims(np.argmax(action_,axis=-1),axis=-1)
        
        #mask of random_action in sumo
        random_action = []
        for j in range(action.shape[1]):
            tl_id = state[0][j]['id']
            ava_list = [i for i in range(8) if i not in self.dic_agent_conf['tl_unavav_index'][tl_id]]
            random_action.append(np.random.choice(ava_list, 1))
        random_action=np.reshape(random_action,(1,self.num_agents,1))
        
        #[batch,agent,2]
        possible_action=np.concatenate([max_action,random_action],axis=-1)
        selection=np.random.choice(
            [0,1],
            size=batch_size*self.num_agents,
            p=[1-self.dic_agent_conf["EPSILON"],self.dic_agent_conf["EPSILON"]])
        act=possible_action.reshape((batch_size*self.num_agents,2))[np.arange(batch_size*self.num_agents),selection]
        act=np.reshape(act,(batch_size,self.num_agents))
        return act,attention


    def choose_action(self, count, state):

        ''' 
        choose the best action for current state 
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        '''
        act,attention=self.action_att_predict([state])
        return act[0],attention[0] 


    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """
        
        """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)

        _state = []
        _next_state = []
        _action=[]
        _reward=[]

        for i in range(len(sample_slice)):  
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)


        #target: [#agents,#samples,#num_actions]    
        _features,_adjs,q_values,_=self.action_att_predict(_state)   
        _next_features,_next_adjs,_,attention= self.action_att_predict(_next_state)
        #target_q_values:[batch,agent,action]
        _,_,target_q_values,_= self.action_att_predict(
            _next_state,
            total_features=_next_features,
            total_adjs=_next_adjs,
            bar=True)

        for i in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[i][j][_action[i][j]] = _reward[i][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])


        #self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features,_adjs]
        self.Y=q_values.copy()
        self.Y_total = [q_values.copy()]
        self.Y_total.append(attention)
        return 

    #TODO: MLP_layers should be defined in the conf file
    #TODO: CNN_layers should be defined in the conf file
    #TODO: CNN_heads should be defined in the conf file
    #TODO: Output_layers should be degined in the conf file
    def build_network(
        self,
        MLP_layers=[32,32], 
        # CNN_layers=[[32,32]],#[[4,32],[4,32]],
        # CNN_heads=[1],#[8,8],
        Output_layers=[]):
        CNN_layers=self.CNN_layers 
        CNN_heads=[1]*len(CNN_layers)
        """
        layer definition
        """
        start_time=time.time()
        assert len(CNN_layers)==len(CNN_heads)

        In=list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,neighbors]
        In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        In.append(Input(shape=(self.num_agents,self.num_neighbors),name="adjacency_matrix"))


        Input_end_time=time.time()
        """
        Currently, the MLP layer 
        -input: [batch,agent,feature_dim]
        -outpout: [#agent,batch,128]
        """
        feature=self.MLP(In[0],MLP_layers)

        Embedding_end_time=time.time()


        #TODO: remove the dense setting
        #feature:[batch,agents,feature_dim]
        att_record_all_layers=list()
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:",CNN_heads[CNN_layer_index])
            if CNN_layer_index==0:
                h,att_record=self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            else:
                h,att_record=self.MultiHeadsAttModel(
                    h,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            att_record_all_layers.append(att_record)

        if len(CNN_layers)>1:
            att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        else:
            att_record_all_layers=att_record_all_layers[0]

        att_record_all_layers=Reshape(
            (len(CNN_layers),self.num_agents,CNN_heads[-1],self.num_neighbors)
            )(att_record_all_layers)

        
        #TODO remove dense net
        for layer_index,layer_size in enumerate(Output_layers):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        #action prediction layer
        #[batch,agent,32]->[batch,agent,action]
        out = Dense(self.num_actions,kernel_initializer='random_normal',name='action_layer')(h)
        #out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model=Model(inputs=In,outputs=[out,att_record_all_layers])

        if self.att_regulatization:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"],'kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])
        # model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
        model.summary()
        network_end=time.time()
        print('build_Input_end_time：',Input_end_time-start_time)
        print('embedding_time:',Embedding_end_time-Input_end_time)
        print('total time:',network_end-start_time)
        return model

    def build_memory(self):

        return []

    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        # hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
        #                           shuffle=False,
        #                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
        hist = self.q_network.fit(self.Xs, self.Y_total, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3,
                                  callbacks=[early_stopping,TensorBoard(log_dir='./temp.tensorboard')])

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D,
                                                                     'slice_function1':self.slice_function1, 
                                                                     'slice_function2':self.slice_function2, 
                                                                     'gather_function':self.gather_function})
        network.set_weights(network_weights)

        if self.att_regulatization:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"] for i in range(self.num_agents)]+['kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])

        return network

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        #pip3 install h5py==2.10.0
        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={"RepeatVector3D": RepeatVector3D,
                                                                     'slice_function1':self.slice_function1, 
                                                                     'slice_function2':self.slice_function2, 
                                                                     'gather_function':self.gather_function})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name) 

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        #self.q_network.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        
        # weights = self.q_network.get_weights()
        # with open(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name),'wb') as f:
        #     pickle.dump(weights, f, -1)

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        
        

class CoLightAgent_cross_scenario(Agent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None, 
        best_round=None, bar_round=None,intersection_id="0"):
        super(CoLightAgent_cross_scenario, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
        
        self.dic_agent_conf = dic_agent_conf
        self.att_regulatization=dic_agent_conf['att_regularization']
        self.CNN_layers=dic_agent_conf['CNN_layers']
        
        #TODO: n_agents should pass as parameter
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors=min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)
        self.vec=np.zeros((1,self.num_neighbors))
        self.vec[0][0]=1

        self.num_actions = 8#len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.len_feature=self.compute_len_feature()
        self.memory = self.build_memory()
        
        self.slice_start, self.slice_end = 0, 0

        if cnt_round == 0: 
            # initialization
            self.q_network = self.build_network()
            #self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            self.q_network = self.build_network()
            self.q_network.load_weights(
                os.path.join(self.dic_path["PATH_TO_MODEL"], "round_{0}_inter_{1}.h5".format(cnt_round-1,intersection_id)),  
                by_name=True)
            # print('init q_bar load')
            #self.q_network_bar = self.build_network_from_copy(self.q_network)

        # decay the epsilon
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
            os.path.join(
                self.dic_path["PATH_TO_MODEL"], 
                "round_-1_inter_{0}.h5".format(intersection_id))):
            #the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f'%(cnt_round,self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])
        


    def compute_len_feature(self):
        from functools import reduce
        len_feature=tuple()
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "adjacency" in feature_name:
                continue
            elif "phase" in feature_name:
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            elif feature_name=="lane_num_vehicle":
                len_feature += (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)
        return sum(len_feature)

    """
    components of the network
    1. MLP encoder of features
    2. CNN layers
    3. q network
    """
    def MLP(self,In_0,layers=[128,128]):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,feature_dim]
        -outpout: [batch,#agents,128]
        """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index,layer_size in enumerate(layers):
            if layer_index==0:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(h)

        return h

    def gather_function(self, inputs):
        In_neighbor, neighbor_repr = inputs
        return tf.batch_gather(neighbor_repr, indices=tf.cast(In_neighbor, tf.int32))
    
    def slice_function1(self, inputs):
        neighbor_repr = inputs
        return neighbor_repr[:, self.slice_start:self.slice_end, self.slice_start:self.slice_end, :]
    
    def slice_function2(self, inputs):
        In_neighbor = inputs
        return In_neighbor[:, self.slice_start:self.slice_end, :]

    def cosine_similarity(self, input):
        x, neighbor_repr = input
        # 将输入张量归一化为单位向量
        norm_inputs = Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(x)
        # 计算余弦相似度矩阵
        dots = Dot(axes=-1)([norm_inputs, norm_inputs])
        dots_part1 = dots[:, self.slice_start:self.slice_end, :self.slice_start]
        dots_part2 = dots[:, self.slice_start:self.slice_end, self.slice_end:]
        cut_dots = Concatenate(axis=2)([dots_part1, dots_part2])
        values, top_k_indices = tf.nn.top_k(cut_dots, k=5)
        neighbor_repr_part1 = neighbor_repr[:, self.slice_start:self.slice_end, :self.slice_start]
        neighbor_repr_part2 = neighbor_repr[:, self.slice_start:self.slice_end, self.slice_end:]        
        cut_neighbor_repr = Concatenate(axis=2)([neighbor_repr_part1, neighbor_repr_part2])
        return Lambda(self.gather_function)([top_k_indices, cut_neighbor_repr])

    def MultiHeadsAttModel(self,In_agent,In_neighbor,l=5, d=128, dv=16, dout=128, nv = 8,suffix=-1):
        """
        input:[bacth,agent,128]
        output:
        -hidden state: [batch,agent,32]
        -attention: [batch,agent,neighbor]
        """
        """
        agent repr
        """
        
        print("In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv", In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv)
        #[batch,agent,dim]->[batch,agent,1,dim]
        agent_repr=Reshape((self.num_agents,1,d))(In_agent)

        """
        neighbor repr
        """
        #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
        print("neighbor_repr.shape", neighbor_repr.shape)
        #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]

        self.slice_start = 0
        neighbor_repr_list = []
        for cnt_gen in range(len(self.dic_traffic_env_conf["maps_tl_nums"])):
            self.slice_end = self.slice_start + self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen]
            slice_neighbor_repr = Lambda(self.slice_function1, output_shape=(self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen],self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen], d))(neighbor_repr)
            slice_In_neighbor = Lambda(self.slice_function2, output_shape=(self.dic_traffic_env_conf["maps_tl_nums"][cnt_gen],l))(In_neighbor)
            
            sim_neighbor_repr = Lambda(self.cosine_similarity)([In_agent, neighbor_repr])
            real_neighbor_repr = Lambda(self.gather_function)([slice_In_neighbor, slice_neighbor_repr])
            neighbor_repr_list.append(Concatenate(axis=2)([sim_neighbor_repr, real_neighbor_repr]))
            self.slice_start = self.slice_end 
        neighbor_repr = Concatenate(axis=1)(neighbor_repr_list)
        
            
        print("neighbor_repr.shape", neighbor_repr.shape)
        """
        attention computation
        """
        self.num_neighbors = 10
        #multi-head
        #[batch,agent,1,dim]->[batch,agent,1,dv*nv]
        agent_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='agent_repr_%d'%suffix)(agent_repr)
        #[batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
        agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
        agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)
        #agent_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,1,dv,nv)),(0,1,4,2,3)))(agent_repr_head)
        #[batch,agent,neighbor,dim]->[batch,agent,neighbor,dv*nv]

        neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
        #[batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        print("DEBUG",neighbor_repr_head.shape)
        print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.num_neighbors,dv,nv)
        neighbor_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_repr_head)
        neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_repr_head)
        #neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,self.num_neighbors,dv,nv)),(0,1,4,2,3)))(neighbor_repr_head)        
        #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
        att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head])
        #[batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
        att_record=Reshape((self.num_agents,nv,self.num_neighbors))(att)


        #self embedding again
        neighbor_hidden_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_hidden_repr_%d'%suffix)(neighbor_repr)
        neighbor_hidden_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_hidden_repr_head)
        neighbor_hidden_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_hidden_repr_head)
        out=Lambda(lambda x:K.mean(K.batch_dot(x[0],x[1]),axis=2))([att,neighbor_hidden_repr_head])
        out=Reshape((self.num_agents,dv))(out)
        out = Dense(dout, activation = "relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
        return out,att_record





    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        #l = to_categorical(adjacency_index_new,num_classes=self.num_agents)
        l = adjacency_index_new
        return l

    def action_att_predict(self,state,total_features=[],total_adjs=[],bar=False):
        #state:[batch,agent,features and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
        if total_features==[] and total_adjs==[]:
            total_features,total_adjs=list(),list()
            for i in range(batch_size): 
                feature=[]
                adj=[] 
                for j in range(self.num_agents):
                    observation=[]
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name])==1:
                                #choose_action
                                observation.extend(self.dic_traffic_env_conf["phase_expansion"]
                                                            [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name=="lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            #feature:[agents,feature]
            total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))
            #adj:[agent,neighbors]   
        if bar:
            all_output= self.q_network_bar.predict([total_features,total_adjs])
        else:
            all_output= self.q_network.predict([total_features,total_adjs])
        action,attention =all_output[0],all_output[1]

        #mask of action in sumo
        action_ = np.exp(action)
        for i in range(action.shape[1]):
            tl_id = state[0][i]['id']
            #print(tl_id,self.dic_agent_conf['tl_unavav_index'][tl_id].unava_index)
            for unindex in self.dic_agent_conf['tl_unavav_index'][tl_id]:
                action_[0][i][unindex] = -1

        #out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        if len(action)>1:
            return total_features,total_adjs,action,attention

        #[batch,agent,1]
        max_action=np.expand_dims(np.argmax(action_,axis=-1),axis=-1)
        
        #mask of random_action in sumo
        random_action = []
        for j in range(action.shape[1]):
            tl_id = state[0][j]['id']
            ava_list = [i for i in range(8) if i not in self.dic_agent_conf['tl_unavav_index'][tl_id]]
            random_action.append(np.random.choice(ava_list, 1))
        random_action=np.reshape(random_action,(1,self.num_agents,1))
        
        #[batch,agent,2]
        possible_action=np.concatenate([max_action,random_action],axis=-1)
        selection=np.random.choice(
            [0,1],
            size=batch_size*self.num_agents,
            p=[1-self.dic_agent_conf["EPSILON"],self.dic_agent_conf["EPSILON"]])
        act=possible_action.reshape((batch_size*self.num_agents,2))[np.arange(batch_size*self.num_agents),selection]
        act=np.reshape(act,(batch_size,self.num_agents))
        return act,attention


    def choose_action(self, count, state):

        ''' 
        choose the best action for current state 
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        '''
        act,attention=self.action_att_predict([state])
        return act[0],attention[0] 


    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """
        
        """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)

        _state = []
        _next_state = []
        _action=[]
        _reward=[]

        for i in range(len(sample_slice)):  
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)


        #target: [#agents,#samples,#num_actions]    
        _features,_adjs,q_values,_=self.action_att_predict(_state)   
        _next_features,_next_adjs,_,attention= self.action_att_predict(_next_state)
        #target_q_values:[batch,agent,action]
        _,_,target_q_values,_= self.action_att_predict(
            _next_state,
            total_features=_next_features,
            total_adjs=_next_adjs,
            bar=True)

        for i in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[i][j][_action[i][j]] = _reward[i][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])


        #self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features,_adjs]
        self.Y=q_values.copy()
        self.Y_total = [q_values.copy()]
        self.Y_total.append(attention)
        return 

    #TODO: MLP_layers should be defined in the conf file
    #TODO: CNN_layers should be defined in the conf file
    #TODO: CNN_heads should be defined in the conf file
    #TODO: Output_layers should be degined in the conf file
    def build_network(
        self,
        MLP_layers=[32,32], 
        # CNN_layers=[[32,32]],#[[4,32],[4,32]],
        # CNN_heads=[1],#[8,8],
        Output_layers=[]):
        CNN_layers=self.CNN_layers 
        CNN_heads=[1]*len(CNN_layers)
        """
        layer definition
        """
        start_time=time.time()
        assert len(CNN_layers)==len(CNN_heads)

        In=list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,neighbors]
        In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        In.append(Input(shape=(self.num_agents,self.num_neighbors),name="adjacency_matrix"))


        Input_end_time=time.time()
        """
        Currently, the MLP layer 
        -input: [batch,agent,feature_dim]
        -outpout: [#agent,batch,128]
        """
        feature=self.MLP(In[0],MLP_layers)

        Embedding_end_time=time.time()


        #TODO: remove the dense setting
        #feature:[batch,agents,feature_dim]
        att_record_all_layers=list()
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:",CNN_heads[CNN_layer_index])
            if CNN_layer_index==0:
                h,att_record=self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            else:
                h,att_record=self.MultiHeadsAttModel(
                    h,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            att_record_all_layers.append(att_record)

        if len(CNN_layers)>1:
            att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        else:
            att_record_all_layers=att_record_all_layers[0]

        att_record_all_layers=Reshape(
            (len(CNN_layers),self.num_agents,CNN_heads[-1],self.num_neighbors)
            )(att_record_all_layers)

        
        #TODO remove dense net
        for layer_index,layer_size in enumerate(Output_layers):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        #action prediction layer
        #[batch,agent,32]->[batch,agent,action]
        out = Dense(self.num_actions,kernel_initializer='random_normal',name='action_layer')(h)
        #out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model=Model(inputs=In,outputs=[out,att_record_all_layers])

        if self.att_regulatization:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"],'kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])
        # model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
        model.summary()
        network_end=time.time()
        print('build_Input_end_time：',Input_end_time-start_time)
        print('embedding_time:',Embedding_end_time-Input_end_time)
        print('total time:',network_end-start_time)
        return model

    def build_memory(self):

        return []

    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        # hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
        #                           shuffle=False,
        #                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
        hist = self.q_network.fit(self.Xs, self.Y_total, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3,
                                  callbacks=[early_stopping,TensorBoard(log_dir='./temp.tensorboard')])

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure,
                                  custom_objects={"RepeatVector3D": RepeatVector3D,
                                                                     'slice_function1':self.slice_function1, 
                                                                     'slice_function2':self.slice_function2, 
                                                                     'gather_function':self.gather_function,
                                                                     'cosine_similarity':self.cosine_similarity})
        network.set_weights(network_weights)

        if self.att_regulatization:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"] for i in range(self.num_agents)]+['kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])

        return network

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        #pip3 install h5py==2.10.0
        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={"RepeatVector3D": RepeatVector3D,
                                                                     'slice_function1':self.slice_function1, 
                                                                     'slice_function2':self.slice_function2, 
                                                                     'gather_function':self.gather_function,
                                                                     'cosine_similarity':self.cosine_similarity})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name) 

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        #self.q_network.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        
        # weights = self.q_network.get_weights()
        # with open(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name),'wb') as f:
        #     pickle.dump(weights, f, -1)

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        
if __name__=='__main__':
    from sumo_files.env.sim_env import TSCSimulator
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

    name = "test"
    config = {
        'name': name,
        "agent": "colight",
        "sumocfg_file": [
                    # "sumo_files/scenarios/nanshan/osm.sumocfg",
                    # "sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg",
                    "sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg",
                    "sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg",
                    # "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
                    # 'sumo_files/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
                    # "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
                    ],
        "eval_sumocfg_file": [
                    # "sumo_files/scenarios/nanshan/osm.sumocfg",
                    "sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg",
                    # "sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg",
                    # "sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg",
                    # "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
                    # 'sumo_files/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
                    # "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
                    ],      
        'output_path': 'sumo_logs/{}/'.format(name),
        "action_type": "select_phase",
        "is_record": True,
        #'is_neighbor_reward': True,
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'state_key': ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num', 'pressure'],
        "model_save": {
            "path": "tsc/{}".format(name)
                        },
        'port_start': 15900,
        'is_dis': True,
        'is_adjacency_remove': False,
        'adjacency_top_k': TOP_K_ADJACENCY 
    }
    
    map_name = name
    DIC_PATH = {
    "PATH_TO_MODEL": f"model/{map_name}",
    "PATH_TO_WORK_DIRECTORY": f"records/{map_name}",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_ERROR": f"errors/{map_name}"
                    }
    config["sumocfg_file"] = config["eval_sumocfg_file"][0]
    if not os.path.exists(config['output_path']):
        try:
            os.makedirs(config['output_path'])
        except:
            pass
    env = TSCSimulator(config, config['port_start'])
    
    dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(env.all_tls)
    
    tl_unavav_index = {}
    for tl in env._crosses:
        tl_unavav_index[tl] = env._crosses[tl].unava_index
    DIC_COLIGHT_AGENT_CONF['tl_unavav_index'] = tl_unavav_index

    done = False
    state = env.reset()
    list_inter_log = [[] for i in range(len(env.all_tls))]
    agents = []
    for i in range(1):
        agent = CoLightAgent(
            dic_agent_conf=DIC_COLIGHT_AGENT_CONF,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path='model/trans_art',
            cnt_round=45, 
            best_round=None,
            intersection_id=str(i)
        )
        agents.append(agent)