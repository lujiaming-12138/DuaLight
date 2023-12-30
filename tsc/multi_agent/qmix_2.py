#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: qmix_2.py
@time: 2023/2/7
"""
# !/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: qmix
@time: 2023/1/9
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tsc.multi_agent.model import fc_block, GATLayer, batch
import numpy as np
from tsc.utils import ensure_shared_grads


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_size, fc_layer_size, rnn_hidden_dim, state_keys):
        super(RNN, self).__init__()
        self.state_keys = state_keys
        self.fc_car_num = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_queue_length = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_occupancy = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_flow = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_stop_car_num = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_pressure = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.reward = fc_block(1, 2, activation=nn.Sigmoid())
        # current phase
        self.current_phase_act = nn.Sigmoid()
        self.current_phase_embedding = nn.Embedding(2, 4)
        # mask
        self.mask_act = nn.Sigmoid()
        self.mask_embedding = nn.Embedding(2, 4)

        self.emb_padding_zero = torch.zeros([1, 246])
        self.act_padding_zero = torch.zeros([1, 8])
        self.reward_zero = torch.zeros(1, 2)

        self.fc1 = fc_block(224, rnn_hidden_dim - 10, activation=nn.ReLU())
        self.rnn = nn.GRUCell(rnn_hidden_dim * 5, rnn_hidden_dim * 4)

    def forward(self, input_list, hidden_state):
        input, pre_r, pre_a = input_list
        all_key_state = []
        bs = (list(input.values())[0].shape)[0]
        for i in self.state_keys:
            if i in ['current_phase', 'mask']:
                continue
            all_key_state.append(eval("self.fc_{}".format(i))(input[i].reshape(-1, 1)))
        if "current_phase" in self.state_keys:
            all_key_state.append(self.current_phase_act(
                self.current_phase_embedding(input['current_phase'])).reshape(-1, 4))
        all_key_state.append(self.mask_act(
            self.mask_embedding(input['mask'])).reshape(-1, 4))
        encode_emb = torch.cat(all_key_state, dim=1).reshape(bs, -1)  # bs, 224
        x = self.fc1(encode_emb)
        neighbor_index = input['neighbor_index']
        encode_emb_add_padding = torch.cat([x, self.emb_padding_zero], dim=0)
        neighbor_embed = encode_emb_add_padding[neighbor_index]
        agg_emd = torch.cat([x.view(bs, 1, -1), neighbor_embed], dim=1).view(bs, -1)
        pre_r = self.reward(pre_r)
        pre_r_add_padding = torch.cat([pre_r, self.reward_zero], dim=0)
        pre_a_add_padding = torch.cat([pre_a, self.act_padding_zero], dim=0)
        neighbor_pre_r, neighbor_pre_a = pre_r_add_padding[neighbor_index], pre_a_add_padding[neighbor_index]
        agg_r = torch.cat([pre_r.view(bs, 1, -1), neighbor_pre_r], dim=1).view(bs, -1)
        agg_a = torch.cat([pre_a.view(bs, 1, -1), neighbor_pre_a], dim=1).view(bs, -1)

        rnn_in = torch.cat([agg_emd, agg_r, agg_a], dim=1)

        h = self.rnn(rnn_in, hidden_state)
        return h


class BodyModel(nn.Module):
    def __init__(self):
        super(BodyModel, self).__init__()
        self.predict_fc1 = fc_block(1024, 256, activation=nn.ReLU())
        self.predict_fc2 = nn.Linear(256, 8)
        self.q_fc1 = fc_block(1024, 256, activation=nn.ReLU())
        self.q_fc2 = nn.Linear(512, 8)

    def forward(self, input):
        predict_1 = self.predict_fc1(input)
        q1 = self.q_fc1(input)
        q2 = self.q_fc2(torch.cat([predict_1, q1], dim=-1))
        predict2 = self.predict_fc2(predict_1)

        return q2, predict2


class QMixNet(nn.Module):
    def __init__(self, state_shape, hyper_hidden_dim, qmix_hidden_dim, two_hyper_layers=False):
        super(QMixNet, self).__init__()
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        self.state_shape = state_shape
        self.hyper_hidden_dim = hyper_hidden_dim
        self.qmix_hidden_dim = qmix_hidden_dim
        if two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, 5 * qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(state_shape, 5 * qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(state_shape, qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(state_shape, qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(qmix_hidden_dim, 1)
                                      )

    def forward(self, q_values,
                states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        steps_num = q_values.size(0)
        q_values = q_values.view(-1, 1, 5)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1, self.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, 5, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(steps_num, -1, 1)
        return q_total


class Runner:
    def __init__(self, env, config, shared_rnn, shared_body, shared_eval_qmix_net,
                 shared_target_rnn, shared_target_body, shared_target_qmix_net, optimizer):
        self.env = env
        # self.state = self.env.reset()
        self.all_tls = self.env.all_tls
        # self.pre_target_load = 0
        self.shared_rnn = shared_rnn
        self.shared_body = shared_body
        self.shared_qmix_net = shared_eval_qmix_net
        self.shared_target_rnn = shared_target_rnn
        self.shared_target_body = shared_target_body
        self.shared_target_qmix_net = shared_target_qmix_net
        self.optimizer = optimizer
        self.state_shape = config["qmix"]["state_shape"]
        self.is_last_action = config["qmix"]['is_last_action']
        input_shape = config["qmix"]["input_shape"]
        # 根据参数决定RNN的输入维度
        if config["qmix"]["is_last_action"]:
            input_shape += 8
        self.rnn = RNN(input_shape, 4, 256, config["environment"]['state_key'])
        self.body = BodyModel()
        self.eval_qmix_net = \
            QMixNet(config["qmix"]["state_shape"] * 5,
                    config["qmix"]["hyper_hidden_dim"],
                    config["qmix"]["qmix_hidden_dim"])  # 把agentsQ值加起来的网络
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(
            self.body.parameters()) + list(self.rnn.parameters())

        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QMIX')

        self.config = config
        self.episode_rewards = []

        self.n_actions = 8
        self.n_agents = 5
        self.config = config

        self.epsilon = config["qmix"]["epsilon"]
        self.anneal_epsilon = config["qmix"]["anneal_epsilon"]
        self.min_epsilon = config["qmix"]["min_epsilon"]

    def get_reward(self, reward):
        ans = []
        for i in self.all_tls:
            ans.append(sum(reward[i].values()))
        return ans

    # @torch.no_grad()
    def generate_mdps(self, unava_phase_index, num_steps=240, eval=False):
        if eval:
            self.rnn.eval()
            self.body.eval()
            ep = 0
        else:
            self.rnn.train()
            self.body.train()
            self.epsilon = \
                self.epsilon - self.anneal_epsilon if self.epsilon > self.min_epsilon else self.epsilon
            ep = self.epsilon
        o, u, r, u_onehot, terminate = [], [], [], [], []
        pred_os, q_values, target_pred_os, target_q_values = [], [], [], []
        all_embeddings, target_all_embeddings = [], []
        state = self.env.reset()
        # terminated = False
        step = 0
        self.init_hidden()
        pre_r = torch.zeros((len(self.env.all_tls), 1))
        pre_a = torch.zeros((len(self.env.all_tls), 8))
        pre_a[:, 0] = 1

        while step < num_steps:
            state = batch(state, self.config['environment']['state_key'], self.all_tls)
            actions, q_value, predict_o, target_q_value, \
                target_predict_o, all_embedding, target_all_embedding = \
                self.choose_action([state, pre_r, pre_a], unava_phase_index, ep)
            pred_os.append(predict_o)
            q_values.append(q_value)
            target_q_values.append(target_q_value)
            # generate onehot vector of th action
            actions_onehot = torch.zeros((len(self.all_tls), 8))
            actions_onehot[torch.arange(len(self.all_tls)), actions] = 1

            tl_action_select = {}
            for tl_index in range(len(self.all_tls)):
                tl_action_select[self.all_tls[tl_index]] = \
                    (self.env._crosses[self.all_tls[tl_index]].green_phases)[actions[tl_index]]
            next_state, reward, terminated, _all_reward = self.env.step(tl_action_select)
            reward = self.get_reward(reward)

            o.append(state)
            u.append(np.reshape(actions, [len(self.all_tls), 1]))
            u_onehot.append(actions_onehot.numpy())
            pre_a = actions_onehot.float()
            r.append([reward])
            pre_r = torch.FloatTensor(reward).reshape((len(self.env.all_tls), 1))
            all_embeddings.append(all_embedding)
            target_all_embeddings.append(target_all_embedding)
            terminate.append([terminated])
            state = deepcopy(next_state)
            step += 1
            if terminated:
                state = self.env.reset()
        self.env.terminate()
        state = batch(state, self.config['environment']['state_key'], self.all_tls)
        o.append(state)

        mdps = dict(o=o.copy(),
                    u=u.copy(),
                    r=r.copy(),
                    pred_os=pred_os.copy(),
                    q_values=q_values.copy(),
                    target_q_values=target_q_values.copy(),
                    u_onehot=u_onehot.copy(),
                    all_embeddings=all_embeddings.copy(),
                    target_all_embeddings=target_all_embeddings.copy(),
                    terminated=terminate.copy())
        # add episode dim
        for key in mdps.keys():
            if key not in ['q_values', 'target_q_values', 'all_embeddings', 'target_all_embeddings', 'pred_os']:
                mdps[key] = np.array([mdps[key]])
        return mdps, _all_reward

    def run(self, unava_phase_index, rollout_counter):
        self._copy_shared_model_to_local()
        mdps, all_reward = self.generate_mdps(unava_phase_index)
        loss = self.learn(mdps, rollout_counter, unava_phase_index)

        return loss, all_reward

    def _copy_shared_model_to_local(self):
        self.rnn.load_state_dict(self.shared_rnn.state_dict())
        self.body.load_state_dict(self.shared_body.state_dict())
        self.eval_qmix_net.load_state_dict(self.shared_qmix_net.state_dict())

    def evaluate(self, unava_phase_index):
        self._copy_shared_model_to_local()
        mdps, episode_reward = self.generate_mdps(unava_phase_index, eval=True)
        # self.init_context_hidden(mdps, evaluate=True)
        # _, episode_reward, _ = self.generate_episode(unava_phase_index, evaluate=True)
        return episode_reward

    def init_hidden(self):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((len(self.all_tls), 256 * 4))
        self.target_hidden = torch.zeros((len(self.all_tls), 256 * 4))

    def learn(self, batch_data, rollout_counter, unava_phase_index):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        if rollout_counter.get() % self.config["qmix"]["target_update_cycle"] == 0:
            self.shared_target_body.load_state_dict(self.shared_body.state_dict())
            self.shared_target_rnn.load_state_dict(self.shared_rnn.state_dict())
            self.shared_target_qmix_net.load_state_dict(self.shared_qmix_net.state_dict())
        # episode_num = batch_data['o'].shape[0]
        self.rnn.train()
        self.body.train()
        self.eval_qmix_net.train()

        # self.init_hidden()
        s, r, terminated = batch_data['o'][0], batch_data['r'][0], batch_data['terminated'][0]
        u = torch.tensor(batch_data["u"][0], dtype=torch.long)

        q_evals, q_targets, all_embeddings, all_embeddings_target, pred_o = \
            batch_data['q_values'], batch_data['target_q_values'], \
                batch_data['all_embeddings'], batch_data['target_all_embeddings'], \
                batch_data['pred_os']

        q_evals = torch.stack(q_evals, dim=0)
        q_targets = torch.stack(q_targets, dim=0)
        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings_target = torch.stack(all_embeddings_target, dim=0)

        if self.config["cuda"]:
            # s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            # s_next = s_next.cuda()
            terminated = terminated.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=2, index=u).squeeze(2)

        # 得到target_q
        for i in range(q_targets.shape[1]):
            q_targets[:, i, unava_phase_index[i]] = -float("inf")
        q_targets = q_targets.max(dim=2)[0]

        s_o = []
        for i in range(len(s)):
            s_o.append(s[i]['occupancy'])
        s_o = torch.cat(s_o).reshape(-1, len(self.all_tls), 8)
        pred_o = torch.cat(pred_o).reshape(-1, len(self.all_tls), 8)
        s_o = s_o[1:]
        # pred_o = pred_o[:-1]

        neighbor_index = torch.cat(
            [torch.arange(len(self.all_tls)).reshape(-1, 1), s[0]['neighbor_index']], 1)
        neighbor_index[neighbor_index == -1] = len(self.all_tls)
        neighbor_index = neighbor_index.expand(240, len(self.all_tls), 5)
        q_evals = torch.cat([q_evals, torch.zeros((240, 1))], dim=1)
        q_targets = torch.cat([q_targets, torch.zeros((240, 1))], dim=1)
        #
        q_evals = torch.gather(q_evals.unsqueeze(1).
                               expand(240, len(self.all_tls), len(self.all_tls) + 1),
                               dim=2, index=neighbor_index)
        q_targets = torch.gather(q_targets.unsqueeze(1).
                                 expand(240, len(self.all_tls), len(self.all_tls) + 1),
                                 dim=2, index=neighbor_index)

        paded_all_embeddings = torch.cat([all_embeddings, torch.zeros((240, 1, 1024))], dim=1)
        paded_all_embeddings = paded_all_embeddings.unsqueeze(1). \
            expand(240, len(self.all_tls), len(self.all_tls) + 1, 1024)
        all_embeddings = torch.gather(
            paded_all_embeddings, dim=2,
            index=neighbor_index.unsqueeze(3).repeat(1, 1, 1, 1024)). \
            reshape(240, len(self.all_tls), -1)
        paded_all_embeddings_target = torch.cat(
            [all_embeddings_target, torch.zeros((240, 1, 1024))], dim=1)
        paded_all_embeddings_target = paded_all_embeddings_target.unsqueeze(1). \
            expand(240, len(self.all_tls), len(self.all_tls) + 1, 1024)
        all_embeddings_target = torch.gather(
            paded_all_embeddings_target, dim=2,
            index=neighbor_index.unsqueeze(3).repeat(1, 1, 1, 1024)). \
            reshape(240, len(self.all_tls), -1)

        q_total_eval = self.eval_qmix_net(q_evals, all_embeddings)
        q_total_target = self.shared_target_qmix_net(q_targets, all_embeddings_target)

        targets = torch.FloatTensor(r).reshape(-1, len(self.all_tls), 1) + \
                  self.config["qmix"]["gamma"] * \
                  q_total_target * \
                  (1 - torch.tensor(terminated, dtype=torch.long)). \
                      unsqueeze(1).expand(q_total_target.shape)

        # td_error = (q_total_eval - targets.detach())
        # pred_error = (s_o - pred_o)
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = torch.nn.functional.mse_loss(q_total_eval, targets.detach())
        loss += torch.nn.functional.mse_loss(s_o.float(), pred_o.float()) * self.config["qmix"][
            "predict_o_loss"]
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.config["qmix"]["gradient_clip"])
        ensure_shared_grads(self.rnn, self.shared_rnn)
        ensure_shared_grads(self.body, self.shared_body)
        ensure_shared_grads(self.eval_qmix_net, self.shared_qmix_net)
        self.optimizer.step()

        return loss.detach()

    # def get_q_values(self, batch_data):
    #     steps_num = len(batch_data['o'][0])
    #     q_evals, q_targets = [], []
    #     pred_o, pred_o_target = [], []
    #     all_embeddings, all_embeddings_target = [], []
    #     for transition_idx in range(steps_num):
    #         inputs, inputs_next = batch_data["o"][0][transition_idx], batch_data["o_next"][0][
    #             transition_idx]
    #         # inputs, inputs_next = self._get_inputs(batch_data, transition_idx)  # 给obs加last_action、agent_id
    #         # for key in batch_data.keys():  # 把batch里的数据转化成tensor
    #         #     if key == 'u':
    #         #         batch_data[key] = torch.tensor(batch_data[key][0], dtype=torch.long)
    #         #     elif key in ['s', 's_next', 'o', 'o_next']:
    #         #         batch_data[key] = batch(batch_data[key][0],
    #         #                                 self.config['environment']['state_key'], self.all_tls)
    #         #     else:
    #         #         batch_data[key] = torch.tensor(batch_data[key][0], dtype=torch.float32)
    #         q_eval, self.eval_hidden, all_embedding, predict_o = \
    #             self.rnn(inputs, self.eval_hidden)
    #
    #         q_target, self.target_hidden, all_embedding_target, predict_o_target = \
    #             self.shared_target_rnn(inputs_next, self.target_hidden)
    #         pred_o.append(predict_o)
    #         pred_o_target.append(predict_o_target)
    #         q_evals.append(q_eval)
    #         q_targets.append(q_target)
    #         all_embeddings.append(all_embedding)
    #         all_embeddings_target.append(all_embedding_target)
    #     # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是( ##episode个数, n_agents，n_actions)
    #     # 把该列表转化成( ##episode个数, max_episode_len， n_agents，n_actions)的数组
    #     q_evals = torch.stack(q_evals, dim=0)
    #     q_targets = torch.stack(q_targets, dim=0)
    #     all_embeddings = torch.stack(all_embeddings, dim=0)
    #     all_embeddings_target = torch.stack(all_embeddings_target, dim=0)
    #     return q_evals, q_targets, all_embeddings, all_embeddings_target, pred_o, pred_o_target

    def choose_action(self, obs, unava_phase_index, epsilon):
        inputs = obs
        hidden_state = self.eval_hidden
        target_hidden_state = self.target_hidden

        if self.config["cuda"]:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        self.eval_hidden = self.rnn(inputs, hidden_state)
        q_value, predict_o = self.body(self.eval_hidden)

        self.target_hidden = self.shared_target_rnn(inputs, target_hidden_state)
        target_q_value, target_predict_o = self.shared_target_body(self.target_hidden)

        # choose action from q value
        for i in range(q_value.shape[0]):
            q_value[i, unava_phase_index[i]] = -float("inf")
        if np.random.uniform() < epsilon:
            action = np.array(
                [np.random.choice(
                    [i for i in range(q_value.shape[1]) if i not in unava_phase_index[j]]) for j
                    in range(q_value.shape[0])])
        else:
            action = torch.argmax(q_value, axis=1).detach().cpu().numpy()

        return action, q_value, predict_o, target_q_value, target_predict_o, \
            self.eval_hidden, self.target_hidden
