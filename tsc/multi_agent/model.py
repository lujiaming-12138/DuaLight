#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: model
@time: 2022/12/9
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

import numpy as np


def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def fc_block(
        in_channels,
        out_channels,
        activation=None,
        use_dropout=False,
        norm_type=None,
        dropout_probability=0.5,
        bias=True
):
    block = [nn.Linear(in_channels, out_channels, bias=bias)]
    xavier_normal_(block[-1].weight)
    if norm_type is not None and norm_type != 'none':
        if norm_type == 'LN':
            block.append(nn.LayerNorm(out_channels))
        else:
            raise NotImplementedError
    if isinstance(activation, torch.nn.Module):
        block.append(activation)
    elif activation is None:
        pass
    else:
        raise NotImplementedError
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class GATLayer(torch.nn.Module):
    def __init__(self, agent_size, neigh_size, num_head, mid_size):
        super(GATLayer, self).__init__()
        self.num_head = num_head
        self.mid_size = mid_size
        self.agent_repr = fc_block(agent_size, num_head * mid_size, activation=nn.ReLU())
        self.neigh_repr = fc_block(neigh_size, num_head * mid_size, activation=nn.ReLU())

        self.neigh_hidden_repr = fc_block(neigh_size, num_head * mid_size, activation=nn.ReLU())
        self.out_fc = fc_block(num_head * mid_size, int(num_head * mid_size / 2), activation=nn.ReLU())

    def forward(self, inputs):
        agent, neighbor, neighbor_mask = inputs
        num_agent = agent.shape[0]
        neighbor_repr_head = self.neigh_repr(neighbor).reshape(
            num_agent, 4, self.num_head, self.mid_size)
        agent_repr_head = self.agent_repr(agent).reshape(
            num_agent, 1, self.num_head, self.mid_size)
        neighbor_repr_head = neighbor_repr_head.permute(0, 2, 1, 3)  #  num_agent self.num_head 4 self.mid_size
        agent_repr_head = agent_repr_head.permute(0, 2, 1, 3)
        neighbor_repr_head = neighbor_repr_head.reshape(-1, 4, self.mid_size).permute(0, 2, 1)  # num_agent*self.num_head self.mid_size 4
        agent_repr_head = agent_repr_head.reshape(-1, 1, self.mid_size)  # num_agent*self.num_head 1 self.mid_size
        att = torch.bmm(agent_repr_head, neighbor_repr_head).reshape(num_agent, self.num_head, 4)  # num_agent self.num_head 4
        att_score = torch.nn.Softmax(-1)(neighbor_mask.unsqueeze(1)*-1e8 + att)

        neighbor_hidden_repr_head = self.neigh_hidden_repr(neighbor).reshape(
            num_agent, 4, self.num_head, self.mid_size)
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.permute(0, 2, 1, 3)  # num_agent num_head 4 self.mid_size
        att_score = (1-neighbor_mask.unsqueeze(1).expand(num_agent, self.num_head, 4).float()) * att_score
        out = (att_score.unsqueeze(-1) * neighbor_hidden_repr_head).mean(2).reshape(num_agent, -1)  # num_agent num_head*self.mid_size
        out = self.out_fc(out)

        return out


class ModelBody(nn.Module):
    def __init__(self, input_size, fc_layer_size, state_keys, num_of_layers=1):
        self.state_keys = state_keys
        super(ModelBody, self).__init__()
        self.name = 'model_body'

        # mlp
        self.fc_car_num = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_queue_length = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_occupancy = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_flow = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_stop_car_num = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_pressure = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_nei_dis = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        # current phase
        self.current_phase_act = nn.Sigmoid()
        self.current_phase_embedding = nn.Embedding(2, 4)
        # mask
        self.mask_act = nn.Sigmoid()
        self.mask_embedding = nn.Embedding(2, 4)
        # constant zero embedding for no neighbor direction
        self.padding_zero = torch.zeros([1, 224])
        # neighbor direction embedding
        self.n_direct_emb = F.one_hot(torch.arange(0, 4))
        # self.neigh_fc = fc_block(228, 256, activation=nn.ReLU())
        self.neigh_fc = fc_block(232, 256, activation=nn.ReLU())
        # agent liner
        self.embedding_fc = fc_block(224, 256, activation=nn.ReLU())
        # self.embedding_fc2 = fc_block(224, 256, activation=nn.ReLU())
        # Multi head GAT
        gat_layers = []
        for i in range(num_of_layers):
            layer = GATLayer(
                256, 256, 4, 128
            )
            gat_layers.append(layer)
        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, input):
        all_key_state = []
        bs = (list(input.values())[0].shape)[0]
        for i in self.state_keys:
            if i in ['current_phase', 'mask']:
                continue
            all_key_state.append(eval("self.fc_{}".format(i))(input[i].reshape(-1, 1)))
        if "current_phase" in self.state_keys:
            all_key_state.append(self.current_phase_act(self.current_phase_embedding(input['current_phase'])).reshape(-1, 4))
        all_key_state.append(self.mask_act(
            self.mask_embedding(input['mask'])).reshape(-1, 4))
        encode_emb = torch.cat(all_key_state, dim=1).reshape(bs, -1)  # bs, 224

        # neighbor embedding
        encode_emb_add_padding = torch.cat([encode_emb, self.padding_zero], dim=0)
        neighbor_index, neighbor_dis = input['neighbor_index'], input['neighbor_dis']
        neighbor_dis = self.fc_nei_dis(neighbor_dis.unsqueeze(-1))
        neighbor_embed = encode_emb_add_padding[neighbor_index]
        neighbor_embed = torch.cat([
            neighbor_embed, neighbor_dis,
            self.n_direct_emb.expand(bs, 4, 4).float()
        ], dim=-1)
        neighbor_embed = self.neigh_fc(neighbor_embed)
        neighbor_mask = neighbor_index == -1


        agent_embed = torch.cat(all_key_state, dim=1).reshape(bs, -1)
        agent_embed1 = self.embedding_fc(agent_embed)

        # GAT
        neighbor_embeddings = self.gat_net((agent_embed1, neighbor_embed, neighbor_mask))


        # agent_embed2 = self.embedding_fc2(agent_embed)
        all_embedding = torch.cat([agent_embed1, neighbor_embeddings], -1)

        return all_embedding

        # return agent_embed2


class ActorModel(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        self.model = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), action_size),
        )

    def forward(self, hidden_states):
        outputs = self.model(hidden_states)
        return outputs


class CriticModel(nn.Module):
    def __init__(self, hidden_size):
        super(CriticModel, self).__init__()
        self.name = 'critic'
        self.c_fc1 = fc_block(hidden_size, int(hidden_size / 2), activation=nn.ReLU())
        self.c_fc2 = nn.Linear(hidden_size, 1)
        self.p_fc1 = fc_block(hidden_size, int(hidden_size / 2), activation=nn.ReLU())
        self.p_fc2 = nn.Linear(int(hidden_size / 2), 8)

    def forward(self, hidden_states):
        c1 = self.c_fc1(hidden_states)
        p1 = self.p_fc1(hidden_states)
        c2 = self.c_fc2(torch.cat([c1, p1], dim=-1))
        p2 = self.p_fc2(p1)
        return c2, p2


class NN_Model(nn.Module):
    def __init__(self, action_size, hidden_layer_size, state_keys, device='cpu'):
        super(NN_Model, self).__init__()
        self.state_keys = state_keys
        self.body_model = ModelBody(1, hidden_layer_size, self.state_keys).to(device)
        self.actor_model = ActorModel(512, action_size).to(device)
        self.critic_model = CriticModel(512).to(device)

    def forward(self, inputs, unava_phase_index, actions=None, eval=False):
        hidden_states = self.body_model(inputs)
        v, p = self.critic_model(hidden_states)
        logits = self.actor_model(hidden_states)
        if unava_phase_index:
            for i in range(hidden_states.shape[0]):
                logits[i, unava_phase_index[i]] = -1e8
        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            if not eval:
                actions = dist.sample()
            else:
                actions = torch.argmax(logits, axis=1)
        log_prob = dist.log_prob(actions).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': actions,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v,
                'predict_o': p}


def batch(env_output, use_keys, all_tl):
    """Transform agent-wise env_output to batch format."""
    if all_tl == ['gym_test']:
        return torch.tensor([env_output])
    obs_batch = {}
    for i in use_keys+['mask', 'neighbor_index', 'neighbor_dis']:
        obs_batch[i] = []
    for agent_id in all_tl:
        out = env_output[agent_id]
        tmp_dict = {k: np.zeros(8) for k in use_keys}
        state, mask, neight_msg = out
        for i in range(len(state)):
            for s in use_keys:
                tmp_dict[s][i] = state[i].get(s, 0)
        for k in use_keys:
            obs_batch[k].append(tmp_dict[k])
        obs_batch['mask'].append(mask)
        obs_batch['neighbor_index'].append(neight_msg[0][0])
        obs_batch['neighbor_dis'].append(neight_msg[0][1])

    for key, val in obs_batch.items():
        if key not in ['current_phase', 'mask', 'neighbor_index']:
            obs_batch[key] = torch.FloatTensor(np.array(val))
        else:
            obs_batch[key] = torch.LongTensor(np.array(val))

    return obs_batch


if __name__ == '__main__':
    from sumo_files.env.sim_env import TSCSimulator

    config = {
        "name": "test",
        "agent": "",
        "sumocfg_file": "sumo_files/scenarios/nanshan/osm.sumocfg",
        # "sumocfg_file": "sumo/scenarios/large_grid2/exp_0.sumocfg",
        # "sumocfg_file": "sumo/scenarios/sumo_wj3/rl_wj.sumocfg",
        # "sumocfg_file": "sumo/scenarios/real_net/most_0.sumocfg",
        # "sumocfg_file": "sumo/scenarios/sumo_fenglin_base_road/base.sumocfg",
        "action_type": "select_phase",
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'state_key': ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num']
    }
    env = TSCSimulator(config, 123)
    next_obs = env.reset()
    all_tl = env.all_tls
    model = NN_Model(8, 4, state_keys=config['state_key'])
    batch_ph = env.tl_phase_index
    unava_phase_index = []
    for i in env.all_tls:
        unava_phase_index.append(env._crosses[i].unava_index)
    for i in range(100):
        batch_next_obs = batch(next_obs, config['state_key'], all_tl)
        action = model(batch_next_obs, unava_phase_index)
        tl_action_select = {}
        for tl_index in range(len(all_tl)):
            tl_action_select[all_tl[tl_index]] = (env._crosses[all_tl[tl_index]].green_phases)[action['a'][tl_index]]
        next_obs, reward, done, all_reward = env.step(tl_action_select)
    env.terminate()

