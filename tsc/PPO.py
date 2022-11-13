#!/usr/bin/env python3
# encoding: utf-8
"""
@paper: A GENERAL SCENARIO-AGNOSTIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL
@file: PPO
"""
import time
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from tsc.utils import Storage, tensor, random_sample, ensure_shared_grads, batch


class Worker:
    def __init__(self, constants, device, env, id):
        self.constants = constants
        self.device = device
        self.env = env
        self.id = id

    def _reset(self):
        raise NotImplementedError

    def _get_prediction(self, states, actions=None, ep_step=None):
        raise NotImplementedError

    def _get_action(self, prediction):
        raise NotImplementedError

    def _copy_shared_model_to_local(self):
        raise NotImplementedError

    def get_reward(self, reward):
        ans = []
        for i in self.all_tls:
            ans.append(sum(reward[i].values()))
        return ans


class PPOWorker(Worker):
    def __init__(self, constants, device, env, shared_NN, local_NN, optimizer, id, num_agnet=None, dont_reset=False):
        super(PPOWorker, self).__init__(constants, device, env, id)
        self.NN = local_NN
        self.shared_NN = shared_NN
        if not dont_reset:  # for the vis agent script this messes things up
            self.state = self.env.reset()
        self.ep_step = 0
        self.opt = optimizer
        if num_agnet:
            self.num_agents = num_agnet
            self.all_tls = ['gym_test']
            self.state_keys = ['gym_test']
            self._all_reward = 0
        else:
            self.num_agents = len(env.all_tls)
            self.all_tls = self.env.all_tls
            self.state_keys = ['mask'] + constants['environment']['state_key']
        self.NN.eval()

    def _get_prediction(self, states, unava_phase_index, actions=None, ep_step=None):
        return self.NN(states, unava_phase_index, actions)

    def _get_action(self, prediction):
        return prediction['a'].cpu().numpy()

    def _copy_shared_model_to_local(self):
        self.NN.load_state_dict(self.shared_NN.state_dict())

    def _stack(self, val):
        assert not isinstance(val, list)
        return np.stack([val] * self.num_agents)

    def train_rollout(self, total_step, unava_phase_index=None):
        storage = Storage(self.constants['episode']['rollout_length'])
        state = deepcopy(self.state)
        step_times = []

        if self.all_tls == ['gym_test']:
            all_reward = []
        else:
            all_reward = None
        self._copy_shared_model_to_local()
        rollout_amt = 0  # keeps track of the amt in the current rollout storage
        while rollout_amt < self.constants['episode']['rollout_length']:
            start_step_time = time.time()
            state = batch(state, self.constants['environment']['state_key'], self.all_tls)
            prediction = self._get_prediction(state, unava_phase_index)
            action = self._get_action(prediction)
            if self.all_tls != ['gym_test']:
                tl_action_select = {}
                for tl_index in range(len(self.all_tls)):
                    tl_action_select[self.all_tls[tl_index]] = \
                        (self.env._crosses[self.all_tls[tl_index]].green_phases)[action[tl_index]]
                next_state, reward, done, _all_reward = self.env.step(tl_action_select)
                reward = self.get_reward(reward)
            else:
                next_state, reward, done, info = self.env.step(action[0])
                self._all_reward += reward  # not use
            self.ep_step += 1

            # This is to stabilize learning, since the first some amt of states wont represent the env very well
            # since it will be more empty than normal
            if self.ep_step > self.constants['episode']['warmup_ep_steps']:
                storage.add(prediction)
                storage.add({'r': tensor(reward, self.device).unsqueeze(-1),
                             'm': tensor(self._stack(1 - done), self.device).unsqueeze(-1)})
                if self.state_keys == ['gym_test']:
                    storage.add({'gym_test': tensor(state, self.device)})
                else:
                    for k in self.state_keys:
                        storage.add({k: tensor(state[k], self.device)})
                    storage.extend({"unava_index": unava_phase_index})
                rollout_amt += 1

            if done:
                # Sync local model with shared model at start of each ep
                # self._copy_shared_model_to_local()
                self.ep_step = 0
                if self.all_tls == ['gym_test']:
                    all_reward.append(self._all_reward)
                    self._all_reward = 0
                else:
                    all_reward = _all_reward
                next_state = self.env.reset()
                # break

            state = deepcopy(next_state)

            total_step += 1

            end_step_time = time.time()
            step_times.append(end_step_time - start_step_time)

        self.state = deepcopy(state)
        state = batch(state, self.constants['environment']['state_key'], self.all_tls)
        prediction = self._get_prediction(state, unava_phase_index)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((self.num_agents, 1)), self.device)
        returns = prediction['v'].detach()
        for i in reversed(range(self.constants['episode']['rollout_length'])):
            # Disc. Return
            returns = storage.r[i] + self.constants['ppo']['discount'] * storage.m[i] * returns
            # GAE
            td_error = storage.r[i] + self.constants['ppo']['discount'] * storage.m[i] * storage.v[i + 1] - storage.v[i]
            advantages = advantages * self.constants['ppo']['gae_tau'] * self.constants['ppo']['discount'] * storage.m[i] + td_error
            # tmp = {'adv': advantages.detach(),
            #        'ret': returns.detach()}
            # storage.insert(tmp)
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()


        actions, log_probs_old, returns, advantages = storage.cat(['a', 'log_pi_a', 'ret', 'adv'])
        states = storage.cat([k for k in self.state_keys])
        unava_phase_indexs = storage.unava_index
        if self.all_tls == ['gym_test']:
            for k, v in enumerate(states):
                states_ = v
        else:
            states_ = {}
            for k, v in enumerate(states):
                states_[self.state_keys[k]] = v
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Train
        self.NN.train()
        batch_times = []
        train_pred_times = []
        break_flag = False
        for i in range(self.constants['ppo']['optimization_epochs']):
            if break_flag:
                break
            # Sync. at start of each epoch
            self._copy_shared_model_to_local()
            sampler = random_sample(np.arange(len(advantages)), min(self.constants['ppo']['minibatch_size'], len(advantages)))
            cnt = 0
            for batch_indices in sampler:
                start_batch_time = time.time()

                batch_indices = tensor(batch_indices, self.device).long()

                # Important Node: these are tensors but dont have a grad
                if self.all_tls == ['gym_test']:
                    unava_phase_indexs_sample = None
                    sampled_states = states_[batch_indices]
                else:
                    sampled_states = {}
                    for k, v in states_.items():
                        sampled_states[k] = v[batch_indices]
                    unava_phase_indexs_sample = []
                    for ind in batch_indices:
                        unava_phase_indexs_sample.append(unava_phase_indexs[ind])
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                start_pred_time = time.time()
                prediction = self._get_prediction(sampled_states, unava_phase_indexs_sample, sampled_actions)
                end_pred_time = time.time()
                train_pred_times.append(end_pred_time - start_pred_time)

                # Calc. Loss
                logratio = prediction['log_pi_a'] - sampled_log_probs_old
                ratio = logratio.exp()

                if self.constants['ppo'].get("target_kl") is not None:
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - logratio).mean()

                    if approx_kl > self.constants['ppo']['target_kl']:
                        break_flag = True
                        break

                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.constants['ppo']['ppo_ratio_clip'],
                                          1.0 + self.constants['ppo']['ppo_ratio_clip']) * sampled_advantages

                # policy loss and value loss are scalars
                policy_loss = -torch.min(obj, obj_clipped).mean()
                entropy_loss = -self.constants['ppo']['entropy_weight'] * prediction['ent'].mean()
                value_loss = self.constants['ppo']['value_loss_coef'] *  torch.nn.functional.mse_loss(sampled_returns, prediction['v'])
                self.opt.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                if self.constants['ppo']['clip_grads']:
                    te = nn.utils.clip_grad_norm_(self.NN.parameters(), self.constants['ppo']['gradient_clip'])
                # for param in self.NN.parameters():
                #     print(param)
                #     if torch.isnan(param).any():
                #         print(1)
                ensure_shared_grads(self.NN, self.shared_NN)
                self.opt.step()
                end_batch_time = time.time()
                batch_times.append(end_batch_time - start_batch_time)
                cnt += 1
        self.NN.eval()
        return policy_loss, entropy_loss, value_loss, sampled_advantages.mean(), all_reward
