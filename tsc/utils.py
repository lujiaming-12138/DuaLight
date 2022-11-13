#!/usr/bin/env python3
# encoding: utf-8
"""
@paper: A GENERAL SCENARIO-AGNOSTIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL
@file: utils
"""
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp


def get_vector_cos_sim(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch(env_output, use_keys, all_tl):
    """Transform agent-wise env_output to batch format."""
    if all_tl == ['gym_test']:
        return torch.tensor([env_output])
    obs_batch = {}
    for i in use_keys+['mask']:
        obs_batch[i] = []
    for agent_id in all_tl:
        out = env_output[agent_id]
        tmp_dict = {k: np.zeros(8) for k in use_keys}
        state, mask = out
        for i in range(len(state)):
            for s in use_keys:
                tmp_dict[s][i] = state[i].get(s, 0)
        for k in use_keys:
            obs_batch[k].append(tmp_dict[k])
        obs_batch['mask'].append(mask)

    for key, val in obs_batch.items():
        if key not in ['current_phase', 'mask']:
            obs_batch[key] = torch.FloatTensor(np.array(val))
        else:
            obs_batch[key] = torch.LongTensor(np.array(val))

    return obs_batch


def flush_writer(writer):
    if writer.all_writers is None:
        return
    for w in writer.all_writers.values():
        w.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return


def load_checkpoint(model, checkpointpath):
    checkpoints = os.listdir(checkpointpath)
    max = 0
    for i in checkpoints:
        try:
            if int(i.split("_")[-1][:-3]) > max:
                max = int(i.split("_")[-1][:-3])
        except:
            pass
    checkpoint = "_".join([checkpoints[0].split("_")[0],"{}.pt".format(max)])
    model.load_state_dict(torch.load(os.path.join(checkpointpath, checkpoint))['model_state_dict'])
    # model.load_state_dict(torch.load(os.path.join(checkpointpath, "model_4000.pt"))['model_state_dict'])
    model.eval()
    print("Load {} model done!".format(checkpoint))
    return model


def load_checkpoint2(model, checkpointpath):

    model.load_state_dict(torch.load(checkpointpath)['model_state_dict'])
    print("Load {} model done!".format(checkpointpath))
    return model


def checkpoint(checkpointpath, model, optimizer, step):
    print("Saving checkpoint to {}".format(checkpointpath))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(checkpointpath, "model_{}.pt".format(step))
    )


# This assigns this agents grads to the shared grads at the very start
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self, amt=1):
        # used by workers
        with self.lock:
            self.val.value += amt

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean',"unava_index"]
        self.keys = keys
        self.size = size
        self.reset()

    def insert(self, data):
        for k, v in data.items():
            assert k in self.keys
            getattr(self, k).insert(0, v)

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def extend(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).extend(v)

    def placeholder(self, size=None):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]