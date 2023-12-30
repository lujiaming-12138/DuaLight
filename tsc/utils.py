#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: utils
@time: 2022/8/2
"""
import os
import random
import xml.etree.ElementTree as ET
from random import sample
from multiprocessing import Process
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.sparse.csgraph import shortest_path
from matplotlib import animation
from matplotlib import pyplot as plt


def get_vector_cos_sim(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch(env_output, use_keys, all_tl):
    """Transform agent-wise env_output to batch format."""
    if all_tl == ['gym_test']:
        return torch.tensor([env_output])
    obs_batch = {}
    for i in use_keys + ['mask']:
        obs_batch[i] = []
    for agent_id in all_tl:
        out = env_output[agent_id]
        tmp_dict = {k: np.zeros(8) for k in use_keys}
        state, mask, _ = out
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

def resco_batch(env_output, use_keys, all_tl):
    """adapt the form of RESCO model input"""
    obs = {}
    for tl in all_tl:
        tl_state_list = []
        states, masks, _ = env_output[tl]
        for i in range(len(masks)):
            if masks[i] == 0:
                tl_state_list.append(list(states[i].values()))
        obs[tl] = np.expand_dims(np.array(tl_state_list), axis=0)
    return obs

def to_colight_obs(obs):
    '''change the sumo obs to colight obs'''
    colight_obs = []
    for tl_id, tl_obs in obs.items():
        obs_dict = dict()
        lane_num_vehicle = []
        stop_car_num = []
        cur_phase = []
        for lane in tl_obs[0]:
            if lane == {}:
                lane_num_vehicle.append(0)
                stop_car_num.append(0)
                cur_phase.append(0)
            else:
                lane_num_vehicle.append(lane['car_num'])
                stop_car_num.append(lane['stop_car_num'])
                cur_phase.append(lane['current_phase'])
        obs_dict['lane_num_vehicle'] = lane_num_vehicle
        obs_dict['lane_num_vehicle_been_stopped_thres1'] = stop_car_num
        obs_dict['adjacency_matrix'] = tl_obs[2][0][0]
        obs_dict['cur_phase'] = cur_phase
        obs_dict['id'] = tl_id
        colight_obs.append(obs_dict)
    return colight_obs

    #for colight TODO
def log(all_tls, list_inter_log, cur_time, before_action_feature, action):

    for inter_ind in range(len(all_tls)):
        list_inter_log[inter_ind].append({"time": cur_time,
                                                "state": before_action_feature[inter_ind],
                                                "action": action[inter_ind]})

def batch_log(path_to_log, list_inter_log, start, stop):
    for inter_ind in range(start, stop):
        if int(inter_ind)%100 == 0:
            print("Batch log for inter ",inter_ind)

        path_to_log_file = os.path.join(path_to_log, "inter_{0}.pkl".format(inter_ind))
        f = open(path_to_log_file, "wb")
        pickle.dump(list_inter_log[inter_ind], f)
        f.close()

def bulk_log_multi_process(path_to_log, all_tls, list_inter_log, batch_size=100):
    assert len(all_tls) == len(list_inter_log)
    if batch_size > len(all_tls):
        batch_size_run = len(all_tls)
    else:
        batch_size_run = batch_size
    process_list = []
    for batch in range(0, len(all_tls), batch_size_run):
        start = batch
        stop = min(batch + batch_size, len(all_tls))
        p = Process(target=batch_log, args=(path_to_log, list_inter_log, start, stop))
        print("before")
        p.start()
        print("end")
        process_list.append(p)
    print("before join")

    for t in process_list:
        t.join()

    print("end join")

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
    checkpoint = "_".join([checkpoints[0].split("_")[0], "{}.pt".format(max)])
    model.load_state_dict(torch.load(os.path.join(checkpointpath, checkpoint))['model_state_dict'])
    # model.load_state_dict(torch.load(os.path.join(checkpointpath, "model_4000.pt"))['model_state_dict'])
    model.eval()
    print("Load {} model done!".format(checkpoint))
    return model


def load_checkpoint2(model, checkpointpath):
    model.load_state_dict(torch.load(checkpointpath)['model_state_dict'])
    print("Load {} model done!".format(checkpointpath))
    return model


def load_checkpoint3(rnn_model, body_model, mixer_model, checkpointpath):
    rnn_model.load_state_dict(torch.load(checkpointpath)['rnn_model_state_dict'])
    body_model.load_state_dict(torch.load(checkpointpath)['body_model_state_dict'])
    mixer_model.load_state_dict(torch.load(checkpointpath)['mixer_model_state_dict'])
    print("Load {} model done!".format(checkpointpath))
    return rnn_model, body_model, mixer_model


def load_checkpoint4(body_model, checkpointpath):
    body_model.load_state_dict(torch.load(checkpointpath)['body_model_state_dict'])
    print("Load {} model done!".format(checkpointpath))
    return body_model


def checkpoint(checkpointpath, model, optimizer, step):
    print("Saving checkpoint to {}".format(checkpointpath))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(checkpointpath, "model_{}.pt".format(step))
    )


def checkpoint2(checkpointpath, shared_eval_rnn, shared_body, shared_eval_qmix_net, optimizer, step):
    print("Saving checkpoint to {}".format(checkpointpath))
    torch.save(
        {
            "rnn_model_state_dict": shared_eval_rnn.state_dict(),
            "body_model_state_dict": shared_body.state_dict(),
            "mixer_model_state_dict": shared_eval_qmix_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(checkpointpath, "model_{}.pt".format(step))
    )


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


# Since tie breaks for shortest path will result in deterministic behavior, i can make it nondeterministic better with jitter
def get_edge_length_dict(filepath, jitter=False):
    edges = {}
    # Parse through .net.xml file for lengths
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('edge'):
        id = c.attrib['id']
        # If function key in attib then continue
        if 'function' in c.attrib:
            continue
        length = float(c[0].attrib['length'])
        if jitter:
            length += random.randint(-1, 1)
        edges[id] = length  # ONLY WORKS FOR SINGLE LANE ROADS
    return edges


# Make a dict {external node: {'gen': gen edge for node, 'rem': rem edge for node} ... } for external nodes
# For adding trips
def get_node_edge_dict(filepath):
    dic = {}
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal' or 'intersection' in node:
            continue
        incLane = c.attrib['incLanes'].split('_0')[0]
        dic[node] = {'gen': incLane.split('___')[1] + '___' + incLane.split('___')[0],
                     'rem': incLane}
    return dic


def get_intersections(filepath):
    arr = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal' or not 'intersection' in node:
            continue
        arr.append(node)
    return arr


# Returns a arr of the nodes, and a dict where the key is a node and value is the index in the arr
def get_node_arr_and_dict(filepath):
    arr = []
    dic = {}
    i = 0
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal':
            continue
        arr.append(node)
        dic[node] = i
        i += 1
    return arr, dic


# For calc. global/localized discounted rewards
# {intA: {intA: 0, intB: 5, ...} ...}
def get_cartesian_intersection_distances(filepath):
    # First gather the positions
    tree = ET.parse(filepath)
    root = tree.getroot()
    intersection_positions = {}
    for c in root.iter('junction'):
        node = c.attrib['id']
        # If not intersection then continue
        if c.attrib['type'] == 'internal' or not 'intersection' in node:
            continue
        x = float(c.attrib['x'])
        y = float(c.attrib['y'])
        intersection_positions[node] = np.array([x, y])
    # Second calc. euclidean distances
    distances = {}
    for outer_k, outer_v in list(intersection_positions.items()):
        distances[outer_k] = {}
        for inner_k, inner_v in list(intersection_positions.items()):
            dist = np.linalg.norm(outer_v - inner_v)
            distances[outer_k][inner_k] = dist
    return distances


def get_non_intersections(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    arr = []
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal' or 'intersection' in node:
            continue
        arr.append(node)
    return arr


# for state interoplation
# {intA: [intB...]...} -> dict of each intersection of arrays of the intersections in its neighborhood
def get_intersection_neighborhoods(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    dic = {}
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal' or 'intersection' not in node:
            continue
        dic[node] = []
        # parse through incoming lanes
        incLanes = c.attrib['incLanes']
        incLanes_list = incLanes.split(' ')
        for lane in incLanes_list:
            lane_split = lane.split('___')[0]
            if 'intersection' in lane_split:
                dic[node].append(lane_split)
    # returns the dic of neighbors and the max number of neighbors ( + 1)
    return dic, max([len(v) for v in list(dic.values())]) + 1  # + 1 for including itself


'''
1. Dont want the light lengths to be less than the yellow phase (3 seconds)
2. Assume NS starting config for lights
'''


class RuleSet:
    def __init__(self, params, net_path=None, constants=None):
        self.net_path = net_path

    def reset(self):
        raise NotImplementedError

    def __call__(self, state):
        raise NotImplementedError


class CycleRuleSet(RuleSet):
    def __init__(self, params, net_path, constants):
        super(CycleRuleSet, self).__init__(params, net_path)
        assert params['phases']
        assert 'cycle_length' in params
        self.net_path = net_path
        self.gen_time = constants['episode']['generation_ep_steps']
        self.NS_mult = params['NS_mult']
        self.EW_mult = params['EW_mult']
        self.phase_end_offset = params['phase_end_offset']
        self.cycle_length = params['cycle_length']
        self.intersections = get_intersections(self.net_path)
        # [{'start_step': t, 'light_lengths': {intersection: {NS: #, EW: #}...}}...]
        self.phases = self._get_light_lengths(params['phases'])
        self.reset()

    def reset(self):
        self.curr_phase_indx = -1  # Keep track of current phase
        # Keep track of current light for each inetrsection as well as current time with that light
        self.info = {intersection: {'curr_light': 'NS', 'curr_time': 0} for intersection in
                     self.intersections}

    def __call__(self, state):
        old_lights = []
        new_lights = []
        t = state['sim_step']
        next_phase = False
        if self.curr_phase_indx < len(self.phases) - 1 and self.phases[self.curr_phase_indx + 1][
            'start_step'] == t:
            next_phase = True
            self.curr_phase_indx += 1

        for intersection in self.intersections:
            if self.info[intersection]['curr_light'] == 'EW':
                assert state[intersection]['curr_phase'] == 1 or state[intersection][
                    'curr_phase'] == 2
            elif self.info[intersection]['curr_light'] == 'NS':
                assert state[intersection]['curr_phase'] == 0 or state[intersection][
                    'curr_phase'] == 3
            # First get old lights to compare with new lights to see if switch is needed
            old_lights.append(self.info[intersection]['curr_light'])
            # If the next phase start step is equal to curr time then switch to next phase
            if next_phase:
                # Select random starting light if not yellow light (1 or 3)
                if state[intersection]['curr_phase'] == 0 or state[intersection][
                    'curr_phase'] == 2:
                    choice = sample(['NS', 'EW'], 1)[0]
                else:
                    if state[intersection]['curr_phase'] == 1:
                        choice = 'EW'
                    else:
                        choice = 'NS'
                self.info[intersection]['curr_light'] = choice
                self.info[intersection]['curr_time'] = 0
            else:  # o.w. continue in curr phase, switch if end of current light time
                light_lengths = self.phases[self.curr_phase_indx]['light_lengths']
                intersection_light_lengths = light_lengths[intersection]
                curr_light = self.info[intersection]['curr_light']
                curr_time = self.info[intersection]['curr_time']
                # If curr time of light is equal to its length then switch
                if curr_time >= intersection_light_lengths[curr_light]:
                    self.info[intersection]['curr_light'] = 'NS' if curr_light == 'EW' else 'EW'
                    self.info[intersection]['curr_time'] = 0
                else:
                    self.info[intersection]['curr_time'] += 1
            # Get new light
            new_lights.append(self.info[intersection]['curr_light'])
        # Compare old lights with new lights
        bin = [0 if x == y else 1 for x, y in zip(old_lights, new_lights)]
        return np.array(bin)

    def _get_light_lengths(self, phases):
        NODES_arr, NODES = get_node_arr_and_dict(self.net_path)
        N = len(list(NODES))
        assert N == len(set(list(NODES.keys())))

        # Matrix of edge lengths for dikstras
        M = np.zeros(shape=(N, N))

        # Need dict of edge lengths
        edge_lengths = get_edge_length_dict(self.net_path, jitter=False)

        # Loop through nodes and fill M using incLanes attrib
        # Also make dict of positions for nodes so I can work with phases
        node_locations = {}
        tree = ET.parse(self.net_path)
        root = tree.getroot()
        for c in root.iter('junction'):
            id = c.attrib['id']
            # Ignore if type is internal
            if c.attrib['type'] == 'internal':
                continue
            # Add location for future use
            node_locations[id] = (c.attrib['x'], c.attrib['y'])
            # Current col
            col = NODES[id]
            # Get the edges to this node through the incLanes attrib (remove _0 for the first lane)
            incLanes = c.attrib['incLanes']
            # Space delimited
            incLanes_list = incLanes.split(' ')
            for lane in incLanes_list:
                in_edge = lane.split('_0')[0]
                in_node = in_edge.split('___')[0]
                row = NODES[in_node]
                length = edge_lengths[in_edge]
                M[row, col] = length

        # Get the shortest path matrix
        _, preds = shortest_path(M, directed=True, return_predecessors=True)

        # Fill dict of gen{i}___rem{j} with all nodes on the shortest route
        # Todo: Important assumption, "intersection" is ONLY in the name/id of an intersection
        shortest_routes = {}
        for i in range(N):
            gen = NODES_arr[i]
            if 'intersection' in gen: continue
            for j in range(N):
                if i == j:
                    continue
                rem = NODES_arr[j]
                if 'intersection' in rem: continue
                route = [rem]
                while preds[i, j] != i:  # via
                    j = preds[i, j]
                    route.append(NODES_arr[int(j)])
                route.append(gen)
                shortest_routes['{}___{}'.format(rem, gen)] = route

        def get_weightings(probs):
            weights = {intersection: {'NS': 0., 'EW': 0.} for intersection in self.intersections}
            for intersection, phases in list(weights.items()):
                # Collect all routes that include the intersection
                for v in list(shortest_routes.values()):
                    if intersection in v:
                        int_indx = v.index(intersection)
                        # Route weight is gen + rem probs
                        route_weight = probs[v[0]]['gen'] + probs[v[-1]]['rem']
                        # Detect if NS or EW
                        # If xs are equal between the node before and the intersection then EW is used
                        if node_locations[v[int_indx - 1]][0] == node_locations[intersection][0]:
                            phases['NS'] += self.NS_mult * route_weight
                        else:
                            assert node_locations[v[int_indx - 1]][1] == \
                                   node_locations[intersection][
                                       1], 'xs not equal so ys should be.'
                            phases['EW'] += self.EW_mult * route_weight
            return weights

        def convert_weights_to_lengths(weights, cycle_length=10):
            ret_dic = {}
            for intersection, dic in list(weights.items()):
                total = dic['NS'] + dic['EW']
                NS = round((dic['NS'] / total) * cycle_length)
                EW = round((dic['EW'] / total) * cycle_length)
                assert NS >= 3 and EW >= 3, 'Assuming yellow phases are 3 make sure these lengths are not less.'
                ret_dic[intersection] = {'NS': NS, 'EW': EW}
            return ret_dic

        # For each phase get weights and cycle lengths
        new_phases = []
        curr_t = 0
        for p, phase in enumerate(phases):
            weights = get_weightings(phase['probs'])
            light_lengths = convert_weights_to_lengths(weights, self.cycle_length)
            phase_length = phase['duration'] * self.gen_time
            # Want to offset the end of each phase, except for the first phase which needs to start at the beginning
            offset = self.phase_end_offset
            if p == 0:
                offset = 0
            new_phases.append(
                {'start_step': round(curr_t + offset), 'light_lengths': light_lengths})
            curr_t += round(phase_length)
        return new_phases


class TimerRuleSet(RuleSet):
    def __init__(self, params, net_path=None, constants=None):
        super(TimerRuleSet, self).__init__(params)
        assert 'length' in params, 'Length not in params'
        self.timer_length = params['length']
        self.offset = params['offset'] if 'offset' in params else None
        self.start_phase = params['start_phase'] if 'start_phase' in params else None
        assert isinstance(self.timer_length, list)
        self.reset()

    def reset(self):
        # Start at offset if given, else zeros
        self.current_phase_timer = [x for x in self.offset] if self.offset else [0] * len(
            self.timer_length)
        self.just_reset = True

    def __call__(self, state):
        # If just reset then send switch for start phases of 1
        if self.start_phase and self.just_reset:
            self.just_reset = False
            return np.array(self.start_phase)
        li = []
        for i in range(len(self.timer_length)):
            # Can ignore the actual env state
            if self.current_phase_timer[i] >= self.timer_length[i]:
                self.current_phase_timer[i] = 0
                li.append(1)  # signals to the env to switch the actual phase
                continue
            self.current_phase_timer[i] += 1
            li.append(0)
        return np.array(li)


# Acknowledge: Ilya Kostrikov (https://github.com/ikostrikov)
# This assigns this agents grads to the shared grads at the very start
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


# Acknowledge: Alexis Jacq (https://github.com/alexis-jacq)
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


# Acknowledge: Shangtong Zhang (https://github.com/ShangtongZhang)
class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean', "unava_index", "predict_o"]
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


# Acknowledge: Shangtong Zhang (https://github.com/ShangtongZhang)
def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


# Acknowledge: Shangtong Zhang (https://github.com/ShangtongZhang)
def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]
