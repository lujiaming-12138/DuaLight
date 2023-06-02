#!/usr/bin/env python3
# encoding: utf-8
import copy
import os
import subprocess
import time
import math
import xml.etree.ElementTree
import numpy as np
import libsumo
import traci
import sumolib
from sumolib import checkBinary
from sumo_files.env.intersection import Intersection


class TSCSimulator:
    def __init__(self, config, port, not_default=True):
        self.not_default = not_default
        self.port = port
        self.name = config.get("name")
        self.seed = config.get('seed', 777)
        self.agent = config.get('agent')
        self._yellow_duration = config.get("yellow_duration")
        self.iter_duration = config.get("iter_duration")
        # self.init_data(is_record, record_stats, output_path)
        self.is_record = config.get("is_record")
        self.config = config
        self.output_path = config.get("output_path")
        self.reward_type = config.get("reward_type")
        self.is_neighbor_reward = config.get("is_neighbor_reward", False)
        self.cfg = config
        self.step_num = config.get("step_num", 1)
        self.p = config.get("p", 1) #for trip.xml output dir
        self._current_time = 0
        self._init_sim(config.get("sumocfg_file"), self.seed,
                       self.config.get("episode_length_time"), config.get("gui"))
        self.all_tls = list(self.sim.trafficlight.getIDList())
        self.infastructure = self._infastructure_extraction1(config.get("sumocfg_file"))
        # delete invalid tl part 1
        rm_num = 0
        ora_num = len(self.all_tls)
        for i in self.infastructure:
            if self.infastructure[i]['entering_lanes_pos'] == []:
                rm_num += 1
                self.all_tls.remove(i)
                print("no infastrurct: {}".format(i))
        for rtl in ['1673431902', '8996', '9153', '9531', '9884']:
            if rtl in self.all_tls:
                rm_num += 1
                self.all_tls.remove(rtl)
        rm_tl = []
        self._crosses = {}
        self.state_key = config['state_key']
        self.all_reward = {}
        self.tl_phase_index = []
        for tl in self.all_tls:
            self._crosses[tl] = Intersection(tl, self.infastructure[tl],
                                             self, self.state_key, self.not_default)
            tl_ava = self._crosses[tl].get_tl_ava()
            if not tl_ava:
                rm_tl.append(tl)
                rm_num += 1
                print("Not ava: {}".format(tl))
        # delete invalid tl part 2
        for tl in rm_tl:
            del self._crosses[tl]
            self.all_tls.remove(tl)
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl].update_timestep()
            self.tl_phase_index.append(self._crosses[tl].get_phase_index())

        print("Remove {} tl, percent: {}".format(rm_num, rm_num / ora_num))
        
        self.is_adjacency_remove = config.get("is_adjacency_remove", True)
        self.adjacency_top_k = config.get("adjacency_top_k", 5)
        
        self.infastructure = self._infastructure_extraction2(config.get("sumocfg_file"),
                                                             self.infastructure,
                                                             config.get("is_dis", False))

        self.action_type = config.get('action_type')

        self.vehicle_info = {}

        self.terminate()

    def _init_sim(self, sumocfg_file, seed, episode_length_time, gui=False):
        self.episode_length_time = episode_length_time
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        if sumocfg_file.split("/")[-1] in ['grid4x4.sumocfg', 'arterial4x4.sumocfg']:
            route_name = sumocfg_file.split("/")[-1][:-8]
            net = "/".join(sumocfg_file.split("/")[:-1] + [route_name + '.net.xml'])
            route = "/".join(sumocfg_file.split("/")[:-1] + [route_name])
            command = [
                checkBinary(app), '-n', net, '-r', route + '_' + str(self.step_num) + '.rou.xml']
            command += ['-a', "/".join(sumocfg_file.split("/")[:-1] + ['e1.add.xml']) + ", " +
                        "/".join(sumocfg_file.split("/")[:-1] + ['e2.add.xml'])]
        else:
            command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--random']
        # command += ['--remote-port', str(self.port)]
        # command += ['--no-step-log', 'True']
        # if self.name != 'real_net':
        #     command += ['--time-to-teleport',
        #                 '600']  # long teleport for safety
        # else:
        #     command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            if not os.path.exists(self.output_path):
                try:
                    os.makedirs(self.output_path)
                except:
                    pass
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_%s_trip.xml' % (
                            self.name, sumocfg_file.split("/")[-1], self.p)),
                        '--tripinfo-output.write-unfinished']
        # subprocess.Popen(command)
        self.step_num += 1
        self.p += 1
        # wait 2s to establish the traci server
        time.sleep(5)

        libsumo.start(command)
        self.sim = libsumo

        if sumocfg_file.split("/")[-1] == 'ingolstadt21.sumocfg':
            self._current_time = 57600 - 1
            self.episode_length_time += self._current_time
            self.sim.simulationStep(self._current_time)
        elif sumocfg_file.split("/")[-1] == 'cologne8.sumocfg':
            self._current_time = 25200 - 1
            self.episode_length_time += self._current_time
            self.sim.simulationStep(self._current_time)
        # self.sim = traci.start(command)

    def _init_action_type(self, action_type):
        assert action_type in ['select_phase', "change", "generate"]
        self.action_type = action_type

    def terminate(self):
        self.sim.close()

    def _do_action(self, action):
        if self.action_type == 'select_phase':
            for tl, a in action.items():
                assert a / 2 not in self._crosses[tl].unava_index, "{}-{}-{}".format(tl, a, self._crosses[tl].unava_index)
                current_phase_index = self._crosses[tl].getCurrentPhaseIndex()
                if a == current_phase_index:
                    continue
                else:
                    yellow_phase = current_phase_index + 1
                    self._crosses[tl].set_phase_by_index(yellow_phase, self._yellow_duration)
            self._current_time += self._yellow_duration
            self.sim.simulationStep(self._current_time)

            for tl, a in action.items():
                self._crosses[tl].set_phase_by_index(a, self.iter_duration)
            self._current_time += self.iter_duration
            self.sim.simulationStep(self._current_time)

        elif self.action_type == 'change':
            for tl, a in action.items():
                if a:
                    current_phase_index = self._crosses[tl].getCurrentPhaseIndex()
                    yellow_phase = current_phase_index + 1
                    self._crosses[tl].set_phase_by_index(yellow_phase, self._yellow_duration)
            self._current_time += self._yellow_duration
            self.sim.simulationStep(self._current_time)

            for tl, a in action.items():
                if a:
                    next_phase = self._crosses[tl].getCurrentPhaseIndex() + 2
                    if next_phase > self._crosses[tl].green_phases[-1]:
                        next_phase = 0
                else:
                    next_phase = self._crosses[tl].getCurrentPhaseIndex()
                self._crosses[tl].set_phase_by_index(next_phase, self.iter_duration)
            self._current_time += self.iter_duration
            self.sim.simulationStep(self._current_time)

        elif self.action_type == 'generate':
            for tl, a in action.items():
                current_phase_str = self._crosses[tl].getCurrentPhase()
                if a != current_phase_str:
                    yellow = self._crosses[tl].getCurrentPhaseYellow(current_phase_str)
                    self._crosses[tl].set_phase(yellow, self._yellow_duration)
            self._current_time += self._yellow_duration
            self.sim.simulationStep(self._current_time)

            for tl, a in action.items():
                self._crosses[tl].set_phase(a, self.iter_duration)
            self.sim.simulationStep(self._current_time)
        else:
            raise NotImplemented

    def _get_reward(self):

        list_reward = {tl: self._crosses[tl].get_reward(self.reward_type) for tl in self.all_tls}

        return list_reward

    def step(self, action):
        self._do_action(action)
        for tl in self.all_tls:
            self._crosses[tl].update_timestep()
        done = False
        obs = self._get_state()
        reward_ = self._get_reward()
        # multi-agent reward sum with neighbor
        if self.is_neighbor_reward:
            reward = {}
            for tl in self.all_tls:
                reward[tl] = {}
                for k in reward_[tl]:
                    reward[tl][k] = reward_[tl][k]
                    count = 0
                    tmp = 0
                    for nei in self._crosses[tl].nearset_inter[0][0]:
                        if nei != -1:
                            count += 1
                            tmp += reward_[self.all_tls[nei]][k]
                    if count > 0:
                        reward[tl][k] += tmp / count
        else:
            reward = reward_
        for tl, v in reward.items():
            for k, r in v.items():
                self.all_reward[tl][k] += r
        if self._current_time >= self.episode_length_time:
            done = True
            self.terminate()
        return obs, reward, done, self.all_reward

    def default_step(self):
        done = False
        self._current_time += 15
        self.sim.simulationStep(self._current_time)
        for tl in self.all_tls:
            self._crosses[tl].update_timestep()
        reward = self._get_reward()
        for tl, v in reward.items():
            for k, r in v.items():
                self.all_reward[tl][k] += r
        if self._current_time >= self.episode_length_time:
            done = True
            self.terminate()

        return done, self.all_reward

    def reset(self):
        """have to terminate previous sim before calling reset"""
        self._current_time = 0
        self._init_sim(self.cfg.get("sumocfg_file"), self.seed, self.config.get("episode_length_time"), self.cfg.get("gui"))
        self.step_num = self.step_num % 1400 + 1
        # self.infastructure = self._infastructure_extraction1(self.cfg.get("sumocfg_file"))
        self.vehicle_info.clear()
        self.all_reward = {}
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl] = Intersection(tl, self.infastructure[tl], self, self.state_key, self.not_default)
            self._crosses[tl].update_timestep()
        return self._get_state()

    def reset_default(self):
        self._current_time = 0
        self._init_sim(self.cfg.get("sumocfg_file"), self.seed, self.config.get("episode_length_time"), self.cfg.get("gui"))
        self.infastructure = self._infastructure_extraction1(self.cfg.get("sumocfg_file"))
        self.vehicle_info.clear()
        self.all_reward = {}
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl] = Intersection(tl, self.infastructure[tl], self, self.state_key,
                                             self.not_default)
            self._crosses[tl].update_timestep()

    def _get_state(self):
        states = {}
        for tid in self.all_tls:
            states[tid] = self._crosses[tid].get_state()
        return states

    def _infastructure_extraction1(self, sumocfg_file):
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        network_file = os.path.join(os.path.split(sumocfg_file)[0], network_file_name)
        net = xml.etree.ElementTree.parse(network_file).getroot()

        traffic_light_node_dict = {}
        for tl in net.findall("tlLogic"):
            if tl.attrib['id'] not in traffic_light_node_dict.keys():
                node_id = tl.attrib['id']
                traffic_light_node_dict[node_id] = {'leaving_lanes': [], 'entering_lanes': [],
                                                    'leaving_lanes_pos': [], 'entering_lanes_pos': [],
                                                    # "total_inter_num": None,
                                                    'adjacency_row': None}
                traffic_light_node_dict[node_id]["phases"] = [child.attrib["state"] for child in tl]

        # for index, item in enumerate(traffic_light_node_dict):
        #     traffic_light_node_dict[item]['total_inter_num'] = total_inter_num

        for edge in net.findall("edge"):
            if not edge.attrib['id'].startswith(":"):
                if edge.attrib['from'] in traffic_light_node_dict.keys():
                    for child in edge:
                        if "id" in child.keys() and child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['from']]['leaving_lanes'].append(
                                child.attrib['id'])
                            traffic_light_node_dict[edge.attrib['from']]['leaving_lanes_pos'].append(child.attrib['shape'])
                if edge.attrib['to'] in traffic_light_node_dict.keys():
                    for child in edge:
                        if "id" in child.keys() and child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['to']]['entering_lanes'].append(child.attrib['id'])
                            traffic_light_node_dict[edge.attrib['to']]['entering_lanes_pos'].append(child.attrib['shape'])

        for junction in net.findall("junction"):
            if junction.attrib['id'] in traffic_light_node_dict.keys():
                traffic_light_node_dict[junction.attrib['id']]['location'] = \
                    {'x': float(junction.attrib['x']), 'y': float(junction.attrib['y'])}

        return traffic_light_node_dict

    def bfs_find_neighbor(self, queue, edge_dict, net_lib, tl):
        seen = set()
        seen.add(tl)
        parents = {tl: None}
        while len(queue) > 0:
            if edge_dict[queue[0]].attrib['from'] != tl:
                if edge_dict[queue[0]].attrib['from'] not in self.all_tls:
                    next_node = edge_dict[queue[0]].attrib['from']
                    if next_node not in seen:
                        seen.add(next_node)
                        next_tmp_edges = net_lib.getNode(next_node).getIncoming()
                        next_edges = {}
                        for nte in next_tmp_edges:
                            if edge_dict[nte._id].attrib['from'] == edge_dict[queue[0]].attrib['to']:
                                continue
                            # net_lib.getShortestPath(SC, CE)
                            c1 = net_lib.getNode(edge_dict[nte._id].attrib['from']).getCoord()
                            c2 = net_lib.getNode(next_node).getCoord()
                            c1 = {"x": c1[0], "y": c1[1]}
                            c2 = {"x": c2[0], "y": c2[1]}
                            next_edges[nte._id] = self._cal_distance(c1, c2)
                        if len(next_edges) > 0:
                            next_edges = sorted(next_edges.items(), key=lambda item: item[1])
                            next_edges = [ne[0] for ne in next_edges]
                        queue.extend(next_edges)
                        parents[next_node] = edge_dict[queue[0]].attrib['to']
                    del queue[0]
                else:
                    assert edge_dict[queue[0]].attrib['from'] in self.all_tls and edge_dict[queue[0]].attrib['from'] != tl, "{}-{}".format(edge_dict[queue[0]].attrib['from'], tl)
                    parents[edge_dict[queue[0]].attrib['from']] = edge_dict[queue[0]].attrib['to']
                    return True, parents, edge_dict[queue[0]].attrib['from']
            else:
                del queue[0]
        return False, None, None
            # length += self._cal_distance(
            #     traffic_light_node_dict[
            #         edge_dict[entering_sequence[es]].attrib['from']]['location'],
            #     traffic_light_node_dict[
            #         edge_dict[entering_sequence[es]].attrib['to']]['location']
            # )

    def _infastructure_extraction2(self, sumocfg_file, traffic_light_node_dict, dis=False):
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        network_file = os.path.join(os.path.split(sumocfg_file)[0], network_file_name)
        net = xml.etree.ElementTree.parse(network_file).getroot()
        net_lib = sumolib.net.readNet(network_file)

        # all_tls is deleted
        all_traffic_light = self.all_tls
        total_inter_num = len(self.all_tls)
        if dis:
            top_k = self.adjacency_top_k
            for i in range(total_inter_num):
                if 'location' not in traffic_light_node_dict[all_traffic_light[i]]:
                    continue
                location_1 = traffic_light_node_dict[all_traffic_light[i]]['location']
                row = np.array([0] * total_inter_num)
                for j in range(total_inter_num):
                    if 'location' not in traffic_light_node_dict[all_traffic_light[j]]:
                        row[j] = 1e8
                        continue
                    location_2 = traffic_light_node_dict[all_traffic_light[j]]['location']
                    dist = self._cal_distance(location_1, location_2)
                    row[j] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = list(range(total_inter_num))

                if self.is_adjacency_remove:
                    adjacency_row_unsorted.remove(i)
                    
                adjacency_row_unsorted = [j for j in adjacency_row_unsorted]
                traffic_light_node_dict[all_traffic_light[i]]['adjacency_row'] = \
                    [[adjacency_row_unsorted, row[adjacency_row_unsorted]], []]
        else:
            # adjacency_row is [entering NWSE neighbor, outgoing NWSE neighbor]
            #  NWSE neighbor contain: 1. neighbor tl name 2. edge distance

            edge_dict = {}
            for edge in net.findall("edge"):
                edge_dict[edge.attrib['id']] = edge

            junction_dict = {}
            for jun in net.findall("junction"):
                junction_dict[jun.attrib["id"]] = jun

            for i in self.all_tls:
                # if i == '89173763':
                # if i == 'J36':
                #     print(1)
                entering_sequence_NWSE = self._crosses[i].entering_sequence_NWSE
                entering_sequence = self._crosses[i].entering_sequence
                outgoing_sequence_NWSE = self._crosses[i].outgoing_sequence_NWSE
                outgoing_sequence = self._crosses[i].outgoing_sequence
                adjacency_row_entering = []
                adjacency_distance_entering = []
                adjacency_row_outgoing = []
                adjacency_distance_outgoing = []
                for es in entering_sequence_NWSE:
                    if es != -1:
                        queue = [entering_sequence[es]]
                        # print(i)
                        flag, parents, last_node = self.bfs_find_neighbor(queue, edge_dict, net_lib, i)
                        # while edge_dict[queue[0]].attrib['from'] not in all_traffic_light:
                        #     next_node = edge_dict[entering_sequence[es]].attrib['from']
                        #     next_tmp_edges = net_lib.getNode(next_node).getIncoming()
                        #     for index, nte in enumerate(next_tmp_edges):
                        #         if edge_dict[nte].attrib['from'] == i:
                        #             rm_index = index
                        #             break
                        #     del next_tmp_edges[rm_index]
                        #     queue.extend(next_tmp_edges)
                        if flag:
                            length = 0
                            last_node_ = last_node
                            while parents[last_node] != None:
                                coor1 = net_lib.getNode(last_node).getCoord()
                                coor2 = net_lib.getNode(parents[last_node]).getCoord()
                                c1 = {"x": coor1[0], "y": coor1[1]}
                                c2 = {"x": coor2[0], "y": coor2[1]}
                                length += self._cal_distance(c1, c2)
                                if length > 700:
                                    break
                                last_node = parents[last_node]
                            if length > 700:
                                no_neigh = True
                            else:
                                no_neigh = False
                        else:
                            no_neigh = True
                    else:
                        no_neigh = True
                    if no_neigh:
                        adjacency_row_entering.append(-1)
                        adjacency_distance_entering.append(1e8)
                    else:
                        adjacency_row_entering.append(
                            self.all_tls.index(last_node_))
                        adjacency_distance_entering.append(length / 100)
                for os_ in outgoing_sequence_NWSE:
                    if os_ != -1 and edge_dict[outgoing_sequence[os_]].attrib['to'] in all_traffic_light:
                        adjacency_row_outgoing.append(
                            self.all_tls.index(edge_dict[outgoing_sequence[os_]].attrib['to']))
                        adjacency_distance_outgoing.append(self._cal_distance(
                            traffic_light_node_dict[
                                edge_dict[outgoing_sequence[os_]].attrib['from']]['location'],
                            traffic_light_node_dict[
                                edge_dict[outgoing_sequence[os_]].attrib['to']]['location']
                        )/100)
                    else:
                        adjacency_row_outgoing.append(-1)
                        adjacency_distance_outgoing.append(1e8)
                traffic_light_node_dict[i]['adjacency_row'] = \
                    [(adjacency_row_entering, adjacency_distance_entering),
                     (adjacency_row_outgoing, adjacency_distance_outgoing)]

        # debug
        # for i in self.all_tls:
        #     print(i)
        #     for j in traffic_light_node_dict[i]['adjacency_row'][0][0]:
        #         if j != -1:
        #             print(self.all_tls[j])
        #     print("")
        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def _coordinate_sequence(list_coord_str):
        import re
        list_coordinate = [re.split(r'[ ,]', lane_str) for lane_str in list_coord_str]
        # x coordinate
        x_all = np.concatenate(list_coordinate).astype('float64')
        west = np.int(np.argmin(x_all)/2)

        y_all = np.array(list_coordinate, dtype=float)[:, [1, 3]]

        south = np.int(np.argmin(y_all)/2)

        east = np.int(np.argmax(x_all)/2)
        north = np.int(np.argmax(y_all)/2)

        list_coord_sort = [west, north, east, south]
        return list_coord_sort

    @staticmethod
    def _sort_lane_id_by_sequence(ids,sequence=[2, 3, 0, 1]):
        result = []
        for i in sequence:
            result.extend(ids[i*3: i*3+3])
        return result

    @staticmethod
    def get_actual_lane_id(lane_id_list):
        actual_lane_id_list = []
        for lane_id in lane_id_list:
            if not lane_id.startswith(":"):
                actual_lane_id_list.append(lane_id)
        return actual_lane_id_list


if __name__ == '__main__':
    config = {
        "name": "colight",
        "agent": "",
        "sumocfg_file": "rl-tsc/sumo_files/scenarios/nanshan/osm.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg",

        "action_type": "select_phase",
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'state_key': ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num', 'pressure']
    }
    env = TSCSimulator(config, 1234)
    env.reset()
    all_tl = env.all_tls
    for i in range(100):
        tl_action_select = {}
        for tl in all_tl:
            a = np.random.choice(env._crosses[tl].green_phases)
            while a / 2 in env._crosses[tl].unava_index:
                a = np.random.choice(env._crosses[tl].green_phases)
            tl_action_select[tl] = a
        next_obs, reward, done, _ = env.step(tl_action_select)
    env.terminate()
