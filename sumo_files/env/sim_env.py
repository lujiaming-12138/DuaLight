#!/usr/bin/env python3
# encoding: utf-8
"""
@paper: A GENERAL SCENARIO-AGNOSTIC REINFORCEMENT LEARNING FOR TRAFFIC SIGNAL CONTROL
@file: ts_control
"""
import os
import subprocess
import time

import numpy as np

import traci
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
        self.cfg = config
        self.step_num = config.get("step_num", 1)
        self._current_time = 0
        self._init_sim(config.get("sumocfg_file"), self.seed, self.config.get("episode_length_time"), config.get("gui"))
        self.all_tls = list(self.sim.trafficlight.getIDList())
        self.infastructure = self._infastructure_extraction(config.get("sumocfg_file"))
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
            self._crosses[tl] = Intersection(tl, self.infastructure[tl], self, self.state_key, self.not_default)
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
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport',
                        '600']  # long teleport for safety
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            if not os.path.exists(self.output_path):
                try:
                    os.mkdir(self.output_path)
                except:
                    pass
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_%s_trip.xml' % (
                            self.name, sumocfg_file.split("/")[-1], self.step_num)),
                        '--tripinfo-output.write-unfinished']
        subprocess.Popen(command)
        self.step_num += 1
        # wait 2s to establish the traci server
        time.sleep(5)

        self.sim = traci.connect(port=self.port, numRetries=500)
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
        reward = self._get_reward()
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
        self.infastructure = self._infastructure_extraction(self.cfg.get("sumocfg_file"))
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
        self.infastructure = self._infastructure_extraction(self.cfg.get("sumocfg_file"))
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

    def _infastructure_extraction(self, sumocfg_file):
        import xml.etree.ElementTree
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
        total_inter_num = len(self.all_tls)
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

        top_k = 5
        all_traffic_light = self.all_tls
        for i in range(total_inter_num):
            if 'location' not in traffic_light_node_dict[all_traffic_light[i]]:
                continue
            location_1 = traffic_light_node_dict[all_traffic_light[i]]['location']
            row = np.array([0]*total_inter_num)
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

            adjacency_row_unsorted.remove(i)
            traffic_light_node_dict[all_traffic_light[i]]['adjacency_row'] = [i] + adjacency_row_unsorted

        # get lane length

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

        list_coord_sort=[west,north,east,south]
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
        "name": "test",
        "agent": "",
         # "sumocfg_file": "sumo_files/scenarios/nanshan/osm.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
        "sumocfg_file": "sumo_files/scenarios/sumo_wj3/rl_wj.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/real_net/most_0.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
        "action_type": "select_phase",
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'state_key': ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num', 'pressure']
    }
    env = TSCSimulator(config, 123)
    env.reset()
    all_tl = env.all_tls
    for i in range(100):
        tl_action_select = {}
        for tl in all_tl:
            tl_action_select[tl] = np.random.choice(env._crosses[tl].green_phases)
        next_obs, reward, done, _ = env.step(tl_action_select)
    env.terminate()
