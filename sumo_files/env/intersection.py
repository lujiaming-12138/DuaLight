#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Haoyuan Jiang
@file: ts_control
@time: 2022/7/19
"""
import numpy as np
import re
from tsc.utils import get_vector_cos_sim


class Intersection:
    def __init__(self, tl_id, msg, env,
                 state=['current_phase', 'car_num', 'queue_length', "occupancy", 'flow',
                        'stop_car_num'], not_default=True):
        self._id = tl_id
        self._env = env
        self.state = state
        self.nearset_inter = msg['adjacency_row']
        self.location = msg.get('location')
        self.yellow_time = 5
        self.green_time = 30

        self._incoming_lanes = list(
            set(self._env.sim.trafficlight.getControlledLanes(self._id)))
        try:
            self._outgoing_lanes = list(
                set([l[0][1] for l in self._env.sim.trafficlight.getControlledLinks(self._id)]))
        except:
            self.tl_ava = False
            return
        # seq = ["r", "s", "l"]  right always can go though
        self.seq = ["l", "s"]

        self.lans_link = self._env.sim.trafficlight.getControlledLinks(self._id)
        self._incominglanes_links = {}  # from : to: [traffic light index, direction]
        try:
            for index, [(fr, to, _)] in enumerate(self.lans_link):
                if fr not in self._incominglanes_links:
                    self._incominglanes_links[fr] = {to: [index]}
                elif to not in self._incominglanes_links[fr]:
                    self._incominglanes_links[fr][to] = [index]
                else:
                    self._incominglanes_links[fr][to].append(index)
        except:
            self.tl_ava = False
            return
        for in_line in self._incominglanes_links:
            msgs = self._env.sim.lane.getLinks(in_line)
            for m in msgs:
                self._incominglanes_links[in_line][m[0]].append(m[-2])

        self.entering_approaches2phase = {}
        for i in self._incoming_lanes:
            approach = "_".join(i.split("_")[:-1])
            if approach not in self.entering_approaches2phase:
                self.entering_approaches2phase[approach] = {}
            phase_index, directions = list(zip(*list(self._incominglanes_links[i].values())))
            if 's' in directions:
                if 's' not in self.entering_approaches2phase[approach]:
                    self.entering_approaches2phase[approach]['s'] = []
                self.entering_approaches2phase[approach]['s'].append([i, phase_index])
            elif "l" in directions or "L" in directions or 't' in directions and len(
                    directions) == 1:
                if 'l' not in self.entering_approaches2phase[approach]:
                    self.entering_approaches2phase[approach]['l'] = []
                self.entering_approaches2phase[approach]['l'].append([i, phase_index])
            elif "r" in directions or "R" in directions:
                if 'r' not in self.entering_approaches2phase[approach]:
                    self.entering_approaches2phase[approach]['r'] = []
                self.entering_approaches2phase[approach]['r'].append([i, phase_index])
            else:
                print(directions)
                print("Not support")
        self.entering_sequence = list(self.entering_approaches2phase.keys())

        self.outgoing_sequence = list("_".join(i.split("_")[:-1]) for i in msg['leaving_lanes'])


        pair = {}
        for j, v in self._incominglanes_links.items():
            jj = "_".join(j.split("_")[:-1])
            for k, v2 in v.items():
                if v2[1] == 's':
                    if jj not in pair:
                        pair[jj] = []
                    pair[jj].append(k)

        order_entering = []
        for i in self.entering_sequence:
            order_entering.append(msg['entering_lanes'].index(i + "_0"))

        order_outgoing = []
        for i in self.outgoing_sequence:
            order_outgoing.append(msg['leaving_lanes'].index(i + "_0"))

        self.tl_ava, self.entering_sequence_NWSE = self._coordinate_sequence(
            msg["entering_lanes_pos"], order_entering, True, pair, self.entering_sequence)

        tl_av, self.outgoing_sequence_NWSE = self._coordinate_sequence(
            msg["leaving_lanes_pos"], order_outgoing, False, pair, self.outgoing_sequence)
        # assert tl_av == self.tl_ava
        self.tl_ava = tl_av and self.tl_ava
        # v["entering_lanes"] = SumoEnv._sort_lane_id_by_sequence(msg["entering_lanes"] ,sequence=entering_sequence)
        # leaving_sequence = self._coordinate_sequence(msg["leaving_lanes_pos"], order, pair, len(self.sequence))
        # v["leaving_lanes"] = SumoEnv._sort_lane_id_by_sequence(msg["leaving_lanes"] ,sequence=leaving_sequence)
        if self.tl_ava:
            self._phase_index()

            self._lane_vehicle_dict = {}
            self._previous_lane_vehicle_dict = {}

            if not_default:
                self.unava_index = self.reset_tl_logic()
                signal_definition = self._env.sim.trafficlight.getAllProgramLogics(self._id)[-1]
                self._env.sim.trafficlight.setProgram(self._id, "custom")
                self.green_phases = [i for i in range(0, 16, 2)]
                self.yellow_phases = [i for i in range(1, 16, 2)]
                self.phases = []
                for i in signal_definition.phases:
                    self.phases.append(i.state)
        # for idx, phase in enumerate(signal_definition.phases):
        #     if phase.state.count('G') >= 1:
        #         self.green_phases.append(idx)
        #     elif phase.state.count('y') >= 1:
        #         self.yellow_phases.append(idx)

    def get_tl_ava(self):
        return self.tl_ava

    def _phase_index(self):
        """get_phase_index
        The phase index is : north(ls), west(ls), south(ls), east(ls)
        """
        self.phase_index = []
        for i in self.entering_sequence_NWSE:
            if i != -1:
                seq_i = self.entering_sequence[i]
                for s in self.seq:
                    _phase = []
                    if s in self.entering_approaches2phase[seq_i]:
                        for l in self.entering_approaches2phase[seq_i][s]:
                            _phase.extend(list((l[1])))
                        self.phase_index.append(_phase)
                    else:
                        self.phase_index.append([])
            else:
                for _ in self.seq:
                    self.phase_index.append([])

    def get_phase_index(self):
        return self.phase_index

    @staticmethod
    def _coordinate_sequence(list_coord_str, order, entering, pair, seq):
        """Result sequence is north, west, south, east."""
        dim = len(list_coord_str[0].split(" ")[0].split(","))
        if entering:
            list_coordinate = [re.split(r'[ ,]', lane_str)[-2 * dim:]
                               for lane_str in list_coord_str]
            list_coordinate = np.array(list_coordinate, dtype=float)[order]
            if dim == 3:
                list_coordinate = np.concatenate([list_coordinate[:, :2], list_coordinate[:, 3:5]],
                                                 axis=1)
            delta_x = list_coordinate[:, 2] - list_coordinate[:, 0]
            delta_y = list_coordinate[:, 3] - list_coordinate[:, 1]
        else:
            list_coordinate = [re.split(r'[ ,]', lane_str)[: 2 * dim]
                               for lane_str in list_coord_str]
            list_coordinate = np.array(list_coordinate, dtype=float)[order]
            if dim == 3:
                list_coordinate = np.concatenate([list_coordinate[:, :2], list_coordinate[:, 3:5]],
                                                 axis=1)
            delta_x = list_coordinate[:, 0] - list_coordinate[:, 2]
            delta_y = list_coordinate[:, 1] - list_coordinate[:, 3]

        vectors = np.array([[delta_x[i], delta_y[i]] for i in range(len(delta_x))])
        if len(np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y <= 0)])==1:
            north_ = [np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y <= 0)][0]]
            east_, south_, west_ = [], [], []
            for i in range(len(vectors)):
                if i == north_[0]:
                    continue
                sim = vectors[i].dot(vectors[north_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[north_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[north_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    east_.append(i)
                elif theta >= 135 and theta < 225:
                    south_.append(i)
                elif theta >= 225 and theta < 315:
                    west_.append(i)
                else:
                    north_.append(i)
        elif len(np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x <= 0)]) == 1:
            east_ = np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x <= 0)]
            north_, south_, west_ = [], [], []
            for i in range(len(vectors)):
                if i == east_[0]:
                    continue
                sim = vectors[i].dot(vectors[east_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[east_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[east_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    south_.append(i)
                elif theta >= 135 and theta < 225:
                    west_.append(i)
                elif theta >= 225 and theta < 315:
                    north_.append(i)
                else:
                    east_.append(i)
        elif len(np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y > 0)]) == 1:
            south_ = np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y > 0)]
            north_, east_, west_ = [], [], []
            for i in range(len(vectors)):
                if i == south_[0]:
                    continue
                sim = vectors[i].dot(vectors[south_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[south_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[south_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    west_.append(i)
                elif theta >= 135 and theta < 225:
                    north_.append(i)
                elif theta >= 225 and theta < 315:
                    east_.append(i)
                else:
                    south_.append(i)
        elif len(np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x > 0)]) == 1:
            west_ = np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x > 0)]
            north_, east_, south_ = [], [], []
            for i in range(len(vectors)):
                if i == west_[0]:
                    continue
                sim = vectors[i].dot(vectors[west_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[west_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[west_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    north_.append(i)
                elif theta >= 135 and theta < 225:
                    east_.append(i)
                elif theta >= 225 and theta < 315:
                    south_.append(i)
                else:
                    west_.append(i)
        else:
            return False, None
        if len(east_) >= 2 or len(south_) >= 2 or len(north_) >= 2 or len(west_) >= 2:
            return False, None
        east = east_[0] if len(east_) > 0 else -1
        west = west_[0] if len(west_) > 0 else -1
        south = south_[0] if len(south_) > 0 else -1
        north = north_[0] if len(north_) > 0 else -1
        list_coord_sort = [north, west, south, east]  # seq index

        return True, list_coord_sort

    def getPhaseDuration(self):
        return self._env.sim.trafficlight.getPhaseDuration(self._id)

    def getCurrentPhaseIndex(self):
        return self._env.sim.trafficlight.getPhase(self._id)

    def getCurrentPhase(self):
        return self.phases[self.getCurrentPhaseIndex()]

    def getCurrentPhaseDuration(self):
        return self._env.sim.trafficlight.getPhaseDuration(self._id)

    def reset_tl_logic(self):
        """
        Order: A: wt-et B: el-et C: wl-wt D: el-wl E: nt-st F: sl-st G: nt-nl H: nl-sl
        """
        logic = []
        unava_index = []
        right_index = []
        for i in range(len(self.lans_link)):
            if i not in np.concatenate(np.array(self.phase_index)):
                right_index.append(i)
        A = self.phase_index[3] + self.phase_index[7]
        B = self.phase_index[6] + self.phase_index[7]
        C = self.phase_index[3] + self.phase_index[2]
        D = self.phase_index[6] + self.phase_index[2]
        E = self.phase_index[1] + self.phase_index[5]
        F = self.phase_index[4] + self.phase_index[5]
        G = self.phase_index[1] + self.phase_index[0]
        H = self.phase_index[4] + self.phase_index[0]

        index2dirc = {}
        for _, v in self._incominglanes_links.items():
            for _, pair in v.items():
                index2dirc[pair[0]] = pair[1]

        sequence = [A, B, C, D, E, F, G, H]
        for i in range(len(sequence)):
            if len(sequence[i]) == 0:
                phase_ = list("r" * len(self.lans_link))
                for j in right_index:
                    phase_[j] = 'g'
                phase_ = self.getCurrentPhaseYellow("".join(phase_))
                logic.append([phase_, self.yellow_time])
                logic.append([phase_, self.yellow_time])
                unava_index.append(i)
                # unava_index.append(2*i+1)
            else:
                phase_ = list("r" * len(self.lans_link))
                for j in right_index:
                    phase_[j] = 'g'
                for j in sequence[i]:
                    if index2dirc[j] == 'r':
                        phase_[j] = 'g'
                    else:
                        phase_[j] = 'G'

                for a in self._incominglanes_links.keys():
                    for _, (index, dirt) in self._incominglanes_links[a].items():
                        if dirt == 'r' and phase_[index] != 'G':
                            phase_[index] = 'g'

                logic.append(["".join(phase_), self.green_time])
                logic.append([self.getCurrentPhaseYellow("".join(phase_)), self.yellow_time])

        phases = []
        for i in logic:
            phases.append(
                self._env.sim.trafficlight.Phase(i[1], i[0], 0, 0))
        logic = self._env.sim.trafficlight.Logic("custom", 0, 0, phases)

        self._env.sim.trafficlight.setProgramLogic(self._id, logic)

        return unava_index

    def set_phase_by_index(self, action, duration):
        """Set phase by default tl logic index.
        Note: This function is conflict with set_phase, you can only use one of them in a
        specific action type.
        This is recommended for the action phase is in default tl logic.
        """
        self._env.sim.trafficlight.setPhase(self._id, int(action))
        self._env.sim.trafficlight.setPhaseDuration(self._id, duration)

    def set_phase(self, action, duration):
        """Set phase by tl logic str.
        Note: This function is conflict with set_phase_by_index, you can only use one of them in a
        specific action type.
        This is recommended for the action phase is arbitrary.
        """
        assert isinstance(action, str)
        self._env.sim.trafficlight.setRedYellowGreenState(self._id, action)
        self._env.sim.trafficlight.setPhaseDuration(self._id, duration)

    def getCurrentPhaseYellow(self, currentPhase):
        yellow_phase = list(currentPhase.replace("G", "y").replace("g", "y"))
        for a in self._incominglanes_links.keys():
            # if len(self._incominglanes_links[a].keys()) == 1 and list(self._incominglanes_links[a].values())[0][1] == 'r':
            #     yellow_phase[list(self._incominglanes_links[a].values())[0][0]] = 'g'
            for _, (index, dirt) in self._incominglanes_links[a].items():
                if dirt == 'r':
                    yellow_phase[index] = 'g'

        return "".join(yellow_phase)

    def update_timestep(self):
        self._previous_lane_vehicle_dict = self._lane_vehicle_dict.copy()
        self._update_lane_vehicle_info()

    def get_lane_traffic_volumn(self):
        traffic_volumn_dict = {}
        now_all_incoming_lanes_veh = set()
        for lane in self._incoming_lanes:
            now_all_incoming_lanes_veh.update(self._lane_vehicle_dict[lane])
        for lane in self._incoming_lanes:
            traffic_volumn = 0
            if lane in self._previous_lane_vehicle_dict:
                for veh in self._previous_lane_vehicle_dict[lane]:
                    if veh not in now_all_incoming_lanes_veh:
                        traffic_volumn += 1
            traffic_volumn = traffic_volumn / (
                    self._env._yellow_duration + self._env.iter_duration)
            traffic_volumn_dict[lane] = traffic_volumn
        return traffic_volumn_dict

    def get_lane_queue_len(self):
        queue_len_dict = {}
        for lane in self._incoming_lanes:
            queue_len = self._env.sim.lanearea.getJamLengthMeters(lane)
            queue_len_dict[lane] = queue_len
        return queue_len_dict

    def _update_lane_vehicle_info(self) -> None:
        for lane in self._incoming_lanes + self._outgoing_lanes:
            self._lane_vehicle_dict[lane] = self._env.sim.lane.getLastStepVehicleIDs(lane)
            for veh in self._lane_vehicle_dict[lane]:
                if veh not in self._env.vehicle_info:
                    self._env.vehicle_info[veh] = {}

    def get_lane_car_number(self):
        car_number_dict = {}
        for lane in self._incoming_lanes:
            car_number = self._env.sim.lanearea.getLastStepHaltingNumber(lane)
            car_number_dict[lane] = car_number
        return car_number_dict

    def get_lane_wait_time(self):
        wait_time_dict = {}
        for lane in self._incoming_lanes:
            wait_time = 0
            for veh in self._lane_vehicle_dict[lane]:
                cur_wait_time = self._env.sim.vehicle.getAccumulatedWaitingTime(veh)
                if 'wait_time' not in self._env.vehicle_info[veh]:
                    self._env.vehicle_info[veh]['wait_time'] = 0
                wait_time += cur_wait_time - self._env.vehicle_info[veh]['wait_time']
                self._env.vehicle_info[veh]['wait_time'] = cur_wait_time
            wait_time_dict[lane] = wait_time
        return wait_time_dict

    def get_lane_delay_time(self):
        delay_time_dict = {}
        for lane in self._incoming_lanes + self._outgoing_lanes:
            delay_time = 0
            for veh in self._lane_vehicle_dict[lane]:
                cur_distace = self._env.sim.vehicle.getDistance(veh)
                cur_time = self._env.sim.vehicle.getLastActionTime(veh)
                if 'distance' not in self._env.vehicle_info[veh]:
                    self._env.vehicle_info[veh]['distance'] = cur_distace
                    self._env.vehicle_info[veh]['time'] = cur_time
                else:
                    real_distance = cur_distace - self._env.vehicle_info[veh]['distance']
                    target_speed = self._env.sim.vehicle.getMaxSpeed(veh)
                    self._env.vehicle_info[veh]['distance'] = cur_distace
                    target_distance = (cur_time - self._env.vehicle_info[veh][
                        'time']) * target_speed
                    self._env.vehicle_info[veh]['time'] = cur_time
                    delay_time += (target_distance - real_distance) / (target_speed + 1e-8)
            delay_time_dict[lane] = delay_time
        return delay_time_dict

    def get_pressure(self):
        pressure = 0
        for lane in self._incoming_lanes:
            pressure += self._env.sim.lane.getLastStepHaltingNumber(lane)
        for lane in self._outgoing_lanes:
            pressure -= self._env.sim.lane.getLastStepHaltingNumber(lane)
        return abs(pressure)

    def get_reward(self, reward_type):
        """Queue length, car_number, wait time, delay_time, pressure, speed score"""
        reward = {}
        if 'queue_len' in reward_type:
            queue_len = np.average(list(self.get_lane_queue_len().values())) / 1000
            reward['queue_len'] = -queue_len
        if 'car_number' in reward_type:
            car_number = np.average(list(self.get_lane_car_number().values()))
            reward['car_number'] = -car_number
        if 'wait_time' in reward_type:
            wait_time = np.average(list(self.get_lane_wait_time().values())) / 1000
            reward['wait_time'] = -wait_time
        if 'delay_time' in reward_type:
            delay_time = np.average(list(self.get_lane_delay_time().values())) / 1e4
            reward['delay_time'] = -delay_time
        if 'pressure' in reward_type:
            pressure = self.get_pressure() / 5000
            reward['pressure'] = -pressure

        return reward

    def get_state(self):
        inline_car_number = {}
        inline_occupancy = {}
        inline_speed = {}
        inline_queue_length = {}
        inline_stop_car_number = {}
        inline_stop_car_number_p = {}
        outline_stop_car_number_p = {}
        for i in self._incoming_lanes:
            inline_car_number[i] = self._env.sim.lanearea.getLastStepVehicleNumber(i)
            inline_occupancy[i] = self._env.sim.lanearea.getLastStepOccupancy(i)  # / 100
            inline_speed[i] = self._env.sim.lanearea.getLastStepMeanSpeed(i)
            inline_queue_length[i] = self._env.sim.lanearea.getJamLengthMeters(i)
            inline_stop_car_number[i] = self._env.sim.lanearea.getLastStepHaltingNumber(i)
            inline_stop_car_number_p[i] = self._env.sim.lane.getLastStepHaltingNumber(i)
        for i in self._outgoing_lanes:
            outline_stop_car_number_p[i] = self._env.sim.lane.getLastStepHaltingNumber(i)
        main_direction_lanes = []

        mask = []
        _flow = self.get_lane_traffic_volumn()
        current_phase = self.getCurrentPhase()
        for i in range(len(self.entering_sequence_NWSE)):
            if self.entering_sequence_NWSE[i] != -1:
                seq_i = self.entering_sequence[self.entering_sequence_NWSE[i]]
                for s in self.seq:
                    obs = {}
                    if s in self.entering_approaches2phase[seq_i]:
                        car_num = 0
                        stop_car_num = 0
                        occupancy = 0
                        queue_length = 0
                        speed = 0
                        flow = 0
                        pressure = 0
                        for l in self.entering_approaches2phase[seq_i][s]:
                            car_num += inline_car_number[l[0]]
                            occupancy += inline_occupancy[l[0]]
                            queue_length += inline_queue_length[l[0]]
                            speed += inline_speed[l[0]]
                            stop_car_num += inline_stop_car_number[l[0]]
                            flow += _flow[l[0]]
                            pressure += inline_stop_car_number_p[l[0]]
                        for l in self.entering_approaches2phase[seq_i][s]:
                            for tmp_links in self.lans_link:
                                for (incoming, outging, _) in tmp_links:
                                    if incoming == l[0]:
                                        pressure -= outline_stop_car_number_p[outging]

                        if "car_num" in self.state:
                            obs['car_num'] = car_num / len(self.entering_approaches2phase[seq_i][s])
                        if "stop_car_num" in self.state:
                            obs['stop_car_num'] = stop_car_num / len(
                                self.entering_approaches2phase[seq_i][s])
                        if "occupancy" in self.state:
                            obs['occupancy'] = occupancy / len(self.entering_approaches2phase[seq_i][s])
                        if 'queue_length' in self.state:
                            obs['queue_length'] = queue_length / len(
                                self.entering_approaches2phase[seq_i][s])
                        if "flow" in self.state:
                            obs['flow'] = flow / len(self.entering_approaches2phase[seq_i][s])
                        if "current_phase" in self.state:
                            res = 0 if s == 'l' else 1
                            tmp = 0
                            for kk in self.phase_index[2 * i + res]:
                                if current_phase[kk] in ['g', "G"]:
                                    tmp = 1
                                elif current_phase[kk] == 'r':
                                    tmp = 0
                                    break
                            obs['current_phase'] = tmp
                        if 'pressure' in self.state:
                            obs['pressure'] = max(pressure, 0)

                        main_direction_lanes.append(obs)
                        mask.append(0)
                    else:
                        mask.append(1)
                        main_direction_lanes.append({})
            else:
                for _ in self.seq:
                    mask.append(1)
                    main_direction_lanes.append({})

        return main_direction_lanes, mask, self.nearset_inter
