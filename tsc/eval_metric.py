import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tsc.multi_agent.qmix_eval_config import map_configs, name, metrics #从evalconfig文件中读取需要评测的地图和实验信息

log_dir = os.path.join(os.getcwd(), 'sumo_logs' + os.sep)#保存评估文件的地址
trials = [folder for folder in next(os.walk(log_dir + f"eval_{name}" + os.sep))[1]]#重复实验对应的文件夹

for metric in metrics:
    output_file = log_dir + f"eval_{name}" + os.sep + 'avg_{}.py'.format(metric)
    run_avg = dict()

    for trial in trials:#重复实验
        for map_name in map_configs:#地图名
            average_per_episode = []
            for i in range(2, 1000, 2):#偶数结尾的文件才有数据
                #trip_file_name = log_dir + names + os.sep + names + "_" + map_name + ".sumocfg_{}_trip.xml".format(i)
                trip_file_name = log_dir + f"eval_{name}" + os.sep + trial + os.sep + name + "_" + map_name + ".sumocfg_{}_trip.xml".format(i)
                if not os.path.exists(trip_file_name):
                    print('No '+trip_file_name)
                    break            
                try:
                    tree = ET.parse(trip_file_name)
                    root = tree.getroot()
                    num_trips, total = 0, 0.0
                    last_departure_time = 0
                    last_depart_id = ''
                    for child in root:
                        try:
                            num_trips += 1
                            total += float(child.attrib[metric])
                            if metric == 'timeLoss':
                                total += float(child.attrib['departDelay'])
                                depart_time = float(child.attrib['depart'])
                                if depart_time > last_departure_time:
                                    #记录trip文件中，最晚出发的车辆和出发时间
                                    last_departure_time = depart_time
                                    last_depart_id = child.attrib['id']
                        except Exception as e:
                            #raise e
                            break
                    
                    if metric == 'timeLoss' and map_name in ['cologne8', 'ingolstadt21', 'base', 'rl_wj']:    # Calc. departure delays
                        #只有具体的每辆车depart的时间的地图才能计算，其他地图车流随机生成，没有固定的depart时间
                        #对delay的计算影响不是很大
                        
                        #读取route文件获取预置的车流信息
                        route_file_name = os.getcwd() + os.sep + map_configs[map_name]['route']
                        tree = ET.parse(route_file_name)

                        root = tree.getroot()
                        last_departure_time = None
                        for child in root:
                            if child.attrib['id'] == last_depart_id: #找到trip中最后出发的车辆的信息
                                last_departure_time = float(child.attrib['depart'])     # Get the time it was suppose to depart
                        never_departed = []
                        if last_departure_time is None: raise Exception('Wrong trip file')
                        for child in root:
                            if child.tag != 'vehicle': continue
                            depart_time = float(child.attrib['depart'])
                            if depart_time > last_departure_time:
                                #记录所有晚于最后出发时间的  未出发时间
                                never_departed.append(depart_time)
                        never_departed = np.asarray(never_departed)
                        #计算没有出发车辆的delay
                        never_departed_delay = np.sum(float(map_configs[map_name]['end_time']) - never_departed)
                        #把没有出发车辆的delay加入到延迟中
                        total += never_departed_delay
                        num_trips += len(never_departed)

                    average = total / num_trips
                    average_per_episode.append(average)
                except ET.ParseError as e:
                    #raise e
                    break
                
            run_name = map_name
            average_per_episode = np.asarray(average_per_episode)

            if run_name in run_avg:
                run_avg[run_name].append(average_per_episode)
            else:
                run_avg[run_name] = [average_per_episode]
            
    alg_res = []
    alg_name = []
    for run_name in run_avg:
        list_runs = run_avg[run_name]
        min_len = min([len(run) for run in list_runs])
        list_runs = [run[:min_len] for run in list_runs]
        avg_delays = np.sum(list_runs, 0)/len(list_runs)
        err = np.std(list_runs, axis=0)#计算1-sigma的error bar，使用重复实验数据计算

        alg_name.append(run_name)
        alg_res.append(avg_delays)

        alg_name.append(run_name+'_yerr')
        alg_res.append(err)


    np.set_printoptions(threshold=sys.maxsize)
    with open(output_file, 'a') as out:#保存到实验log对应的目录下
        out.write(f"{metric} = ")
        out.write("{ \n")
        for i, res in enumerate(alg_res):
            out.write("'{}': {},\n".format(name + ' ' + alg_name[i], res.tolist()))
        out.write("} \n")
