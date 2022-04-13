import json
import os

import matplotlib.pyplot as plt
from robustperiod import robust_period, robust_period_full, plot_robust_period

import numpy as np
import csv

def robust_period_process(param_pack):
    (timemap, db, num_wavelets, lmb, c, zeta, precision, logpath, task_num) = param_pack
    print("[RobustPeriod] Task %d running..." % (task_num))
    periods, W, bivar, periodograms, p_vals, ACF = robust_period_full(timemap, 'db10', num_wavelets, lmb, c, zeta)
    periods = (periods * precision).tolist()
    with open(logpath, "w") as f:
        json.dump({"Periods": periods}, f)
    print("[RobustPeriod] Task %d finish." % (task_num))
    return periods, W, bivar, periodograms, p_vals, ACF


'''
    多进程地使用 robust_period 程序对数据进行分析，并输出结果
'''
def robust_period_drive_multiprocess(logreader, tm, lmb=1e+6, c=2, num_wavelets=5, zeta=1.345, path=".", cpu_worker_num=4):
    # Create process pool, map thread, reduce thread
    from multiprocessing import Pool

    # Time
    import time
    start_time = time.time()

    # Params
    event_period_flag = {}  # 与 event_period_data 同为 {uid: 数据} 形式的集合
    event_period_data = {}

    # Loop to create tasks
    task_map = {}       # {uid: 任务编号（从0开始）}
    task_list = []
    task_list_count = 0
    for i in logreader:
        unique_id = i.unique_id()
        if unique_id in task_map:
            continue

        timemap, precision, point_count = tm.resolve(event_id=[unique_id])

        # 当只存在1个或2个数据点时，作没周期处理
        if point_count == 1 or point_count == 2:
            task_map[unique_id] = -1
            continue

        task_map[unique_id] = task_list_count
        task_list_count += 1
        param = (timemap, 'db10', num_wavelets, lmb, c, zeta, precision, os.path.join(path, unique_id + ".tasklog"), task_list_count)
        task_list.append(param)

    # DEBUG
    print("[POOL] %d tasks in total." % (len(task_list)))

    # Use Multiprocessing to process this
    with Pool(cpu_worker_num) as p:
        outputs = p.map(robust_period_process, task_list)

    # Process results
    for i in logreader:
        unique_id = i.unique_id()
        if unique_id in task_map and task_map[unique_id] != -1 and len(outputs[task_map[unique_id]][0]) != 0:
            periods = outputs[task_map[unique_id]][0]
            event_period_data[unique_id] = periods
            event_period_flag[unique_id] = 1
        else:
            event_period_data[unique_id] = [0.0, 0.0, 0.0, 0.0, 0.0]
            event_period_flag[unique_id] = 0

    # Print time elapsed
    print("--- Robust Period runs for %s seconds (on %d entries) ---" % (time.time() - start_time, len(task_list)))

    return event_period_flag, event_period_data


'''
    (已废弃，未使用）使用 robust_period 程序对数据进行分析，并输出结果
'''
def robust_period_drive(logreader, tm, lmb=1e+6, c=2, num_wavelets=5, zeta=1.345, path="."):
    event_period_flag = {}      # 与 event_period_data 同为 {uid: 数据} 形式的集合
    unique_id_flag = {}         # 与 unique_id_data 同为 {uid: 数据} 形式的集合
    event_period_data = {}
    unique_id_data = {}

    # Log path
    data_path = os.path.join(path, "event_period_data.log")

    # CSV file 作为备份 输出备份数据（防止程序异常终止而丢失数据）
    with open(data_path, 'w', newline='') as csvfile:
        sparewriter = csv.writer(csvfile)

        for i in logreader:
            uid = i.uid()
            unique_id = i.unique_id()
            if unique_id in unique_id_data:
                event_period_flag[uid] = unique_id_flag[unique_id]
                event_period_data[uid] = unique_id_data[unique_id]
                sparewriter.writerow([unique_id, uid, unique_id_flag[unique_id],
                                      unique_id_data[unique_id]])
                continue

            timemap, precision, point_count = tm.resolve(event_id=[unique_id])
            print("Unique_id: %d, uid: %s, precision: %f with %d points." %
                  (unique_id, uid, precision, point_count))

            # 当只存在1个或2个数据点时，作没周期处理
            if point_count == 1 or point_count == 2:
                unique_id_flag[unique_id] = 0
                event_period_flag[uid] = 0
                unique_id_data[unique_id] = [0.0]
                event_period_data[uid] = [0.0]
                sparewriter.writerow([unique_id, uid, unique_id_flag[unique_id],
                                     unique_id_data[unique_id]])
                continue

            # ---------- Robust Period --------------

            periods, W, bivar, periodograms, p_vals, ACF = robust_period_full(
                timemap, 'db10', num_wavelets, lmb, c, zeta)
            # plot_robust_period(periods, W, bivar, periodograms, p_vals, ACF)
            print(periods)

            if len(periods) == 0:
                unique_id_flag[unique_id] = 0
                event_period_flag[uid] = 0
                unique_id_data[unique_id] = [0.0]
                event_period_data[uid] = [0.0]
            else:
                unique_id_flag[unique_id] = 1
                event_period_flag[uid] = 1
                unique_id_data[unique_id] = (periods * precision).tolist()
                event_period_data[uid] = (periods * precision).tolist()

            # Save while processing
            sparewriter.writerow([unique_id, uid, unique_id_flag[unique_id],
                                 unique_id_data[unique_id]])


    return event_period_flag, event_period_data


