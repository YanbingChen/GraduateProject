import numpy as np
from timeline_mapping import TimelineMap
from log_reader import LogReader

import matplotlib.pyplot as plt
from robustperiod import robust_period, robust_period_full, plot_robust_period
from robust_period_drive import robust_period_drive

import json

'''
    main.py 它做了什么：
    1 读取 conn.log 并将数据载入程序
    2 运行 timeline mapping 任务，对序列进行映射
    3 调用 robust_period_drive 运行 robust_period 任务
    4 将计算所得周期存入文件
'''

# ------ Pre process & Timeline mapping ------
logreader = LogReader("../zeek/conn.log", ["id.orig_h", "id.resp_h"])
tm = TimelineMap()
for i in logreader:
    tm.insert(i.unique_id(), i.get_timestamp())

event_period_flag, event_period_data = \
    robust_period_drive(logreader, tm, path="../out")

# print(event_period_data)
with open("../out/event_period_flag.txt", 'w') as f:
    json.dump(event_period_flag, f)
with open("../out/event_period_data.txt", 'w') as f:
    json.dump(event_period_data, f)


# Store as numpy
# logs_py = logreader.to_numpy()