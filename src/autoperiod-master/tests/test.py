import os

#测试autoperiod 内部改成多周期判断 不止输出峰值周期
import numpy as np
import matplotlib.pyplot as plt

from autoperiod.autoperiod import Autoperiod
from autoperiod.helpers import load_google_trends_csv, load_gpfs_csv
from autoperiod.plotting import Plotter

times, values = load_gpfs_csv(os.path.join(os.getcwd(), "../tests/data/test_time_map_1.csv"))
p = Autoperiod(times, values, plotter=Plotter())
times, values = load_gpfs_csv(os.path.join(os.getcwd(), "../tests/data/test_time_map.csv"))
p = Autoperiod(times, values, plotter=Plotter())
#times, values = load_google_trends_csv(os.path.join(os.getcwd(), "../tests/data/trends_easter.csv"))
#p = Autoperiod(times, values, plotter=Plotter())
print(p.period)

