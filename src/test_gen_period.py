import random
import pandas as pd
import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from src.robustperiod.robustperiod import robust_period_full

#生成周期性检测的测试数据
test_time_map = np.zeros(1000)
precision=1
for i in range(0, 999, 5):
    test_time_map[i] += 1
for i in range(6, 999, 9):
    test_time_map[i] += 1
for i in range(3, 999, 25):
    test_time_map[i] += 1
for i in range(20, 999, 43):
    test_time_map[i] += 1

for i in range(20):
    test_time_map[random.randint(0,999)]+=1
print(test_time_map)
np.save("../out/test_time_map.npy", test_time_map)
print("save .npy done")
periods, W, bivar, periodograms, p_vals, ACF = robust_period_full(
    test_time_map, 'db10', num_wavelet=5, lmb=1e+6, c=2, zeta=1.345)
print(periods.tolist())
#fftTransfer(test_time_map)
x = np.arange(len(test_time_map))  # x轴
plt.plot(x,test_time_map)
plt.show()
test_npytocsv = pd.DataFrame(data = test_time_map)
test_npytocsv.to_csv('../out/test_time_map.csv')

#对比傅里叶分析和自相关系数结合起来判断周期性的方法
fft_series = fft(test_time_map)
power = np.abs(fft_series)
plt.plot(power)
plt.show()
sample_freq = fftfreq(fft_series.size)
plt.plot(sample_freq)
plt.show()

pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
powers = power[pos_mask]

top_k_seasons = 100
top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
top_k_power = powers[top_k_idxs]
fft_periods = (1 / freqs[top_k_idxs]).astype(int)

# Expected time period
for lag in fft_periods:
    # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
    acf_score = acf(test_time_map, nlags=lag)[-1]
    if acf_score>0:
        print(f"lag: {lag} fft acf: {acf_score}")



