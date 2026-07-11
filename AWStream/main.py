import shutil
import csv
import cv2
import os
import math
import pandas as pd
import env
import subprocess
from utils import get_video_bit, calculate_marco_f1, format_conversion, rewrite, load_one_trace
import numpy as np
import json
from bisect import bisect_left
import scipy.stats as stats

# 33
VIDEO_BIT_RATE = [34.45444444444444, 44.24555555555559, 51.623333333333306, 58.94555555555559, 76.15666666666665,
                  88.5288888888889, 103.91888888888894, 105.67777777777783, 140.4288888888889, 166.67555555555586,
                  193.57000000000028, 225.0855555555556, 238.65888888888875, 257.1055555555561, 303.64777777777783,
                  322.6011111111113, 385.41777777777776, 528.026666666667, 597.0666666666666, 710.7733333333331,
                  780.3055555555551, 961.484444444446, 1078.8922222222218, 1170.9877777777767, 1335.9466666666676,
                  1406.736666666667, 1823.6900000000016, 1923.0044444444457, 2573.1711111111053, 3465.591111111112,
                  3533.948888888885, 6508.732222222213, 11958.148888888883]

CHUNK_LENGTH = 2
alpha = 0.8

with open('pareto_list.txt.txt', 'r') as file:
    content = file.read()
C = json.loads(content)
C = np.array(C, dtype=object)


def test():
    cooked_bw, cooked_name = load_one_trace('../dataset/', '4G')
    net_env = env_12h.Environment(cooked_bw=cooked_bw)

    state = 'Startup'
    bit = 10
    qp, s, r = select_c(bit)

    while True:
        QE, QC, RC, R, SProbeDone, end_of_video = net_env.get_video_chunk(state, bit, qp, s, r)
        if end_of_video != True:
            state, bit = regulate_source(state, bit, QE, QC, RC, R, SProbeDone)
            qp, s, r = select_c(bit)
        else:
            # f1_mean = np.mean(net_env.F1)
            # f1_std = np.std(net_env.F1, ddof=1)
            # f1_standard_error = f1_std / np.sqrt(len(net_env.F1))
            # f1_interval = stats.norm.interval(0.95, loc=f1_mean, scale=f1_standard_error)
            #
            # lag_mean = np.mean(net_env.lag)
            # lag_std = np.std(net_env.lag, ddof=1)
            # lag_standard_error = lag_std / np.sqrt(len(net_env.lag))
            # lag_interval = stats.norm.interval(0.95, loc=lag_mean, scale=lag_standard_error)
            #
            return (net_env.F1, net_env.lag, net_env.bw_use, net_env.Reward, cooked_name,
                    net_env.lag_1, net_env.lag_2, net_env.lag_3, net_env.lag_4, net_env.lag_5)


def select_c(bit):
    index = len(VIDEO_BIT_RATE) - 1 - bit
    return C[index][0], C[index][1], C[index][2]


def regulate_source(state, last, qe, qc, rc, r, sprobedone):
    next_state = state
    if state == 'Startup':
        if qc or rc:
            index = bisect_left(VIDEO_BIT_RATE, alpha * r)
            n = max(0, index - 1)
            if n > last:
                n = last
            next_state = 'Degrade'
        elif qe:
            n = min(len(VIDEO_BIT_RATE) - 1, last+1)
            if n == len(VIDEO_BIT_RATE) - 1:
                next_state = 'Steady'
            else:
                next_state = 'Startup'
        else:
            n = last
            next_state = 'Steady'

    elif state == 'Degrade':
        if qc or rc:
            index = bisect_left(VIDEO_BIT_RATE, alpha * r)
            n = max(0, index - 1)
            if n > last:
                n = last
            next_state = 'Degrade'
        else:
            n = last
            next_state = 'Steady'

    elif state == 'Steady':
        if qc or rc:
            index = bisect_left(VIDEO_BIT_RATE, alpha * r)
            n = max(0, index - 1)
            if n > last:
                n = last
            next_state = 'Degrade'
        elif qe and last < len(VIDEO_BIT_RATE) - 1:
            if sprobedone:
                n = min(len(VIDEO_BIT_RATE) - 1, last + 1)
                next_state = 'Steady'
            else:
                n = last
                next_state = 'Steady'
        else:
            n = last
            next_state = 'Steady'
    else:
        print(f"Unknown state: {state}")
        n = 10
    return next_state, n

if __name__ == '__main__':
    output_dir = f'Results'
    os.makedirs(output_dir, exist_ok=True)

    F1, lag, bw_use, reward, bw_name, lag_1, lag_2, lag_3, lag_4, lag_5 = test()
    with open(f'{output_dir}/test.txt', 'w') as f:
        for i in range(len(F1)):
            f.write(f"{F1[i]} {lag[i]} {bw_use[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")


