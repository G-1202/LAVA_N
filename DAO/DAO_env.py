import os
import csv
import shutil
import math
import pandas as pd
import numpy as np
from DAO_utils import encoder, get_video_bit, detect_data, calculate_marco_f1, bandwidth_predictor, C_R


VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
resolutions = ['480p', '720p', '1080p']
button = 33
re = [[720, 480], [1280, 720], [1920, 1080]]

Length = 2
FPS = 30
BW_estimate = 5
RTT = 0.08

INFER_T = 0.0025
FRAME = [60, 30, 20, 10]

df = pd.read_csv(f'coding_time.csv')
ENCODE_T = df['mean_time'].tolist()
class Environment:
    def __init__(
        self,
        cooked_bw,
        cooked_name,
        start,
        chunk_start
    ):
        self.start_fix = start
        self.df = pd.read_hdf(f'..dataset/AD.h5', 'encoding_data')
        self.chunk = CHUNK_NUM[v_id]
        self.video_frame_start = 0
        self.video_frame = self.video_frame_start
        self.video_chunk_counter = 1
        self.cooked_bw = cooked_bw *2
        self.bw_name = cooked_name

        #  start point of the trace
        self.start = start
        self.c_start = chunk_start
        self.video_start_shoot = self.start - 2
        self.server_bu = []
        self.last_end = 0

        self.bw = []
        self.transfer_lag = []
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans (+RTT)
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance
        self.wait_t = []

        self.real_bit = []

        self.F1 = []
        self.bw_use = []
        self.Reward = []

    def get_video_chunk(self, bit, re):

        index = self.video_chunk_counter - 1
        index += self.c_start

        bit = 10 - bit
        sk = 0
        re = 2 - re
        video_chunk_size = self.df.loc[(index, bit, sk, re), 'Size']
        self.bw_use.append(self.df.loc[(index, bit, sk, re), 'Bitrate'] / 1000)
        m = self.df.loc[(index, bit, sk, re), 'Accuracy']
        real_b = self.df.loc[(index, bit, sk, re), 'Bitrate']

        encode_t = ENCODE_T[bit*16+sk*4+re]
        end = self.start + encode_t
        self.lag_1.append(encode_t)
        self.lag_2.append(self.start - self.video_start_shoot - Length)
        bw = 0
        latency = 0
        v_s = video_chunk_size
        while True:
            if math.ceil(end) == end:
                real_bw = self.cooked_bw[int(end + 1)]
                duration = 1
            else:
                real_bw = self.cooked_bw[math.ceil(end)]
                duration = math.ceil(end) - end

            if video_chunk_size - real_bw * 1000 * duration >= 0:
                video_chunk_size = video_chunk_size - real_bw * 1000 * duration
                end += duration
            else:
                end += video_chunk_size / (real_bw * 1000)
                video_chunk_size = 0

            if video_chunk_size == 0:
                latency = end + RTT - self.video_start_shoot - Length
                self.lag_3.append(end - encode_t - self.start + RTT)
                bw = v_s / (self.lag_3[-1])
                bw = bw / 125
                self.bw.append(bw)
                self.start = end + RTT
                break

        l_e = self.last_end
        while self.server_bu:
            if max(l_e, self.server_bu[0][0]) + self.server_bu[0][1] * INFER_T <= end:
                l_e = max(l_e, self.server_bu[0][0]) + self.server_bu[0][1] * INFER_T
                self.server_bu.pop(0)
            else:
                break
        if self.server_bu:
            self.last_end = self.server_bu[0][0]
        else:
            self.last_end = end

        latency = latency + wait_t(self.server_bu) + FRAME[sk] * INFER_T
        self.transfer_lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(FRAME[sk] * INFER_T)

        self.server_bu.append([end, FRAME[sk]])

        self.F1.append(m)
        self.Reward.append(C_R(m, latency))

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        if self.start < self.video_start_shoot + Length:
            if self.video_start_shoot + Length - self.start >= 1:
                self.start = self.video_start_shoot + Length
            else:
                self.start = self.video_start_shoot + Length

        bw_est = self.bw[-1]

        if self.video_chunk_counter > self.TOTAL:
            end_of_video = True
        return bw_est, index + 1, end_of_video

def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else: return 0