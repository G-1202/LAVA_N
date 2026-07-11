import numpy as np
import pandas as pd
import math
import csv
import json
from utils import C_R


FPS = 30
Length = 2
Queue_LENGTH = 5

RC_thresholds = 1.5
RTT = 0.08
BW_estimate = 5

INFER_T = 0.0025
FRAME = [60, 30, 20, 10]

df = pd.read_csv(f'coding_time.csv')
ENCODE_T = df['mean_time'].tolist()

# 33
VIDEO_BIT_RATE = [34.45444444444444, 44.24555555555559, 51.623333333333306, 58.94555555555559, 76.15666666666665,
                  88.5288888888889, 103.91888888888894, 105.67777777777783, 140.4288888888889, 166.67555555555586,
                  193.57000000000028, 225.0855555555556, 238.65888888888875, 257.1055555555561, 303.64777777777783,
                  322.6011111111113, 385.41777777777776, 528.026666666667, 597.0666666666666, 710.7733333333331,
                  780.3055555555551, 961.484444444446, 1078.8922222222218, 1170.9877777777767, 1335.9466666666676,
                  1406.736666666667, 1823.6900000000016, 1923.0044444444457, 2573.1711111111053, 3465.591111111112,
                  3533.948888888885, 6508.732222222213, 11958.148888888883]

class Environment:
    def __init__(
        self,
        cooked_bw
    ):
        self.start_fix = 0
        self.TOTAL = 600
        self.df = pd.read_hdf(f'../dataset/AD_QP.h5', 'encoding_data')
        self.chunk = 600
        self.state = 'start'
        self.video_frame_start = 0
        self.video_frame = self.video_frame_start
        self.video_chunk_counter = 1
        self.cooked_bw = cooked_bw

        #  start point of the trace
        self.start = start
        self.c_start = 0
        self.video_start_shoot = self.start - 2
        self.start_shoot_fix = self.video_start_shoot
        self.server_bu = []
        self.last_end = 0

        self.F1 = []
        self.lag = []
        self.lag_1 = [] # encode
        self.lag_2 = [] # camera wait
        self.lag_3 = [] # trans (+RTT)
        self.lag_4 = [] # server wait
        self.lag_5 = [] # inferance

        self.Reward = []
        self.bw_use = []
        self.bw = []

        self.bit = []

        self.QP = []
        self.S = []
        self.R = []


    def get_video_chunk(self, state, bit, qp, s, r):
        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        self.bit.append(bit)
        if len(self.QP) < self.video_chunk_counter:
            self.QP.append(qp)
            self.S.append(s)
            self.R.append(r)

        now_qp = self.QP[self.video_chunk_counter - 1]
        now_s = self.S[self.video_chunk_counter - 1]
        now_r = self.R[self.video_chunk_counter - 1]
        self.state = state
        index = (self.video_chunk_counter - 1) % self.chunk
        index += self.c_start
        video_chunk_size = self.df.loc[(index, now_qp, now_s, now_r), 'Size']
        self.bw_use.append(self.df.loc[(index, now_qp, now_s, now_r), 'Bitrate']/1000)

        encode_t = ENCODE_T[now_qp * 16 + now_s * 4 + now_r]
        end = self.start + encode_t
        self.lag_1.append(encode_t)
        self.lag_2.append(self.start - self.video_start_shoot - Length)
        bw = 0
        latency = 0
        v_s = video_chunk_size
        while True:
            if math.ceil(end) == end:
                if end + 1 >= len(self.cooked_bw):
                    real_bw = self.cooked_bw[int(end + 1) % len(self.cooked_bw)]
                else:
                    real_bw = self.cooked_bw[int(end + 1)]
                duration = 1
            else:
                if math.ceil(end) >= len(self.cooked_bw):
                    real_bw = self.cooked_bw[math.ceil(end) % len(self.cooked_bw)]
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
                latency = end - self.video_start_shoot - Length + RTT
                self.lag_3.append(end-encode_t-self.start+RTT)
                bw = v_s / (self.lag_3[-1])
                bw = bw/125
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
        if self.server_bu: self.last_end = self.server_bu[0][0]
        else: self.last_end=end

        latency = latency + wait_t(self.server_bu) + FRAME[now_s]*INFER_T
        self.lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(FRAME[now_s]*INFER_T)

        self.server_bu.append([end, FRAME[now_s]])
        R = self.bw[-1]
        f = self.df.loc[(index, now_qp, now_s, now_r), 'Accuracy']
        self.Reward.append(C_R(f, latency))
        self.F1.append(f)
        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        buffer_size = (self.start - self.start_shoot_fix) // 2 - self.video_chunk_counter + 1
        wait = int(min(buffer_size, Queue_LENGTH))
        for i in range(wait):
            self.QP.append(qp)
            self.S.append(s)
            self.R.append(r)
        QE = is_empty(buffer_size)
        QC = is_full(buffer_size)
        RC = detect_congestion(self.lag_3[-1])
        SProbeDone = probe_signal(R, bit+1)

        if self.video_chunk_counter > self.TOTAL:
            end_of_video = True

        return QE, QC, RC, R, SProbeDone, end_of_video


def bandwidth_predictor(bw, est):
    if est > len(bw):
        return np.mean(bw)
    else:
        b = bw[-est:]
        return np.mean(b)

def is_empty(q):
    return q <= 0


def is_full(q):
    return q >= Queue_LENGTH


def detect_congestion(l):
    if l > RC_thresholds:
        return True
    else:
        return False

def probe_signal(bw_est, next_bit):
    if next_bit <= len(VIDEO_BIT_RATE) - 1:
        return bw_est > VIDEO_BIT_RATE[next_bit]
    else:
        return False

def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else: return 0