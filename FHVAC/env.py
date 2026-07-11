import pandas as pd
import math
import numpy as np
from utils import load_one_trace

RANDOM_SEED = 42

QP = [23, 28, 33, 38, 43]
SKIP = [0, 1, 2, 5]
RE = [[1920, 1080], [1280, 720], [720, 480], [320, 240]]

FPS = 30
Length = 2
RTT = 0.08
BW_estimate = 5
Lmax = 0.5

df = pd.read_csv('coding_time.csv')
ENCODE_T = df['mean_time'].tolist()

INFER_T = 0.0025
FRAME = [60, 30, 20, 10]

class Environment:
    def __init__(
        self,
        cooked_trace,
        random_seed=RANDOM_SEED,
    ):
        np.random.seed(random_seed)
        self.cooked_bw = cooked_trace
        self.name = 0
        self.df = pd.read_hdf('../dataset/AD_QP.h5', 'encoding_data')

        self.TOTAL = 1800
        self.CHUNK = 600

        self.video_chunk_counter = 1
        self.start = np.random.randint(2, len(self.cooked_bw))
        self.video_start_shoot = self.start - 2
        self.start_shoot_fix = self.video_start_shoot

        self.Q = 0
        self.bw = []

        self.lag = []
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance

        self.server_bu = []
        self.last_end = 0
    def get_video_chunk(self, qp, s, r):
        index = (self.video_chunk_counter - 1) % self.CHUNK
        encode_t = ENCODE_T[qp * 16 + s * 4 + r]
        video_chunk_size = self.df.loc[(index, qp, s, r), 'Size']

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
                latency = end + RTT - self.video_start_shoot - Length
                self.lag_3.append(end - encode_t - self.start + RTT)
                bw = v_s / (self.lag_3[-1])
                bw = bw/1000
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

        latency += wait_t(self.server_bu) + FRAME[s] * INFER_T
        self.lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(FRAME[s] * INFER_T)
        self.server_bu.append([end, FRAME[s]])

        bw_est = bandwidth_predictor(self.bw, BW_estimate)
        self.Q = max(self.Q + self.lag_3[-1] - Lmax, 0)
        f = self.df.loc[(index, qp, s, r), 'Accuracy']

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        if self.video_chunk_counter <= self.TOTAL:
            index_n = (self.video_chunk_counter - 1) % self.CHUNK
            encode_d = ENCODE_T[qp * 16 + s * 4 + r]
            chunk_size_next = self.df.loc[(index_n, qp, s, r), 'Size']
            dynamics = (chunk_size_next - v_s)/v_s
            self.start += encode_d
        else:
            dynamics = 0

        buffer_size = ((self.start - self.start_shoot_fix)//2 - self.video_chunk_counter + 1) * Length

        if self.video_chunk_counter > self.TOTAL:
            end_of_video = True
            self.video_chunk_counter = 1
            self.Q = 0
            self.bw = []

        return bw_est, bw/125, self.lag_3[-1]+self.lag_2[-1], buffer_size, v_s/1000000, dynamics, f, self.Q, end_of_video


def bandwidth_predictor(bw, est):
    bw_subset = bw[-est:] if est <= len(bw) else bw
    r = len(bw_subset) / sum(1 / x for x in bw_subset)
    return r

def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else: return 0