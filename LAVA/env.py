import numpy as np
import pandas as pd
import math
import os
import cv2
from Throughput_Predictor import TP, b3

TOTAL = 600

SKIP = [0, 1, 2, 5]
FPS = 30
Length = 2
RTT = 0.08
BW_estimate = 5
INFER_T = 0.0025
FRAME = [60, 30, 20, 10]

df = pd.read_csv(f'coding_time.csv')
ENCODE_T = df['mean_time'].tolist()

frame_path = f'../dataset/AD_frames'

class Environment:
    def __init__(
        self,
        cooked_bw,
        cooked_name,
        start,
        chunk_start,
        lambda_decay,
        theta,
    ):
        self.start_fix = start
        self.df = pd.read_hdf(f'../dataset/AD.h5', 'encoding_data')
        self.video_frame_start = 0
        self.video_frame = self.video_frame_start
        self.video_chunk_counter = 1
        self.cooked_bw = cooked_bw
        self.bw_name = cooked_name

        #  start point of the trace
        self.start = start
        self.c_start = chunk_start
        self.video_start_shoot = self.start - 2
        self.server_bu = []
        self.last_end = 0

        self.F1 = []
        self.lag = []
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance

        self.bw = []
        self.duration = []
        self.end_t = []

        self.bw_pre = []

        self.infer_end = []

        self.small = []
        self.mid = []
        self.large = []
        self.move = []
        self.skip = []
        self.frame = []
        self.config = []

        self.s = [0.0] * 8
        self.m = [0.0] * 8
        self.la = [0.0] * 8
        self.mo = [0.0] * 8
        self.sk = [0.0] * 8
        self.fr = []
        default_img = np.zeros((224, 224, 3), dtype=np.float32)
        self.fr.append(default_img)
        self.con = [[0.0, 0.0, 0.0]] * 8
        self.save_num = 0

        self.lambda_decay = lambda_decay
        self.theta = theta

    def get_video_chunk(self, bit, s, r):

        self.config.append([bit, s, r])
        index = self.video_chunk_counter - 1
        index += self.c_start
        video_chunk_size = self.df.loc[(index, bit, s, r), 'Size']

        encode_t = ENCODE_T[bit*16+s*4+r]
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
                l_t = end - encode_t - self.start + RTT
                self.lag_3.append(l_t)
                self.duration.append(l_t)
                bw = v_s / (self.lag_3[-1])
                bw = bw/125
                self.bw.append(bw)
                self.end_t.append(end+RTT)
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

        latency = latency + wait_t(self.server_bu) + FRAME[s] * INFER_T
        self.lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(FRAME[s] * INFER_T)

        self.server_bu.append([end, FRAME[s]])

        f = self.df.loc[(index, bit, s, r), 'Accuracy']

        small = self.df.loc[(index, bit, s, r), 'Small']
        mid = self.df.loc[(index, bit, s, r), 'Mid']
        large = self.df.loc[(index, bit, s, r), 'Large']
        move = self.df.loc[(index, bit, s, r), 'Move']
        self.F1.append(f)
        self.small.append(small)
        self.mid.append(mid)
        self.large.append(large)
        self.move.append(move)
        self.skip.append(s)
        self.frame.append((index+1, bit, s, r))

        infer_time = INFER_T * 60 / (SKIP[s] + 1)
        self.infer_end.append(end + infer_time)

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        # Update the ideal start time for transmitting the next segment
        if self.start < self.video_start_shoot + Length:
            if self.video_start_shoot + Length - self.start >= 1:
                self.start = self.video_start_shoot + Length
                if self.start < TOTAL*2:
                    b_p, t_d = probe(self.cooked_bw[int(self.start)])
                    self.bw.append(b_p)
                    self.duration.append(t_d)
                    self.end_t.append(self.start - 1 + t_d*2)
            else:
                self.start = self.video_start_shoot + Length

        # Predict bandwidth
        if BW_estimate > len(self.bw):
            bw_est = b_3(self.bw, self.duration, self.end_t, self.start, lambda_decay = self.lambda_decay)
        else:
            bw_est = TP(self.bw[-BW_estimate:], self.duration[-BW_estimate:], self.end_t[-BW_estimate:], self.start, lambda_decay = self.lambda_decay, theta = self.theta)

        while self.save_num < self.video_chunk_counter-1:
            if self.infer_end[self.save_num] <= self.start:
                add_element(self.s, self.small[self.save_num])
                add_element(self.m, self.mid[self.save_num])
                add_element(self.la, self.large[self.save_num])
                add_element(self.mo, self.move[self.save_num])
                add_element(self.sk, self.skip[self.save_num])
                add_frame(self.fr, self.frame[self.save_num])
                add_element(self.con, self.config[self.save_num])
                self.save_num += 1
            else:
                break

        theta_t = self.start - self.video_start_shoot - Length

        if self.video_chunk_counter > TOTAL:
            end_of_video = True
        if end_of_video != True: self.bw_pre.append(bw_est/1000)
        return bw_est, theta_t, self.lag_3[-1]+self.lag_2[-1], (self.video_chunk_counter - 1)+self.c_start, end_of_video

def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else: return 0

def add_element(l, small):
    l.append(small)
    if len(l) > 8:
        l.pop(0)

def add_frame(l, knob):
    index = knob[0]
    b = knob[1]
    s = knob[2]
    r = knob[3]
    file_name = f"{index}_{b * 16 + s * 4 + r}.jpg"
    image_path = os.path.join(frame_path, file_name)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"The image file cannot be read: {image_path}")
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_proc = img_rgb.astype(np.float32) / 255.0
    l.append(img_proc)
    if len(l) > 8:
        l.pop(0)

def probe(bw):
    if bw <= 0:
        return 0
    probe_packet_size = 1500
    probe_duration = probe_packet_size / (bw * 1000)
    probe_bw = probe_packet_size / (probe_duration * 125)

    return probe_bw, probe_duration
