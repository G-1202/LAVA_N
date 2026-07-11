import numpy as np
from DAO_utils import load_one_trace
from AE.AE_predict import predict
from bisect import bisect_left
import DAO_env
import os

VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
resolutions = ['480p', '720p', '1080p']
button = 33
RE = [[720, 480], [1280, 720], [1920, 1080]]

# Path of video frames
image = '../dataset/AD'

def test():
    cooked_bw, cooked_name = load_one_trace('../dataset/', '4G')
    net_env = DAO_env.Environment(cooked_bw=cooked_bw, cooked_name=cooked_name,
                                      start=0, chunk_start=0)
    bit = 0
    re = 0
    while True:
        r, n, end_of_video = net_env.get_video_chunk(bit, re)
        if end_of_video != True:
            index = bisect_left(VIDEO_BIT_RATE, r)
            select_bitrate = max(0, index - 1)

            pre_score = predict(f'{image}/{n}/00.jpg')
            pre_score = pre_score[0]
            button_list = np.zeros((len(VIDEO_BIT_RATE), len(RE)))
            convert(pre_score, button_list, len(VIDEO_BIT_RATE), len(RE))
            sub_array = button_list[:select_bitrate, :]
            if sub_array.size == 0:
                bit = 0
                re = 0
            else:
                max_pos = np.unravel_index(np.argmax(sub_array), sub_array.shape)
                bit = max_pos[0]
                re = max_pos[1]
        else:
            return (net_env.F1, net_env.transfer_lag, net_env.bw_use, net_env.Reward, cooked_name,
                    net_env.lag_1, net_env.lag_2, net_env.lag_3, net_env.lag_4, net_env.lag_5)


def convert(a, b, p, q):
    for i in range(q):
        start_idx = i * p
        end_idx = start_idx + p
        b[:, i] = a[start_idx:end_idx]

if __name__ == '__main__':
    output_dir = f'.Results'
    os.makedirs(output_dir, exist_ok=True)

    F1, lag, bw_use, reward, bw_name, lag_1, lag_2, lag_3, lag_4, lag_5 = test()

    with open(f'test.txt', 'w') as f:
        for i in range(len(F1)):
            f.write(f"{F1[i]} {lag[i]} {bw_use[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")