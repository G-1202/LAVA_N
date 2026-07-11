import env
import pandas as pd
import numpy as np
import torch
import os
import sys
from utils import load_one_trace
from Video_Size_Predictor.network import V_Model
from Inference_Accuracy_Predictor.network import I_Model

VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
REAL = [12000, 10000, 8600, 6500, 5000, 3300, 2200, 1200, 800, 400, 200]
Length = 2

INPUT_STEPS = 8
OUTPUT_STEPS = 1
FEATURES = 5
num_heads = 4
num_layers = 3
OUTPUT = 176

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

v_model = V_Model(model_dim=128, output_dim=OUTPUT)
v_model.load_state_dict(torch.load('Video_Size_Predictor/model/V_model.pth'))
v_model.eval()
v_model = v_model.to(device)

i_model = I_Model(model_dim=256, output_dim=OUTPUT)
i_model.load_state_dict(torch.load('Inference_Accuracy_Predictor/model/I_model.pth'))
i_model.eval()
i_model = i_model.to(device)

df = pd.read_csv(f'coding_time.csv')
mean_time_list = df['mean_time'].tolist()

def test(TL, tao, start, chunk_start, lambda_decay=0.2, theta=0.5):
    cooked_bw, cooked_name = load_one_trace('../dataset/', '4G')
    net_env = env.Environment(cooked_bw=cooked_bw, cooked_name=cooked_name, start=start, chunk_start=chunk_start,
                                  lambda_decay=lambda_decay, theta=theta)
    bit = 10
    skip = 0
    re = 0
    while True:
        r, theta_t, _, n_i, end_of_video = net_env.get_video_chunk(bit, skip, re)

        # Execute prediction and decision encoding parameters
        if end_of_video != True:
            # theta_t is the waiting time
            if theta_t >= TL:
                bit = 10
                skip = 2
                re = 3
            else:
                llist = fv_pre(net_env.s, net_env.m, net_env.la, net_env.mo, net_env.fr[-1], net_env.con, r, theta_t)

                ind = optimization(llist, TL, tao)
                bit = ind[0]
                skip = ind[1]
                re = ind[2]
        else:
            return (net_env.F1, net_env.lag, net_env.lag_1, net_env.lag_2, net_env.lag_3, net_env.lag_4, net_env.lag_5)

"""
    Output the estimated latency and inference accuracy of the next video segment under all configurations.
"""
def fv_pre(S_l, M_l, L_l, Mo_l, fr, con_l, r, wait):
    prediction = []
    """
        Integrates multiple input streams and uses trained models to predict
        accuracy, latency, and bandwidth usage across different configurations.

        Returns:
            list: Predictions for each configuration knob.
    """
    # Core algorithm implementation details are withheld
    # Complete code will be released upon paper acceptance
    return prediction


def to_k(knob):
    bit = knob // 16  # 0 to 10, because 176 // 16 = 11
    remainder = knob % 16
    skip = remainder // 4  # 0 to 3
    re = remainder % 4  # 0 to 3
    return (bit, skip, re)


"""
    Online Optimization.
"""
def optimization(llist, tl, alpha):
    best_score = -float('inf')
    best_config = []
    for item in llist:
        acc, latency = item[1], item[2]
        if latency > tl:
            continue
        current_score = acc - alpha*latency
        if current_score > best_score:
            best_score = current_score
            best_config = item[0]
    if best_config == []:
        best_config = (10, 2, 3)
    return best_config


if __name__ == '__main__':
    output_dir = f'../Results/AD/test.txt'
    os.makedirs(output_dir, exist_ok=True)

    F1, lag, lag_1, lag_2, lag_3, lag_4, lag_5 = test(2, 0.1, 0, 0, lambda_decay=0.2, theta = 0.5)

    with open(f'{output_dir}', 'w') as f:
        for i in range(len(F1)):
            f.write(f"{F1[i]} {lag[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")


