import env
import pandas as pd
import numpy as np
import torch
import os
import sys
from utils import load_one_trace
from Video_Size_Predictor.network import V_Model

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

df = pd.read_csv(f'coding_time.csv')
mean_time_list = df['mean_time'].tolist()

noa = [0.9184, 0.8136, 0.7406, 0.6602, 0.8755, 0.7784, 0.711, 0.6369, 0.8399, 0.7519, 0.6949, 0.6181, 0.7615, 0.6888, 0.6431, 0.5768, 0.911, 0.8076, 0.7379, 0.6607, 0.8692,
       0.7734, 0.7107, 0.6351, 0.8379, 0.7528, 0.692, 0.6178, 0.7608, 0.689, 0.6423, 0.5737, 0.9031, 0.8023, 0.7378, 0.6595, 0.8672, 0.7713, 0.7066, 0.6321, 0.8354, 0.7495,
       0.6929, 0.6155, 0.757, 0.6866, 0.6423, 0.5731, 0.89, 0.7999, 0.7367, 0.658, 0.8565, 0.7678, 0.7049, 0.6272, 0.8261, 0.7469, 0.689, 0.6083, 0.7516, 0.6839, 0.6417, 0.5722,
       0.8765, 0.7925, 0.7345, 0.654, 0.8454, 0.7594, 0.6965, 0.6197, 0.8187, 0.7417, 0.6874, 0.6025, 0.7455, 0.683, 0.6387, 0.5674, 0.8509, 0.7767, 0.7234, 0.6463, 0.8218,
       0.7508, 0.6874, 0.6038, 0.7915, 0.7311, 0.6786, 0.5852, 0.7376, 0.6785, 0.6363, 0.5585, 0.8181, 0.759, 0.7155, 0.6355, 0.7984, 0.7309, 0.6651, 0.5868, 0.774, 0.7179,
       0.6659, 0.5675, 0.7216, 0.6722, 0.6328, 0.5461, 0.7522, 0.7161, 0.6867, 0.6101, 0.7383, 0.6843, 0.6272, 0.549, 0.7218, 0.6849, 0.6304, 0.5305, 0.688, 0.6472, 0.6079, 0.5191,
       0.6967, 0.6755, 0.6571, 0.5875, 0.6757, 0.6343, 0.5913, 0.5176, 0.6676, 0.6409, 0.5963, 0.4987, 0.6525, 0.6268, 0.584, 0.4968, 0.5822, 0.5782, 0.5791, 0.531, 0.5518,
       0.533, 0.507, 0.4484, 0.5549, 0.5336, 0.5071, 0.428, 0.5571, 0.5512, 0.5176, 0.4346, 0.4661, 0.4475, 0.4596, 0.4521, 0.4381, 0.4123, 0.3841, 0.3545, 0.4475, 0.4162, 0.3924, 0.3416, 0.4506, 0.4367, 0.4157, 0.3534]

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
                llist, pre_acc, pre_beta = wo_AP(net_env.s, net_env.m, net_env.la, net_env.mo, net_env.fr[-1], net_env.con, r, theta_t)

                ind = optimization(llist, TL, tao)
                bit = ind[0]
                skip = ind[1]
                re = ind[2]
        else:
            return (net_env.F1, net_env.lag, net_env.lag_1, net_env.lag_2, net_env.lag_3, net_env.lag_4, net_env.lag_5)


def wo_AP(S_l, M_l, L_l, Mo_l, con_l, r, wait):
    combined = []
    for i in range(8):
        combined.append([S_l[i], M_l[i], L_l[i], Mo_l[i], con_l[i][0], con_l[i][1], con_l[i][2]])
    X3 = torch.tensor(combined, dtype=torch.float32)
    X3 = X3.to(device)
    with torch.no_grad():
        sample_X3 = X3.unsqueeze(0)
        outputs_2 = v_model(sample_X3)
    Y2_pred = outputs_2.squeeze().cpu().numpy()
    prediction = []
    for knob in range(len(Y2_pred)):
        a = to_k(knob)
        code_t = mean_time_list[knob]
        prediction.append(
            [a, # config
             noa[knob], # accuracy
             code_t + (2 * Y2_pred[knob] * REAL[a[0]] / r) + wait, # latency
             Y2_pred[knob] * REAL[a[0]] / 1000]) # bw usage
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
    output_dir = f'../Results/AD/test_wo_AP.txt'
    os.makedirs(output_dir, exist_ok=True)

    F1, lag, lag_1, lag_2, lag_3, lag_4, lag_5 = test(2, 0.1, 0, 0, lambda_decay=0.2, theta = 0.5)

    with open(f'{output_dir}', 'w') as f:
        for i in range(len(F1)):
            f.write(f"{F1[i]} {lag[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")



