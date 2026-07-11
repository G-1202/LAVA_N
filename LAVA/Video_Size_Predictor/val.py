from utils import load_one_trace, I_Model
import pandas as pd
import numpy as np
import torch
import os
import env

VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
REAL = [12000, 10000, 8600, 6500, 5000, 3300, 2200, 1200, 800, 400, 200]
Length = 2

INPUT_STEPS = 8
OUTPUT_STEPS = 1
FEATURES = 5
OUTPUT = 176

i_model = I_Model(model_dim=256, output_dim=OUTPUT)
i_model.load_state_dict(torch.load('../Inference_Accuracy_Predictor/model/I_model.pth'))
i_model.eval()

df = pd.read_csv(f'../coding_time.csv')
mean_time_list = df['mean_time'].tolist()

def fv_pre(beta_model, acc_model, device, S_l, M_l, L_l, Mo_l, Sk_l, fr, con_l, r, wait):
    combined = []
    c_acc = []
    for i in range(8):
        combined.append([S_l[i], M_l[i], L_l[i], Mo_l[i], Sk_l[i]])
        c_acc.append([S_l[i], M_l[i], L_l[i], Mo_l[i], con_l[i][0], con_l[i][1], con_l[i][2]])
    X1 = torch.tensor(combined, dtype=torch.float32)
    X1 = X1.to(device)

    X2 = np.array(fr)
    X2 = torch.tensor(X2, dtype=torch.float32).permute(2, 0, 1)
    X2 = X2.to(device)

    Xc = torch.tensor(c_acc, dtype=torch.float32)
    Xc = Xc.to(device)

    with torch.no_grad():
        sample_X1 = X1.unsqueeze(0)
        sample_X2 = X2.unsqueeze(0)
        sample_Xc = Xc.unsqueeze(0)
        outputs_1 = acc_model(sample_Xc, sample_X2)
        outputs_2 = beta_model(sample_X1)
    Y1_pred = outputs_1.squeeze().cpu().numpy()
    Y2_pred = outputs_2.squeeze().cpu().numpy()
    prediction = []
    for knob in range(len(Y1_pred)):
        a = to_k(knob)
        code_t = mean_time_list[knob]
        prediction.append(
            [a, # config
             Y1_pred[knob], # accuracy
             code_t + (2 * Y2_pred[knob] * REAL[a[0]] / r) + wait, # latency
             Y2_pred[knob] * REAL[a[0]] / 1000]) # bw usage
    return prediction, Y1_pred, Y2_pred

def to_k(knob):
    bit = knob // 16  # 0 to 10, because 176 // 16 = 11
    remainder = knob % 16
    skip = remainder // 4  # 0 to 3
    re = remainder % 4  # 0 to 3
    return (bit, skip, re)

def optimization(llist, tl=2, alpha=0.1):
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

def calculate_MAE(df, pre_acc, n):
    true_acc = df.xs(n, level='CHUNK')['Ratio'].tolist()
    mae_acc = sum(abs(pred - true) for pred, true in zip(pre_acc, true_acc)) / len(true_acc)
    return mae_acc

def test(net_env, size_model, df, device):
    I_model = i_model.to(device)
    MAE_acc = []
    bit = 10
    skip = 0
    re = 0
    t_l = 2
    while True:
        r, theta_t, _, n_i, end_of_video = net_env.get_video_chunk(bit, skip, re)
        if end_of_video != True:
            if theta_t >= t_l:
                bit = 10
                skip = 2
                re = 3
            else:
                llist, pre_acc, _ = fv_pre(size_model, I_model, device, net_env.s, net_env.m, net_env.la,
                                           net_env.mo, net_env.sk, net_env.fr[-1], net_env.con, r, theta_t)

                mae_acc = calculate_MAE(df, pre_acc, n_i)
                MAE_acc.append(mae_acc)
                ind = optimization(llist)
                bit = ind[0]
                skip = ind[1]
                re = ind[2]
        else:
            return MAE_acc

def val(acc_model, cuda):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    acc_model.eval()
    acc_model = acc_model.to(device)
    cooked_bw, cooked_name = load_one_trace('../dataset/', '4G')

    df = pd.read_hdf(f'../dataset/AD.h5', 'encoding_data')
    net_env = env.Environment(cooked_bw=cooked_bw, cooked_name=cooked_name, start=0, chunk_start=0)
    MAE_acc= test(net_env, acc_model, df, device)
    mae = np.mean(MAE_acc[7:])
    return mae
