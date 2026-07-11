import time
from collections import deque
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PPO import Actor
import env_fix
from utils import load_one_trace, C_R
import scipy.stats as stats
import time

S_INFO = 8
S_LEN = 8  # past 8
L = 2

SUMMARY_DIR = 'Results'
LOG_FILE_VALID = 'Results/test_results/log_valid'
TEST_LOG_FOLDER_VALID = 'Results/test_results/'

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor


def evaluation(model, net_env):
    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)
    # reward_sum = 0
    done = True
    last_knob = 40

    while True:
        with torch.no_grad():
            prob = model(state.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 16  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 16
        skip = remainder // 4  # 0 to 3
        re = remainder % 4  # 0 to 3

        bw, latency, buffer_size, size, dynamics, _, end_of_video = net_env.get_video_chunk(qp, skip, re)

        # dequeue history record
        state = np.roll(state, -1, axis=1)
        state[0, -1] = bw
        state[1, -1] = latency
        state[2, -1] = buffer_size
        state[3, -1] = qp
        state[4, -1] = skip
        state[5, -1] = re
        state[6, -1] = size
        state[7, -1] = dynamics
        state = torch.from_numpy(state)

        if end_of_video:
            f1_mean = np.mean(net_env.F1)
            # f1_std = np.std(net_env.F1)
            lag_mean = np.mean(net_env.lag)
            # lag_std = np.std(net_env.lag)
            Reward_mean = np.mean(net_env.Reward)
            return f1_mean, lag_mean, Reward_mean


def valid(shared_model, epoch, log_file):
    os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    os.system('mkdir ' + TEST_LOG_FOLDER_VALID)

    model = Actor().type(dtype)
    model.eval()
    model.load_state_dict(shared_model.state_dict())

    cooked_bw, cooked_name = load_one_trace('../dataset/', '4G')
    TOTAL = 86400
    chunk = 600
    env = env_fix.Environment(cooked_bw=cooked_bw, start=0, name=NAME[0], chunk=chunk, total=TOTAL)
    f1, lag, reward = evaluation(model, env)
    print(epoch, cooked_name, f1, lag)
    log_file.write(str(int(epoch)) + '\t' +
                   str(f1) + '\t' +
                   str(lag) + '\t' +
                   str(reward) + '\n')
    log_file.flush()
    add_str = 'CASVA'
    model_save_path = SUMMARY_DIR + "/%s_%d.model" % (add_str, int(epoch))
    torch.save(shared_model.state_dict(), model_save_path)


def test(test_model, start=0, chunk=600, total=600, chunk_start=0):
    model = Actor().type(dtype)
    model.eval()
    model.load_state_dict(torch.load(test_model))

    cooked_bw, cooked_name = load_one_trace('../dataset/', '4G')

    env = env_fix.Environment(cooked_bw=cooked_bw, start=start, chunk=chunk,
                              total=total, chunk_start=chunk_start)

    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)

    while True:
        with torch.no_grad():
            prob = model(state.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 16  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 16
        skip = remainder // 4  # 0 to 3
        re = remainder % 4  # 0 to 3

        bw, latency, buffer_size, size, dynamics, f1, end_of_video = env.get_video_chunk(qp, skip, re)

        # dequeue history record
        state = np.roll(state, -1, axis=1)
        state[0, -1] = bw
        state[1, -1] = latency
        state[2, -1] = buffer_size
        state[3, -1] = qp
        state[4, -1] = skip
        state[5, -1] = re
        state[6, -1] = size
        state[7, -1] = dynamics
        state = torch.from_numpy(state)

        if end_of_video:
            return (env.F1, env.lag, env.bw_use, env.Reward, cooked_name,

                    env.lag_1, env.lag_2, env.lag_3, env.lag_4, env.lag_5)
