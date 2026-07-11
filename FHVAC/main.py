import numpy as np
import torch
import pandas as pd
from test import RL_test, IL_test, Fusion_test
from utils import load_one_trace
import scipy.stats as stats
import os

RLMODEL = 'Results/RL/RL.model'
ILMODEL = 'Results/Fusion/IL.model'
Fusion = 'Results/Fusion/Fusion.model'

if __name__ == '__main__':
    output_dir = f'Results'
    os.makedirs(output_dir, exist_ok=True)

    F1, lag, bw_use, reward, bw_name, lag_1, lag_2, lag_3, lag_4, lag_5 = Fusion_test(RLMODEL, ILMODEL, Fusion)

    with open(f'test.txt', 'w') as f:
        for i in range(len(F1)):
            f.write(f"{F1[i]} {lag[i]} {bw_use[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")


