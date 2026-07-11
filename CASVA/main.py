import numpy as np
from test import test
import os

MODEL = 'Results/CASVA.model'

if __name__ == '__main__':
    output_dir = f'Results'
    os.makedirs(output_dir, exist_ok=True)

    F1, lag, bw_use, reward, bw_name, lag_1, lag_2, lag_3, lag_4, lag_5 = test(MODEL)
    with open(f'test.txt', 'w') as f:
        for i in range(len(F1)):
            f.write(f"{F1[i]} {lag[i]} {bw_use[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")







