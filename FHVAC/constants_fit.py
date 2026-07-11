import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# RE = [[1920, 1080], [1280, 720], [720, 480], [320, 240]]
# SKIP = [0, 1, 2, 5]
# QP = [23, 28, 33, 38, 43]

QP = [23, 28, 33, 38, 43]
FPS = [1.0, 0.5, 0.3333, 0.1667]
RE = [1.0, 0.4444, 0.1667, 0.0370]
fo = 30

def accuracy_model(X, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    r, f, q = X
    term1 = (c1 - c2 * np.exp(c3 * r))
    term2 = (c4 - c5 * np.exp(c6 * f))
    term3 = (c7 * q**2 + c8 * q + c9)
    return term1 * term2 * term3

def size_model(X , c10, c11, c12, c13):
    r, f, q = X
    term1 = fo * f
    term2 = c10 * r**2 + c11 * r + c12
    term3 = np.exp((-c13) * q)
    return term1 * term2 * term3

def accuracy_fit():
    r_data = []
    f_data = []
    q_data = []
    a_data = []
    df = pd.read_hdf(f'../dataset/AD_QP.h5', 'encoding_data')
    for i in range(600):
        for j in range(5):  # QP:j
            for m in range(4):  # skip:m
                for n in range(4):  # re:n
                    r_data.append(RE[n])
                    f_data.append(FPS[m])
                    q_data.append(QP[j])
                    a_data.append(df.loc[(i, j, m, n), 'Accuracy'])

    X_data = np.vstack((r_data, f_data, q_data))

    initial_guess = [1.0, 1.0, -0.01, 1.0, 1.0, -0.01, 1.0, 1.0, 1.0]

    # lower_bounds = [0, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]
    # upper_bounds = [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, np.inf, np.inf]

    params_opt, params_covariance = curve_fit(accuracy_model, X_data, a_data, p0=initial_guess, maxfev=100000)

    c1, c2, c3, c4, c5, c6, c7, c8, c9 = params_opt
    print(c1, c2, c3, c4, c5, c6, c7, c8, c9)

    # print(f"c1 = {c1}")
    # print(f"c2 = {c2}")
    # print(f"c3 = {c3}")
    # print(f"c4 = {c4}")
    # print(f"c5 = {c5}")
    # print(f"c6 = {c6}")
    # print(f"c7 = {c7}")
    # print(f"c8 = {c8}")
    # print(f"c9 = {c9}")

    a_pred = accuracy_model(X_data, *params_opt)
    residuals = a_data - a_pred
    mse = np.mean(residuals**2)
    print(f"error：{mse}")

def size_fit():
    r_data = []
    f_data = []
    q_data = []
    s_data = []
    df = pd.read_hdf(f'AD_QP.h5', 'encoding_data')
    for i in range(600):
        for j in range(5):  # QP:j
            for m in range(4):  # skip:m
                for n in range(4):  # re:n
                    r_data.append(RE[n])
                    f_data.append(FPS[m])
                    q_data.append(QP[j])
                    s_data.append((df.loc[(i, j, m, n), 'Size'])/1000000)

    X_data = np.vstack((r_data, f_data, q_data))
    initial_guess = [-0.5, 2, 0.02, 0.10]

    # lower_bounds = [0, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]
    # upper_bounds = [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, np.inf, np.inf]

    params_opt, params_covariance = curve_fit(size_model, X_data, s_data, p0=initial_guess, maxfev=100000)

    c10, c11, c12, c13 = params_opt
    print(c10, c11, c12, c13)

    # print(f"c10 = {c10}")
    # print(f"c11 = {c11}")
    # print(f"c12 = {c12}")
    # print(f"c13 = {c13}")

    s_pred = size_model(X_data, *params_opt)
    residuals = s_data - s_pred
    mse = np.mean(residuals**2)
    print(f"error：{mse}")

if __name__ == '__main__':
    # accuracy_fit()
    size_fit()