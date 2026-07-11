import numpy as np
import math

"""
    TP is the complete version of the throughput predictor, while b1~b3 are variants.
"""

def b1(bw):
    return np.mean(bw)

def b2(bw, timestamps, current_time, lambda_decay=0.2):
    intervals = np.array([current_time - ts for ts in timestamps])
    bw_array = np.array(bw)
    weights = np.exp(-lambda_decay * intervals)
    weights /= np.sum(weights)
    prior_mean = np.dot(weights, bw_array)
    bw_est = prior_mean
    return bw_est

def b3(bw, transfer_times, timestamps, current_time, lambda_decay=0.2):
    intervals = np.array([current_time - ts for ts in timestamps])
    bw_array = np.array(bw)
    transfer_times_array = np.array(transfer_times)

    weights = transfer_times_array * np.exp(-lambda_decay * intervals)
    weights /= np.sum(weights)
    prior_mean = np.dot(weights, bw_array)
    bw_est = prior_mean
    return bw_est

def TP(bw, transfer_times, timestamps, current_time, lambda_decay=0.2, theta=0.5, k=3):
    intervals = np.array([current_time - ts for ts in timestamps])
    bw_array = np.array(bw)
    transfer_times_array = np.array(transfer_times)

    weights = transfer_times_array * np.exp(-lambda_decay * intervals)
    weights /= np.sum(weights)
    prior_mean = np.dot(weights, bw_array)
    bw_est = prior_mean

    # protection mechanism
    recent_throughputs = bw_array[-k:]
    observed_mean = np.mean(recent_throughputs)
    observed_variance = np.var(recent_throughputs)
    b_min = min(recent_throughputs)
    if math.sqrt(observed_variance)/observed_mean > theta:
        bw_est = min(b_min, bw_est)


    return bw_est
