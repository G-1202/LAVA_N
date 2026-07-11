import random
import numpy as np
import env_fix
from utils import load_one_trace
import scipy.stats as stats

L = 2
Lmax = 0.5
V = 1
l1 = 3.5
l2 = 0.5
I_max = 100
t_end = 0.01
fo = 30
INFER_T = 0.0025

QP = [23, 28, 33, 38, 43]
FPS = [1.0, 0.5, 0.3333, 0.1667]
RE = [1.0, 0.4444, 0.1667, 0.0370]

A = [0, 1, 2, 3, 4]
B = [0, 1, 2, 3]

# 替换成用其他数据拟合的参数
acc_param = [1.3007394495116849, 0.5614772019716007, -5.219189501776808, 1.000328185485448, 0.9995658994055635,
             -0.00011957049019982996, -0.3452723793825076, 14.714862245266417, 682.2000635894923]
size_param = [-0.5138689339324571, 2.110318941825457, 0.025522872260969734, 0.12150365880725592]

def initialize_population(size):
    population = []
    for _ in range(size):
        r = random.choice(B)
        f = random.choice(B)
        q = random.choice(A)
        population.append((int(r), int(f), int(q)))
    return population

def calculate_fitness(individual, Q, bw):
    r, f, q = individual
    target_value = target(Q, RE[r], FPS[f], QP[q], bw)
    # if target_value <= 0:
    #     return float('inf')
    # return 1.0 / target_value
    return -target_value

def tournament_selection(population, fitness, k=3):
    selected = random.sample(population, k)
    best_individual = max(selected, key=lambda x: fitness[x])
    return best_individual

def two_point_crossover(parent1, parent2):
    r1, f1, q1 = parent1
    r2, f2, q2 = parent2
    child1 = (r1, f2, q1)
    child2 = (r2, f1, q2)
    return child1, child2

def mutation(individual, mutation_rate=0.01):
    if random.random() < mutation_rate:
        return (random.choice(B), random.choice(B), random.choice(A))
    return individual

def accuracy_model(r, f, q, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    term1 = (c1 - c2 * np.exp(c3 * r))
    term2 = (c4 - c5 * np.exp(c6 * f))
    term3 = (c7 * q**2 + c8 * q + c9)
    return term1 * term2 * term3

def latency_model(r, f, q, bw, c10, c11, c12, c13):
    term1 = fo * f
    term2 = c10 * r**2 + c11 * r + c12
    term3 = np.exp((-c13) * q)
    s = term1 * term2 * term3 * 1000000
    return s/(bw*1000)

# 公式(10)
def target(Q, r, f, q, bw):
    a_t = accuracy_model(r, f, q, *acc_param)
    p_t = fo * f * L * INFER_T
    l_t = latency_model(r, f, q, bw, *size_param)
    Q = max(Q+l_t-Lmax, 0)
    return Q * l_t - V * (a_t  + l1 * p_t +l2 * l_t)

def rule_based(Q, bw):
    if Q == 0 and bw == 0:
        return (1, 1, 1)
    N = 20
    population = initialize_population(N)

    fitness = {ind: calculate_fitness(ind, Q, bw) for ind in population}
    F0 = max(fitness.values())
    i = 0
    while i <= I_max:
        best_individual = max(fitness, key=fitness.get)
        stud_pop = [best_individual] * (N // 2)
        rest_pop = [random.choice(population) for _ in range(N - len(stud_pop))]
        temp_pop = [tournament_selection(population, fitness) for _ in range(len(rest_pop))]
        new_population = stud_pop + temp_pop
        offspring = []
        for j in range(0, len(new_population), 2):
            if j + 1 < len(new_population):
                parent1, parent2 = new_population[j], new_population[j + 1]
                child1, child2 = two_point_crossover(parent1, parent2)
                offspring.append(mutation(child1))
                offspring.append(mutation(child2))
        population = offspring
        fitness = {ind: calculate_fitness(ind, Q, bw) for ind in population}
        F_new = max(fitness.values())
        if i > I_max or abs(F_new - F0) < t_end:
            break
        F0 = F_new
        i += 1
    selected_config = best_individual
    return selected_config

def test(index, start, chunk, total, chunk_start):
    cooked_bw, cooked_name = load_one_trace('train_trace/', index)
    env = env_fix.Environment(cooked_bw=cooked_bw, start=start, chunk=chunk, total=total, chunk_start = chunk_start)
    qp = 1
    skip = 1
    re = 1
    while True:
        bw_est, _, _, _, _, _, _, Q, end_of_video = env.get_video_chunk(qp, skip, re)
        best = rule_based(Q, bw_est)
        qp = best[2]
        skip = best[1]
        re = best[0]

        if end_of_video:
            f1_mean = np.mean(env.F1)
            # f1_std = np.std(env.F1, ddof=1)
            # f1_standard_error = f1_std / np.sqrt(len(env.F1))
            # f1_interval = stats.norm.interval(0.95, loc=f1_mean, scale=f1_standard_error)

            lag_mean = np.mean(env.lag)
            # lag_std = np.std(env.lag, ddof=1)
            # lag_standard_error = lag_std / np.sqrt(len(env.lag))
            # lag_interval = stats.norm.interval(0.95, loc=lag_mean, scale=lag_standard_error)

            print(f1_mean, lag_mean)


if __name__ == '__main__':
    test(0, 0, 1800, 1800, 0)
