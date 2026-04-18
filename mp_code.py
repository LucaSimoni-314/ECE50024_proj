import os
import numpy as np
import time
from multiprocessing import Pool
from scipy.linalg import toeplitz
from worker import runner
from tqdm import tqdm  # Import tqdm
# --- Config ---
N, K = 10, 3
ALPHA_VALS = [1.0, 10.0, 25.0, 50.0]
C_VALS = [0.2, 0.5, 0.8]
OPTIMIZERS = ["Gradient Descent", "Projected GD", "Nesterov", "Stochastic GD", "L-BFGS", "HyperGD"]
LR_VALS = [0.05, 0.01, 0.005, 0.001]
TOTAL_TRIALS = 5000
BATCH_SIZE = 1000
MAX_ITERS = 4000
MASTER_SEED = 43

def generate_A(N, K):
    half = K // 2
    x = np.arange(K) - half
    kernel = np.exp(-x**2 / (2 * (K/4)**2))
    kernel /= kernel.sum()
    column = np.zeros(N)
    column[:K] = kernel[::-1]
    row = np.zeros(N)
    row[0] = column[0]
    return toeplitz(column, row)

def worker_unpack(args):
    return runner(*args)

def task_generator(A_matrix):
    rng = np.random.default_rng(MASTER_SEED)
    all_seeds = rng.integers(0, int(1e8), size=TOTAL_TRIALS)

    for alpha in ALPHA_VALS:
        for c in C_VALS:
            for opt in OPTIMIZERS:
                for lr in LR_VALS:
                    for i in range(0, TOTAL_TRIALS, BATCH_SIZE):
                        yield (opt, all_seeds[i:i+BATCH_SIZE], A_matrix, alpha, c, MAX_ITERS, lr)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    stats = {}
    print("Running Legacy-Compatible Simulation...")

    A_matrix = generate_A(N, K)
    total_tasks = len(ALPHA_VALS) * len(C_VALS) * len(OPTIMIZERS) * len(LR_VALS) * (TOTAL_TRIALS // BATCH_SIZE)
    wall_start = time.perf_counter()

    try:
        with Pool(processes=31) as pool:
            iterator = pool.imap_unordered(worker_unpack, task_generator(A_matrix))
            for batch in tqdm(iterator, total=total_tasks, desc="Simulating", unit="batch"):
                key = (batch['optimizer_name'], batch['alpha'], batch['c_val'], batch['lr'])

                if key not in stats:
                    stats[key] = {'n': 0, 'avg_err': 0.0, 'avg_time': 0.0}

                s = stats[key]
                m = batch['n']
                n = s['n']
                new_n = n + m

                s['avg_err'] = (s['avg_err'] * n + batch['sum_err']) / new_n
                s['avg_time'] = (s['avg_time'] * n + batch['sum_time']) / new_n
                s['n'] = new_n

    except KeyboardInterrupt:
        print("\nSimulation interrupted. Cleaning up...")

    wall_end = time.perf_counter()

    print("\n" + "="*100)
    print(f"{'Optimizer':<15} | {'Alpha':<6} | {'C':<5} | {'LR':<6} | {'Avg MSE':<15} | {'Avg Time (ms)':<15}")
    print("-" * 100)
    for key in sorted(stats.keys()):
        d = stats[key]
        print(f"{key[0]:<15} | {key[1]:<6.1f} | {key[2]:<5.1f} | {key[3]:<6.4f} | {d['avg_err']:<15.4f} | {d['avg_time']/1e6:<15.2f}")
    print("="*100)
    print(f"\nTotal wall time: {wall_end - wall_start:.2f}s")