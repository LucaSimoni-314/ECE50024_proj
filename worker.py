import numpy as np
import time
from optimizers.optimizers import StandardGD, HypergradientDescent, ProjectedGD, StochasticGD, NesterovMomentum, LBFGSOptimizer
from scipy.linalg import toeplitz

def runner(opt_name, seed_batch, A_matrix, alpha_val, c_val, max_iters, lr=0.005):
    local_err_sum = 0.0
    local_time_sum = 0
    batch_count = len(seed_batch)
    sigmoid = lambda z, a, c: 1 / (1 + np.exp(-a * (z - c)))
    N = A_matrix.shape[0]

    for seed in seed_batch:
        rng = np.random.default_rng(int(seed))

        y_target = rng.integers(0, 2, size=N)
        while y_target.sum() == 0 or y_target.sum() == N:
            y_target = rng.integers(0, 2, size=N)

        start_pt = rng.uniform(0.05, 0.95, size=N)

        args = (y_target, A_matrix, alpha_val, c_val)

        if opt_name == "L-BFGS":
            optimizer = LBFGSOptimizer(start_pt.copy(), *args)
        elif opt_name == "HyperGD":
            optimizer = HypergradientDescent(start_pt.copy(), *args, initial_lr=lr, hyper_lr=lr)
        else:
            opts_map = {
                "Projected GD": ProjectedGD,
                "Nesterov": NesterovMomentum,
                "Stochastic GD": StochasticGD,
                "Gradient Descent": StandardGD
            }
            optimizer = opts_map[opt_name](start_pt.copy(), *args, learning_rate=lr)

        t_start = time.perf_counter()
        if opt_name == "L-BFGS":
            optimizer.run()
        else:
            for _ in range(max_iters):
                optimizer.step()
        t_end = time.perf_counter()

        elapsed_ns = round((t_end - t_start) * 1e9)
        y_pred = sigmoid(A_matrix @ optimizer.x, alpha_val, c_val)
        # Calculate Relative Percent Error
        t_norm = np.linalg.norm(y_target)
        if t_norm == 0: # Safety check (though your while-loop prevents this)
            error = 0.0
        else:
            # (||y_target - y_pred|| / ||y_target||) * 100
            error = (np.linalg.norm(y_target - y_pred) / t_norm) * 100

        local_err_sum += error
        local_time_sum += elapsed_ns

    return {
        "optimizer_name": opt_name,
        "alpha": alpha_val,
        "c_val": c_val,
        "lr": lr,
        "sum_err": local_err_sum,
        "sum_time": local_time_sum,
        "n": batch_count
    }

def generate_A(N, K, seed=None):
    half = K // 2
    x = np.arange(K) - half
    kernel = np.exp(-x**2 / (2 * (K/4)**2))
    kernel /= kernel.sum()
    column = np.zeros(N)
    column[:K] = kernel[::-1]
    row = np.zeros(N)
    row[0] = column[0]
    return toeplitz(column, row)

if __name__ == "__main__":
    N, K = 10, 2
    MASTER_SEED = 50
    A_matrix = generate_A(N, K)

    result = runner(
        opt_name="Projected GD",
        seed_batch=[0, 1, 2, 3, 4],
        A_matrix=A_matrix,
        alpha_val=10.0,
        c_val=0.5,
        max_iters=400,
        lr=0.005
    )

    print(f"Optimizer : {result['optimizer_name']}")
    print(f"Alpha     : {result['alpha']}")
    print(f"C         : {result['c_val']}")
    print(f"LR        : {result['lr']}")
    print(f"Avg MSE   : {result['sum_err'] / result['n']:.6f}")
    print(f"Avg Time  : {result['sum_time'] / result['n'] / 1e6:.4f}ms")