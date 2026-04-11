import torch
import numpy as np
import time
import argparse
from pathlib import Path
from optimizers.optimizers import StandardGD, ProjectedGD, StochasticGD, NesterovMomentum, LBFGSOptimizer

def sigmoid_activation(z, alpha, c):
    """Standard sigmoidal activation function."""
    return 1 / (1 + np.exp(-alpha * (z - c)))

def execute_optimization_run(A, x_true, alpha, c, n, seed, algo, lr=0.1, iters=200, verbose=False):
    """
    Bridges Torch data generation with NumPy-based custom optimizer classes.
    """
    # 1. Data Conversion
    A_np = A.detach().cpu().numpy()
    x_true_np = x_true.detach().cpu().numpy()
    
    # 2. Target Generation (Matches your binary-like CNN objective)
    with torch.no_grad():
        z_true = A @ x_true
        y_target_torch = 1 / (1 + torch.exp(-alpha * (z_true - c)))
        y_target_np = y_target_torch.detach().cpu().numpy()

    # 3. Initialization
    # Center-heavy start for 2D, random for higher dimensions
    start_pt = np.array([0.1, 0.9]) if n == 2 else np.random.rand(n)
    
    # 4. Optimizer Selection Map
    opt_args = (y_target_np, A_np, alpha, c)
    
    algorithms = {
        'gd': StandardGD(start_pt, *opt_args, learning_rate=lr),
        'pgd': ProjectedGD(start_pt, *opt_args, learning_rate=lr),
        'nesterov': NesterovMomentum(start_pt, *opt_args, learning_rate=lr, momentum=0.9),
        'sgd': StochasticGD(start_pt, *opt_args, learning_rate=lr, batch_size=min(32, n)),
        'lbfgs': LBFGSOptimizer(start_pt, *opt_args)
    }

    optimizer = algorithms.get(algo.lower(), algorithms['gd'])
    
    # 5. Execution Loop
    start_time = time.time()
    
    if algo.lower() == 'lbfgs':
        optimizer.run() # Quasi-Newton batch run
    else:
        for i in range(iters):
            optimizer.step()
            if verbose and i % 50 == 0:
                # Calculate Relative L2 Error
                rel_error = np.linalg.norm(optimizer.x - x_true_np) / np.linalg.norm(x_true_np)
                print(f"[{algo.upper()}] Step {i:03d} | Rel Error: {rel_error:.6f}")

    # 6. Final Metrics Extraction
    end_time = time.time()
    prediction = sigmoid_activation(A_np @ optimizer.x, alpha, c)
    final_sse_loss = np.sum((y_target_np - prediction)**2)
    
    return {
        "dimension": n, 
        "stiffness_alpha": alpha, 
        "offset_c": c, 
        "seed": seed, 
        "algo_used": algo,
        "final_loss": float(final_sse_loss), 
        "runtime_sec": end_time - start_time, 
        "learning_rate": lr, 
        "total_iters": iters,
        "solution_estimate": optimizer.x
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECE50024 Inverse CNN Solver")
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algo', type=str, default='pgd', help='gd, pgd, nesterov, sgd, lbfgs')
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()

    # Environment Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Dummy identity matrix for standalone CLI testing
    A_val = torch.eye(args.n) 
    x_val = torch.rand(args.n)

    print(f"--- Running Solver: {args.algo.upper()} | N={args.n} | Alpha={args.alpha} ---")
    
    results = execute_optimization_run(
        A_val, x_val, args.alpha, args.c, args.n, args.seed, args.algo, lr=args.lr, verbose=True
    )
    
    print("\n" + "="*30)
    print(f"RESULTS FOR {args.algo.upper()}:")
    print(f"Final SSE Loss: {results['final_loss']:.10f}")
    print(f"Time Elapsed:   {results['runtime_sec']:.4f}s")
    print("="*30)