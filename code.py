import torch
import time
import argparse

def h(z, alpha, c):
    return 1 / (1 + torch.exp(-alpha * (z - c)))

def solve_task(A, x_true, alpha, c, n, seed, algo, lr=0.001, iters=500, verbose=False, **kwargs):
    with torch.no_grad():
        y = (torch.matmul(A, x_true) > c).float()

    x_hat = torch.zeros(n, requires_grad=True)
    start_time = time.time()
    if algo == 'gd':
        for i in range(iters):
            loss = torch.sum((y - h(A @ x_hat, alpha, c))**2)
            loss.backward()
            with torch.no_grad():
                x_hat -= lr * x_hat.grad
                x_hat.clamp_(0, 1)
                x_hat.grad.zero_()
            if verbose and i % 10 == 0:
                print(f"Iteration {i}: Loss {loss.item():.6f}")
    
    res_loss = torch.sum((y - h(A @ x_hat, alpha, c))**2).item()
    return {
        "n": n, "alpha": alpha, "c": c, "seed": seed, "algo": algo,
        "loss": res_loss, "time": time.time() - start_time, "lr": lr, "iters": iters
    }

if __name__ == "__main__":
    # This block ONLY runs when you call 'python solver.py' directly
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=00)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algo', type=str, default='gd')
    args = parser.parse_args()

    print(f"--- DEBUG RUN: {args.algo.upper()} ---")
    result = solve_task(args.n, args.alpha, args.c, args.seed, args.algo, verbose=True)
    print(f"\nFinal Result: {result}")