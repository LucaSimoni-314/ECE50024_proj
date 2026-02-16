import torch
import pandas as pd
import itertools
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from multiprocessing import Pool, cpu_count
from code import solve_task # Ensure this accepts (A, x_true, alpha, c, lr, iters, ...)
from pathlib import Path
from tqdm import tqdm

def get_gaussian_toeplitz(n, sigma=50):
    """
    Generates convolution matrix A following the user's implementation image logic.
    """
    # Create kernel centered around N/2
    x_axis = np.arange(-n // 2, n // 2)
    kernel = np.exp(-(x_axis**2) / (2 * sigma**2))
    
    # Normalization (optional, but standard in implementation)
    kernel /= kernel.sum() 

    # Align peak for Toeplitz construction
    shifted_kernel = np.fft.fftshift(kernel)
    
    # Construct Toeplitz matrix using the first row as defined in the image
    # first_row = shifted_kernel[np.append(0, np.arange(n-1, 0, -1))]
    A_kernel = toeplitz(shifted_kernel)

    return torch.tensor(A_kernel, dtype=torch.float32)

def worker(p):
    return solve_task(**p)

def main(out_dir):
    out_dir = Path(out_dir) 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    ns = [10, 50, 100, 200]
    alphas = [1.0, 5.0, 10.0, 20.0]
    cs = [0.2, 0.5, 0.8, 0.95]
    lrs = [0.001, 0.005, 0.01]
    iters_list = [500, 1000]
    
    # Reduced seeds for high-core efficiency while maintaining statistical significance
    op_seeds = range(2)  
    sig_seeds = range(10)   

    print("Generating task grid with Gaussian Toeplitz A...")
    for n in ns:
        # A is fixed for a given N and kernel (sigma=50)
        A_shared = get_gaussian_toeplitz(n)
        
        for ss in sig_seeds:
            torch.manual_seed(ss + 1000)
            x_fixed = torch.rand(n)
            
            for alpha, c, lr, iters in itertools.product(alphas, cs, lrs, iters_list):
                tasks.append({
                    'A': A_shared, 'x_true': x_fixed, 'n': n,
                    'alpha': alpha, 'c': c, 'lr': lr, 'iters': iters,
                    'seed': ss, 'algo': 'gd'
                })

    results = []
    with Pool(cpu_count()) as p:
        # imap_unordered returns results as soon as they are ready
        for res in tqdm(p.imap_unordered(worker, tasks), total=len(tasks)):
            results.append(res)
    
    df = pd.DataFrame(results)
    df['mse'] = df['loss'] / df['n']
    df.to_csv(out_dir / "raw_data.csv", index=False)

    # --- ANALYSIS & PLOTTING ---
    default_n, default_alpha, default_c = 100, 10.0, 0.5
    default_lr, default_iters = 0.005, 500

    plot_configs = [
        {'var': 'alpha', 'fix': {'n': default_n, 'c': default_c, 'lr': default_lr, 'iters': default_iters}, 'color': 'blue'},
        {'var': 'c',     'fix': {'n': default_n, 'alpha': default_alpha, 'lr': default_lr, 'iters': default_iters}, 'color': 'green'},
        {'var': 'n',     'fix': {'alpha': default_alpha, 'c': default_c, 'lr': default_lr, 'iters': default_iters}, 'color': 'red'},
        {'var': 'lr',    'fix': {'n': default_n, 'alpha': default_alpha, 'c': default_c, 'iters': default_iters}, 'color': 'purple'},
        {'var': 'iters', 'fix': {'n': default_n, 'alpha': default_alpha, 'c': default_c, 'lr': default_lr}, 'color': 'orange'}
    ]   
    friendly_names = {
            'alpha': 'Stiffness (α)',
            'c': 'Threshold (c)',
            'n': 'Input Dimension (N)',
            'lr': 'Learning Rate (η)',
            'iters': 'Iterations'
        }
    defaults = {'n': 100, 'alpha': 10.0, 'c': 0.5, 'lr': 0.005, 'iters': 500}
    for cfg in plot_configs:
        v = cfg['var']
        # Filter data where all other variables are at their default value
        query_parts = [f"{k} == {val}" for k, val in defaults.items() if k != v]
        subset = df.query(" & ".join(query_parts))
        
        if subset.empty:
            print(f"Skipping {v}: No data found for specified defaults.")
            continue

        # Group by the independent variable and average over seeds
        avg = subset.groupby(v).mean(numeric_only=True).reset_index()
        
        # Create a clean string of what is fixed for the subtitle
        fixed_desc = ", ".join([f"{friendly_names[k]}={val}" for k, val in defaults.items() if k != v])

        # Initialize Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 1. MSE Plot (Primary Y-axis)
        color_mse = cfg['color']
        lns1 = ax1.plot(avg[v], avg['mse'], marker='o', markersize=8, color=color_mse, 
                        linewidth=2, label='Mean Squared Error (MSE)')
        ax1.set_xlabel(friendly_names.get(v, v), fontsize=12, fontweight='bold')
        ax1.set_ylabel('Reconstruction MSE', fontsize=12, color=color_mse, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_mse)
        
        # Data Labels for MSE
        for x, y in zip(avg[v], avg['mse']):
            ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9, color=color_mse, fontweight='bold')

        # 2. Time Plot (Secondary Y-axis)
        ax2 = ax1.twinx()
        color_time = 'dimgray'
        lns2 = ax2.plot(avg[v], avg['time'], marker='s', markersize=6, color=color_time, 
                        linestyle='--', alpha=0.7, label='Execution Time (s)')
        ax2.set_ylabel('Time (seconds)', fontsize=12, color=color_time, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_time)

        # Formatting
        if v == 'lr': ax1.set_xscale('log')
        plt.title(f"Optimization Performance vs {friendly_names.get(v, v)}\n"
                f"Fixed: {fixed_desc}", fontsize=13, pad=20)
        
        ax1.grid(True, which="both", ls="-", alpha=0.15)
        
        # Combined Legend
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

        plt.tight_layout()
        plt.savefig(out_dir / f"analysis_{v}.png", dpi=300)
        plt.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='ece_results')
    args = parser.parse_args()
    main(out_dir=args.dir)