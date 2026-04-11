import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from pathlib import Path

# ==========================================
# 1. Setup Data (N=200, K=7)
# ==========================================
np.random.seed(42)
torch.manual_seed(42)
N, K = 200, 7
# N, K = 2 , 2 

kernel_raw = np.abs(np.random.randn(K)) 
kernel = kernel_raw / np.sum(kernel_raw)
column = np.zeros(N); column[:K] = kernel
row = np.zeros(N); row[0] = kernel[0]
A_np = toeplitz(column, row)
A_torch = torch.tensor(A_np, dtype=torch.float32)

# Generate True X (Actual signal values)
x_true = torch.rand(N)
x_true_np = x_true.numpy()
c_val = 0.5
output_dir = Path("landscape_plots")
output_dir.mkdir(exist_ok=True)

# ==========================================
# 2. Loop Over Alpha
# ==========================================
for alpha in [1, 10, 50, 100]:
    print(f"Generating Actual-Coord Landscape (N=200) for alpha={alpha}...")
    
    with torch.no_grad():
        z_true = A_torch @ x_true
        y_target = 1 / (1 + torch.exp(-alpha * (z_true - c_val)))

    # Grid matches your 2D Optimizer range exactly
    grid_res = 100
    _range = np.linspace(-0.2, 1.2, grid_res)
    X, Y = np.meshgrid(_range, _range)
    Z = np.zeros_like(X)

    for i in range(grid_res):
        for j in range(grid_res):
            # START with the full 200D true vector
            x_curr = x_true.clone()
            
            # OVERWRITE only the first two indices with the actual grid coordinates
            x_curr[0] = X[i,j]
            x_curr[1] = Y[i,j]
            
            # Project/Clamp to [0,1]
            x_curr = torch.clamp(x_curr, 0, 1)
            
            z_grid = A_torch @ x_curr
            h_z = 1 / (1 + torch.exp(-alpha * (z_grid - c_val)))
            Z[i,j] = torch.sum((y_target - h_z)**2).item()

    # ==========================================
    # 3. Visualization
    # ==========================================
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(f'N=2 Landscape (Alpha={alpha})', fontsize=20)

    # 3D View
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x[0]'); ax1.set_ylabel('x[1]'); ax1.set_zlabel('Loss J(x)')
    ax1.view_init(elev=30, azim=225)

    # 2D View
    ax2 = fig.add_subplot(122)
    contours = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
    cbar = fig.colorbar(contours, ax=ax2)
    cbar.set_label('Loss J(x) (SSE)', rotation=270, labelpad=15)

    # Boundary Box [0, 1]
    rect = plt.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='black', 
                        facecolor='none', linestyle='--', label='Feasible Set [0,1]')
    ax2.add_patch(rect)

    ax2.set_xlabel('x[0]'); ax2.set_ylabel('x[1]')
    ax2.set_xlim(-0.2, 1.2); ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), shadow=True)

    plt.savefig(output_dir / f"landscape_N200_actual_alpha_{alpha}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

print("Process Complete. Files are in landscape_plots/.")