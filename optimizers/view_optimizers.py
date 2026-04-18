import numpy as np
import matplotlib.pyplot as plt
import os
from optimizers import HypergradientDescent, StandardGD, ProjectedGD, StochasticGD, NesterovMomentum, LBFGSOptimizer
from scipy.linalg import toeplitz,circulant

# ==========================================
# 1. Setup Directory and Constants
# ==========================================
output_folder = "test/"
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Define the range of alpha values to test
alpha_values = [1.0, 5.0, 10.0, 25.0, 50.0]
plt.rcParams['agg.path.chunksize'] = 10000    # Add this line
# Setup Static Data
np.random.seed(42)
N, K = 2, 2
kernel_raw = np.abs(np.random.randn(K)) 
kernel = kernel_raw / np.sum(kernel_raw)

# 1. Prepare the first column
column = np.zeros(N)
column[:K] = kernel
row = np.zeros(N)
row[0] = column[0]
if N > 1:
        row[1:] = column[:0:-1] 
A_matrix = toeplitz(column, row)
c_val = 0.5
start_pt = np.array([0.9, 0.3])
max_iters = 200 # Reduced slightly for faster multi-graph generation

# ==========================================
# 2. Main Loop for Alpha Values
# ==========================================
sigmoid = lambda z, alpha, c: 1 / (1 + np.exp(-alpha * (z - c)))
inv_sigmoid = lambda h, alpha, c: c - (np.log(1/h - 1) / alpha)
y_target = np.random.randint(0, 2, size=N).astype(float)

for alpha_val in alpha_values:
        print(f"Generating plot for alpha = {alpha_val}...")
        # x_true = inv_sigmoid(y_target,alpha_val, c_val)
        # Generate target y based on current alpha


        def get_error(x_final):
                y_pred = sigmoid(x_final @ A_matrix, alpha_val, c_val)
                
                # Calculate the relative norm of the error (residual)
                # This will go to 0 as your optimizer converges
                error = (np.linalg.norm(y_target - y_pred) / np.linalg.norm(y_target)) * 100
                return error
        args = (y_target, A_matrix, alpha_val, c_val)

        opts = {
                "Projected GD": ProjectedGD(start_pt, *args, learning_rate=0.05),
                "Nesterov": NesterovMomentum(start_pt, *args, learning_rate=0.05, momentum=0.05),
                "Stochastic GD": StochasticGD(start_pt, *args, learning_rate=0.05, batch_size=1),
                "L-BFGS": LBFGSOptimizer(start_pt, *args),
                "HyperGD": HypergradientDescent(start_pt, *args, initial_lr=0.05, hyper_lr=1e-2)
        }

        paths = {}
        stats = {}

        for name, opt in opts.items():
                path = [start_pt.copy()]
                if name == "L-BFGS":
                        opt.run()
                        path.append(opt.x.copy()) # Ensures a line is drawn from start to finish
                else:
                        for _ in range(max_iters):
                                path.append(opt.step().copy())
                
                paths[name] = np.array(path)
                stats[name] = {"error": get_error(opt.x)}
        print(get_error(opt.x))
        print(sigmoid(opt.x @ A_matrix, alpha_val, c_val), y_target )
        # ==========================================
        # 3. Plotting
        # ==========================================
        fig, ax = plt.subplots(figsize=(10, 8))

        # Landscape Calculation
        grid_res = 100
        _range = np.linspace(-0.2, 1.2, grid_res)
        X, Y = np.meshgrid(_range, _range)
        Z = np.zeros_like(X)
        for i in range(grid_res):
                for j in range(grid_res):
                        z_grid = A_matrix @ np.array([X[i,j], Y[i,j]])
                        h_z = 1 / (1 + np.exp(-alpha_val * (z_grid - c_val)))
                        Z[i,j] = np.sum((y_target - h_z)**2)

        contours = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
        cbar = fig.colorbar(contours, ax=ax)
        cbar.set_label('Loss J(x) (SSE)', rotation=270, labelpad=15)

        colors = {
        'Standard GD': 'r',
        'Projected GD': 'orange',
        'Nesterov': 'c',
        'Stochastic GD': 'm',
        'L-BFGS': 'k',
        'HyperGD': 'b'    
        }

        for name, path in paths.items():
            label_str = f"{name} | Error: {stats[name]['error']:.2f}%"
            
            # Use ONLY ONE plot call with '-x' to show the line and markers together
            # This uses the color from your dictionary and ensures every point has an 'x'
            ax.plot(path[:,0], path[:,1], '-x', 
                    color=colors[name], 
                    markersize=4, 
                    label=label_str, 
                    linewidth=1.5, 
                    alpha=0.9,
                    zorder=10)

            # Add an arrowhead to show final direction
            if len(path) > 1:
                ax.annotate('', xy=(path[-1,0], path[-1,1]), xytext=(path[-2,0], path[-2,1]),
                            arrowprops=dict(arrowstyle='->', color=colors[name], lw=1.5))

        # Start point and bounds (kept outside the path loop logic but inside alpha loop)
        ax.scatter(start_pt[0], start_pt[1], color='white', edgecolor='black', 
                    s=80, marker='o', label="Start x", zorder=11)
        
        rect = plt.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', 
                                facecolor='none', linestyle='--', label='Bounds [0,1]')
        ax.add_patch(rect)

        ax.set_title(f'Optimizer Comparison (Alpha={alpha_val})')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize='small')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)

        # Save and close
        file_name = f"Optimizer_alpha_{int(alpha_val)}.png"
        plt.savefig(os.path.join(output_folder, file_name), dpi=150, bbox_inches='tight')
plt.close(fig)
