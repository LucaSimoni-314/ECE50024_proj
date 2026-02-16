import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from pathlib import Path

# ============================================================
# 1. Objective Definition
# ============================================================
class InverseCNNObjective:
    def __init__(self, A, y, alpha_val, c):
        self.A = A
        self.y = y
        self.alpha_val = alpha_val
        self.c = c

    def forward(self, x):
        z = self.A @ x
        out = torch.sigmoid(self.alpha_val * (z - self.c))
        return torch.sum((self.y - out) ** 2)

    def project(self, x):
        # Clamping ensures the 100D (or 512D) vector stays in the unit hypercube
        return torch.clamp(x, 0.0, 1.0)

# =======================================   =====================
# 2. Matrix Generation (Toeplitz Example)
# ============================================================
def get_gaussian_toeplitz(n, sigma=50):
    x_axis = np.arange(-n // 2, n // 2)
    kernel = np.exp(-(x_axis**2) / (2 * sigma**2))
    kernel /= kernel.sum() 
    shifted_kernel = np.fft.fftshift(kernel)
    first_row_indices = np.append(0, np.arange(n - 1, 0, -1))
    A_np = toeplitz(shifted_kernel[first_row_indices])
    return torch.tensor(A_np, dtype=torch.float32)

# ============================================================
# 3. 2D Surface Projection (Range 0 to 3)
# ============================================================
def evaluate_custom_range_slice(objective, theta_star, resolution=150):
    """
    Evaluates f(alpha, beta) = L(theta* + alpha*xi + beta*eta)
    where alpha and beta range from 0 to 3.
    """
    N = theta_star.numel()
    
    # Standard orthonormal direction vectors
    xi = torch.randn(N)
    xi /= torch.norm(xi)
    eta = torch.randn(N)
    eta -= torch.dot(eta, xi) * xi 
    eta /= torch.norm(eta)

    # UPDATED RANGE: 0 to 3
    grid_vals = np.linspace(-.5, .5, resolution)
    Alpha, Beta = np.meshgrid(grid_vals, grid_vals)
    Z = np.zeros_like(Alpha)

    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                # Apply the perturbation based on the 0-3 grid
                perturbation = Alpha[i, j] * xi + Beta[i, j] * eta
                x = objective.project(theta_star + perturbation)
                Z[i, j] = objective.forward(x).item()
                
    return Alpha, Beta, Z

# ============================================================
# 4. Visualization & Saving
# ============================================================
def save_custom_plots(Alpha, Beta, Z, alpha_param, output_dir):
    fig = plt.figure(figsize=(16, 7))
    
    # Use a standard f-string without LaTeX backslashes
    fig.suptitle(f"Topological Projection of J(x) into R^2 (Alpha Stiffness = {alpha_param})", 
                 fontsize=16, fontweight='bold')

    # 3D View
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(Alpha, Beta, Z, cmap='magma', antialiased=True)
    ax1.set_title(f"3D Loss Manifold")

    # 2D View
    ax2 = fig.add_subplot(122)
    # Using levels=50 provides enough detail to see the "staircase" steps
    cp = ax2.contourf(Alpha, Beta, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, ax=ax2, label='Cost J(x)')
    
    ax2.set_title("2D Loss Contour")
    ax2.set_xlabel("Direction Alpha Offset")
    ax2.set_ylabel("Direction Beta Offset")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = output_dir / f"range_0_3_alpha_{alpha_param}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()
# ============================================================
# 5. Main Execution
# ============================================================
def solve_for_x_hat(objective, initial_guess, lr=0.001, iterations=10000):
    # We want to optimize x, so we enable gradients
    x_hat = initial_guess.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_hat], lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        loss = objective.forward(x_hat)
        loss.backward()
        optimizer.step()
        
        # Keep x_hat in the [0, 1] box
        with torch.no_grad():
            x_hat.clamp_(0, 1)
            
    return x_hat.detach()


if __name__ == "__main__":
    torch.manual_seed(42)
    N, c, sigma = 512, 0.25, 50
    A = get_gaussian_toeplitz(N, sigma=sigma)
    
    random_x = torch.rand(N) # The center point
    
    output_dir = Path("range_plots")
    output_dir.mkdir(exist_ok=True)

    # Testing with a higher alpha to see the sharp plateaus in the large range
    test_alphas = [1,5,10, 50, 100]

    for alpha in test_alphas:
        print(f"Processing alpha={alpha} in range [0, 1.2]...")
        
        # 1. Generate the binary target y
        with torch.no_grad():
            y = torch.sigmoid(torch.tensor(alpha) * (A @ random_x - c)).round()
        
        # 2. Initialize the Objective FIRST
        obj = InverseCNNObjective(A, y, alpha, c)
        
        # 3. Solve for the optimized point (theta_star) using the objective
        # Pass 'obj' instead of 'y'
        theta_star = solve_for_x_hat(obj, random_x)
        
        # 4. Evaluate the slice centered at the OPTIMIZED point
        # Use theta_star instead of y
        Alpha, Beta, Z = evaluate_custom_range_slice(obj, theta_star)
        
        # 5. Plot
        save_custom_plots(Alpha, Beta, Z, alpha, output_dir)