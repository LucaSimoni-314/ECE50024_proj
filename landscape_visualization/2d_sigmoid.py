import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. Setup & Functions
# ==========================================
def sigmoid_h(z, alpha, c):
    """Calculates h(z) = 1 / (1 + exp(-alpha * (z - c)))"""
    return 1 / (1 + np.exp(-alpha * (z - c)))

# Output folder
output_dir = Path("sigmoid_raw")
output_dir.mkdir(exist_ok=True)

# The range of the input z (arbitrary signal value)
z_vals = np.linspace(-5.0, 5.0, 500)

# ==========================================
# Plot 1: Varying Alpha (Stiffness)
# ==========================================
plt.figure(figsize=(12, 7))

# We fix the offset C=0 to isolate Alpha's effect
fixed_c = 0.0
# Smooth to highly shattered/stiff
alpha_variants = [1.0, 5.0, 20.0, 100.0]

for alpha in alpha_variants:
    h_z = sigmoid_h(z_vals, alpha, fixed_c)
    # linewidth increases with stiffness to make it clear
    lw = 1 + alpha/50
    label_str = f"Alpha={alpha:.1f}"
    plt.plot(z_vals, h_z, label=label_str, linewidth=lw)

plt.axvline(fixed_c, color='black', linestyle='--', alpha=0.5, label='Offset c=0')
plt.title(f'Raw Sigmoid stiffness $\\alpha$ variance (Fixed $c=0$)')
plt.xlabel('Input Signal value (z)')
plt.ylabel('Activation h(z)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(shadow=True)

plt.savefig(output_dir / "sigmoid_alpha_variance.png", dpi=150)
plt.show()

# ==========================================
# Plot 2: Varying C (Offset/Bias)
# ==========================================
plt.figure(figsize=(12, 7))

# We fix the stiffness Alpha=5.0 to isolate C's effect
fixed_alpha = 10.0
# Negative, Zero, Positive offsets
c_variants = [-3.0, -1.0, 0.0, 1.0, 3.0]

for c in c_variants:
    h_z = sigmoid_h(z_vals, fixed_alpha, c)
    label_str = f"Offset c={c:.1f}"
    plt.plot(z_vals, h_z, label=label_str, linewidth=2)

plt.title(f'Raw Sigmoid offset $c$ variance (Fixed $\\alpha=10$)')
plt.xlabel('Input Signal value (z)')
plt.ylabel('Activation h(z)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(shadow=True)

plt.savefig(output_dir / "sigmoid_c_variance.png", dpi=150)
plt.show()

print(f"Done. Raw sigmoid plots are in: {output_dir.absolute()}/")