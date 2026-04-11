import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. Setup
# ==========================================
output_dir = Path("symbolic_diagrams")
output_dir.mkdir(exist_ok=True)

# Define a large figure with a white background for clean document import
fig = plt.figure(figsize=(18, 9))
ax = fig.add_subplot(111)
ax.axis('off') # Hide axes, only show the diagram
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

fig.suptitle(r'Depiction of Sliding Sum as Toeplitz Multiplication ($Ax = z$)', 
             fontsize=24, y=0.98, weight='bold')

# --- Helper function for manual brackets ---
def draw_bracketed_column(x_pos, y_pos, title, content_text, font_sz=20, linespacing=2.0):
    """Manually places vertical brackets around text content."""
    ax.text(x_pos, y_pos + 15, title, fontsize=24, weight='bold', ha='center')
    # Use a large serif font for the brackets for a classic look
    ax.text(x_pos - 1, y_pos, "[", fontsize=70, family='serif', ha='right')
    ax.text(x_pos, y_pos+3, content_text, fontsize=font_sz, 
            linespacing=linespacing, ha='left')
    # Offset the closing bracket based on width estimates
    offset_x = font_sz * 0.15 # Rough adjustment for character width
    ax.text(x_pos + offset_x, y_pos, "]", fontsize=70, family='serif', ha='left')

# ==========================================
# 2. Assemble the Multiplication Diagram
# ==========================================
y_base = 35 # Base vertical position for the main equation

# --- Vector x (The Input Pulse) ---
# Content is single letters a-g
x_content = "a\nb\nc\nd\ne\nf\ng\n..."
# Place x on the left
draw_bracketed_column(15, y_base, "x", x_content, linespacing=1.8)

# --- Multiplication Sign 'x' ---
ax.text(28, y_base + 12, r"$\times$", fontsize=50, ha='center', color='darkgrey')

# --- Matrix A (The Toeplitz Kernel) ---
# We use a 5x4 slice to fit clearly
a_matrix_content = (
    "w1  0   0   0   ...\n"
    "w2  w1  0   0   ...\n"
    "w3  w2  w1  0   ...\n"
    "0   w3  w2  w1  ...\n"
    "0   0   w3  w2  ...\n"
    "...  ...  ...  ...  ..."
)
# Place A in the center, using a monospace font for perfect column alignment
ax.text(45, y_base + 15, "A", fontsize=24, weight='bold', ha='center')
ax.text(37, y_base + 3, "[", fontsize=90, family='serif', ha='right')
ax.text(38, y_base+5, a_matrix_content, fontsize=20, family='monospace', linespacing=1.65)
ax.text(70, y_base + 3, "]", fontsize=90, family='serif', ha='left')

# --- Equals Sign '=' ---
ax.text(76, y_base + 12, "=", fontsize=50, ha='center', color='darkgrey')

# --- Vector z (The Convolved Result) ---
# We write the content symbolically for z[4] to match your request
z_content = (
    "...\n"
    "z_3\n"
    r"$w_3 c + w_2 d + w_1 e$" + "\n"
    "z_5\n"
    "..."
)
# Place z on the right
# Font size is slightly larger for the red result line
draw_bracketed_column(90, y_base, "z", z_content, font_sz=16, linespacing=2.2)

# ==========================================
# 3. Highlight the Calculation Rule
# ==========================================
# This box and arrow clearly show the user's requested "small portion sum"
rule_title = r"One output ($z_4$) is the sliding sum of a small portion ($[c, d, e]$):"
rule_equation = r'$z_4 = (w_3 \times c) + (w_2 \times d) + (w_1 \times e)$ '

# Box in the bottom left
ax.text(5, 12, rule_title, fontsize=18, color='darkgreen', weight='bold')
ax.text(5, 5, rule_equation, fontsize=28, color='firebrick', weight='bold')

# --- Depicting the Flow (Arrow) ---
# Draw an arrow from the [c,d,e] section in x to the highlighted line in z
ax.annotate("", xy=(87, y_base + 10), xytext=(15, y_base + 10),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", 
                            color='red', alpha=0.3, linewidth=2))

# ==========================================
# 4. Save to the concepts directory
# ==========================================
file_name = output_dir / "toeplitz_multiplication_final.png"
# Ensure the background is white for easy document import
plt.savefig(file_name, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Diagram complete! Final multiplication image saved to: {file_name.absolute()}")