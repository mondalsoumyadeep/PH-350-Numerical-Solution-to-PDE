import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# ----------------------------
# Global style
# ----------------------------
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'pdf.fonttype': 42,
})

# ----------------------------
# Parameters
# ----------------------------
L = 64
N = 64
dx = L / N
dt = 0.01
steps = 5000
save_every = 50
x = np.arange(0, L, dx)

# Two diffusion coefficients to compare
D1 = 10.0   # Coeff = (stable < 0.5)
D2 = 55.0   # Coeff = (unstable> 0.5)

Coeff1 = D1 * dt / dx**2
Coeff2 = D2 * dt / dx**2

# print(f"Case 1: Coeff = {Coeff1:.3f}")
# print(f"Case 2: Coeff = {Coeff2:.3f}")

# ----------------------------
# Laplace operator
# ----------------------------
def laplace(u, D):
    return D * (dt / dx**2) * (np.roll(u, -1) + np.roll(u, 1) - 2 * u)

# ----------------------------
# Initial condition
# ----------------------------
u1 = np.sin(2 * np.pi * x / L)
u2 = u1.copy()

# ----------------------------
# Folder for frames
# ----------------------------
out_dir = "frames_cfl"
os.makedirs(out_dir, exist_ok=True)
frame_paths = []

# ----------------------------
# Simulation loop
# ----------------------------
for t in range(1,steps):
    u1 += laplace(u1, D1)
    u2 += laplace(u2, D2)

    if t % save_every == 0:
        fig, axs = plt.subplots(1, 2, figsize=(12,4), sharey=True)

        # Left: case 1
        axs[0].plot(x, u1, color="green", label=f"D·dt/dx² = {Coeff1:.2f}")
        axs[0].set_ylim(-2, 2)
        axs[0].set_title(f"Step {t}")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("u")
        axs[0].legend(loc="upper right")

        # Right: case 2
        axs[1].plot(x, u2, color="red", label=f"D·dt/dx² = {Coeff2:.2f}")
        axs[1].set_ylim(-2, 2)
        axs[1].set_title(f"Step {t}")
        axs[1].set_xlabel("x")
        axs[1].legend(loc="upper right")

        frame_path = os.path.join(out_dir, f"frame_{t:05d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

# ----------------------------
# Make video
# ----------------------------
video_path = "cfl_comparison.mp4"
with imageio.get_writer(video_path, fps=10) as writer:
    for f in frame_paths:
        writer.append_data(imageio.imread(f))


