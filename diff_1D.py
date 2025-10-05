import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


# Folder to save images
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)
frame_paths = []
# ----------------------------
# Global plotting style
# ----------------------------
plt.rcParams.update({
    'font.family': 'Helvetica',  
    'font.size': 20,
    'axes.labelsize': 25,
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
D = 10.0       
steps = 5000
save_every = 50  

x = np.arange(0, L, dx)

def laplace(u):
    du = np.zeros(N)
    for i in range(N):
        iNext = (i + 1) % N
        iPrev = (i - 1) % N
        du[i] = D * (dt / dx**2) * (u[iNext] + u[iPrev] - 2 * u[i])
    return du

# ----------------------------
# Initial condition
# ----------------------------
u = np.sin(2 * np.pi * x / L)


# ----------------------------
# Simulation loop
# ----------------------------
for t in range(1,steps):
    # print(np.sum(u))
    du = laplace(u)
    u = u + du


    if t % save_every == 0:
        
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(x, u, color="purple", lw=2)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"Iteration = {t}")
        frame_path = os.path.join(out_dir, f"frame_{t:05d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

# ----------------------------
# Make video from frames
# ----------------------------
video_path = "diffusion.mp4"
with imageio.get_writer(video_path, fps=10) as writer:
    for f in frame_paths:
        writer.append_data(imageio.imread(f))


