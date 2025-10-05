import numpy as np
import matplotlib.pyplot as plt
import imageio, os

plt.rcParams.update({'font.family':'Helvetica','font.size':14})

L = 256
N = 256
dx = L / N
dt = 1
steps = int(1e4)
save_every = 50

#Diffusivity of two species
Du = 0.16
Dv = 0.08
#paramters carying which we can generates pattern

#gray scots
# F = 0.06 
# k = 0.062
#Waves
# F = 0.022
# k = 0.051
#Stripes
# F = 0.035
# k = 0.061
#Chaos
# F = 0.014
# k = 0.054
#pattern
F = 0.018
k = 0.047

U = np.ones((N,N))
V = np.zeros((N,N))
#Initial
U[N//2-10:N//2+10, N//2-10:N//2+10] = 0.50
V[N//2-10:N//2+10, N//2-10:N//2+10] = 0.25
#Vectorized using np.roll
def laplace(Z):
    return (np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1)-4*Z)/dx**2

out_dir = "frames_gs"
os.makedirs(out_dir, exist_ok=True)
frames = []
#Main loop
for t in range(1,steps+1):
    U += dt * (Du * laplace(U) - U * V * V + F * (1 - U))
    V += dt * (Dv * laplace(V) + U * V * V - (F + k) * V)
    if t % save_every == 0:
        fig, ax = plt.subplots()
        ax.imshow(V, cmap="inferno")
        # ax.set_title(f"Step {t}")
        ax.axis("off")
        f = os.path.join(out_dir, f"frame_{t:05d}.png")
        plt.savefig(f, dpi=600, bbox_inches="tight")
        plt.close()
        frames.append(f)

video_path = "pattern.mp4"
with imageio.get_writer(video_path, fps=20) as w:
    for f in frames:
        w.append_data(imageio.imread(f))


