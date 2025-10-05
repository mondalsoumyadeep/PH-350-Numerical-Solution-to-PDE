import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha*x - beta*x*y
    dydt = delta*x*y - gamma*y
    return [dxdt, dydt]

alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
z0 = [40, 9]
t_span = (0, 75)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(lotka_volterra, t_span, z0, t_eval=t_eval, args=(alpha, beta, delta, gamma))

plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label="Prey", lw=2)
plt.plot(sol.t, sol.y[1], label="Predator", lw=2)
plt.xlabel("Time", fontsize=25, family="Helvetica")
plt.ylabel("Population", fontsize=25, family="Helvetica")
plt.xticks(fontsize=20, family="Helvetica")
plt.yticks(fontsize=20, family="Helvetica")
plt.legend(fontsize=20, ncol=2, bbox_to_anchor=(0.5, 1.15), loc="upper center")
plt.savefig("LV.png",dpi=300)

