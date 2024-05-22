# %%

import numpy as np
from scipy.stats import t 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from einops import reduce, rearrange, repeat, einsum
from matplotlib.gridspec import GridSpec


mode = "STILLS" # "ANIMATION" or "STILLS"

# Generate data
FRAMES = 50
DF = 3
x = np.linspace(-3, 30, 200)
# Q is the PDF of a t-distribution with 3 degrees of freedom
Q_x = t.pdf(x, DF)
P_x = np.broadcast_to(Q_x, (FRAMES, len(x)))

threshold = np.geomspace(1.5, 100, FRAMES)
C = 1
left_factor = (1 - C/threshold**0.8) / t.cdf(threshold, DF)
right_factor = (C / threshold**0.8) / (1 - t.cdf(threshold, DF))

left_mask = np.greater.outer(threshold, x)
right_mask = np.less.outer(threshold, x)

P_x = P_x * left_factor[:, None] * left_mask + P_x * right_factor[:, None] * right_mask

# For now these are computed in the interval only
# mu_P = einsum(P_x, x, 'i j,j->i') / einsum(P_x, 'i j->i')
mu_P = np.zeros(FRAMES)
D_KL = np.zeros(FRAMES)
for i in range(FRAMES):
    mu_P[i] = t.expect(lambda x: x, args=(DF,), loc=0, scale=1, lb=-np.inf, ub=threshold[i]) * left_factor[i] + \
              t.expect(lambda x: x, args=(DF,), loc=0, scale=1, lb=threshold[i], ub=np.inf) * right_factor[i]
D_KL = t.cdf(threshold, DF) * left_factor * np.log(left_factor) + (1 - t.cdf(threshold, DF)) * right_factor * np.log(right_factor)
# D_KL = P_x * np.log(P_x / Q_x) / einsum(P_x, 'i j->i')[:, None]
# D_KL = reduce(D_KL, 'i j->i', 'sum') 

# %%

# Create a figure with two subplots
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

# Bar graph on the left panel
def update_bar(i):
    ax1.clear()
    ax1.bar(['E[X]', 'D_KL'], [mu_P[i], D_KL[i]], color=['blue', 'orange'])
    ax1.set_ylim(0, 5)
    if mode == "ANIMATION": ax1.set_title(f'Iteration: {i+1}')

# Line graph on the right panel
def update_line(i):
    ax2.clear()
    ax2.plot(x, P_x[i], label='P_t(x)', color='blue')
    ax2.plot(x, Q_x, label='Q(x)', color='orange')
    ax2.set_ylim(0, 0.4)
    ax2.set_xlabel('x')
    ax2.set_title(f'Threshold: t={threshold[i]:.2f}')
    ax2.legend()

    ax3.clear()
    ax3.plot(x, P_x[i], label='P_t(x)', color='blue')
    ax3.plot(x, Q_x, label='Q(x)', color='orange')
    ax3.set_ylim(10**-10, 10)
    ax3.set_yscale('log')
    ax3.set_xlabel('x')
    ax3.legend()

# Create the animation
def animate(i):
    update_bar(i)
    update_line(i)

print(f"Starting animation with {FRAMES} frames")
ani = FuncAnimation(fig, animate, frames=FRAMES, interval=100, repeat=False)

# Save the animation as a GIF
if mode == "ANIMATION": ani.save('animated_graph.gif', writer='pillow')
elif mode == "STILLS": 
    for i in range(FRAMES):
        animate(i)
        plt.savefig(f"goodhart_kl_{i+1:03d}.png")
    print("Done saving stills")
plt.show()
plt.close()
# %%
