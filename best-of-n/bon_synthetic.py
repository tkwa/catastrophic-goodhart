# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import levy

# Sample n_samples points from t-distributions with known degrees of freedom

def generate_one_data(n_samples, spec):
    n_samples = int(n_samples)
    match spec:
        case ('t', df):
            factor = ((df - 2)/df)**0.5
            return factor * np.random.standard_t(df, size=n_samples)
        case 'lognormal': 
            return np.random.lognormal(0, np.log(2), size=n_samples) - 1
        case 'normal':
            return np.random.normal(0, 1, size=n_samples)
        case 'levy':
            return levy.rvs(size=n_samples)
        case _:
            raise ValueError("Unknown distribution")

def generate_data(n_samples, spec_left, spec_right):
    return np.stack(
        (2 * generate_one_data(n_samples, spec_left), generate_one_data(n_samples, spec_right)),
        axis=1
    )


# Compute the top values of x+y
def compute_argmax(data):
    return np.argmax(data[:, 0] + data[:, 1])

def make_subplot(ax, x_spec, y_spec, n_trials=100, max_log_n = 8):
    a = []
    for n in np.geomspace(1, 4**max_log_n, max_log_n+1):
        for _ in range(n_trials):
            samples = generate_data(n, x_spec, y_spec)
            argmax = np.argmax(samples[:, 0] + samples[:, 1])
            best_x, best_y = samples[argmax]
            a.append((round(n), best_x, best_y))

    df = pd.DataFrame(a, columns=['n', 'x', 'y'])
    # Add sum column
    df['sum'] = df['x'] + df['y']

    # Seaborn line plot with two lines for x and y, with scatter points as well
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x='n', y='x', ax=ax, label='Utility')
    sns.lineplot(data=df, x='n', y='y', ax=ax, label='Error')
    sns.lineplot(data=df, x='n', y='sum', ax=ax, label='Proxy reward')
    ax.title.set_text(f"{x_spec} vs {y_spec}")
    if 'levy' in (x_spec, y_spec):
        ax.set_yscale('log')
    ax.set_xscale('log')
# plt.yscale('log')

# %%
fig, axs = plt.subplots(2, 5)
fig.set_size_inches(20, 12)
for j, y_spec in enumerate(['normal', 'lognormal', ('t', 3), ('t', 5), 'levy']):
    for i, x_spec in enumerate(['normal', ('t', 10)]):
        make_subplot(axs[i][j], x_spec, y_spec)
        
plt.show()

# %%
fig = plt.figure()
make_subplot(fig.gca(), ('t', 10), ('t', 5), max_log_n=10)
plt.show()
# %%

# %%

# %%

# %%

# %%
