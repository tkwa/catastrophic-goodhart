# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample n_samples points from t-distributions with known degrees of freedom
def generate_data(n_samples, df1, df2):
    """
    Both are equal variance
    """
    factor_1 = ((df1 - 2)/df1)**0.5
    factor_2 = ((df2 - 2)/df2)**0.5
    n_samples = int(n_samples)
    return np.stack(
        (2 * factor_1*np.random.standard_t(df1, size=n_samples), factor_2*np.random.standard_t(df2, size=n_samples)),
        axis=1)


# Compute the top values of x+y
def compute_argmax(data):
    return np.argmax(data[:, 0] + data[:, 1])

a = []
for n in np.geomspace(1, 4**8, 8+1):
    for _ in range(100):
        samples = generate_data(n, 10000, 3)
        argmax = np.argmax(samples[:, 0] + samples[:, 1])
        best_x, best_y = samples[argmax]
        a.append((round(n), best_x, best_y))

df = pd.DataFrame(a, columns=['n', 'x', 'y'])
# Add sum column
df['sum'] = df['x'] + df['y']

# Seaborn line plot with two lines for x and y, with scatter points as well
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='n', y='x', label='Utility')
sns.lineplot(data=df, x='n', y='y', label='Error')
sns.lineplot(data=df, x='n', y='sum', label='Proxy reward')

plt.xscale('log')
# plt.yscale('log')

# %%

# %%

# %%

# %%

# %%

# %%
