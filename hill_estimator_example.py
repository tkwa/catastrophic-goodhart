# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto

# Generate sample data from a heavy-tailed distribution (e.g., Pareto)
np.random.seed(0)
sample_data = pareto.rvs(b=2.62, size=1000)

# Sort data in descending order
sorted_data = np.sort(sample_data)[::-1]

# Number of upper order statistics to consider
k = np.arange(1, len(sorted_data) + 1)

# Calculate Hill estimator
hill_estimator = np.cumsum(np.log(sorted_data[:len(k)])) / k - np.log(sorted_data[:len(k)])

# Plot the Hill estimator
plt.plot(k, hill_estimator)
plt.xlabel('Number of upper order statistics')
plt.ylabel('Hill estimator')
plt.title('Hill Estimator for Heavy-tailed Distribution')
plt.grid(True)
plt.show()

# %%
