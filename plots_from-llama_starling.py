# %%

import math
import os
import torch
import torch as t
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
# from gcg import run_gcg, replacement_gradient
from einops import repeat, reduce, rearrange
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

from starlingclass import GPTRewardModel

## Define the reward model function class
# %%
reward_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_batch_size = 4

print(f"Loading reward model")
model_path = "meta-llama/Llama-2-7b-chat-hf"
hf_model = AutoModelForCausalLM.from_pretrained(model_path)
# %%



reward_model = GPTRewardModel(hf_model, "meta-llama/Llama-2-7b-chat-hf").to(reward_device)
reward_tokenizer = reward_model.tokenizer
reward_tokenizer.truncation_side = "left"

directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
for fpath in os.listdir(directory):
    if fpath.endswith(".pt") or fpath.endswith("model.bin"):
        checkpoint = os.path.join(directory, fpath)
        break

print(f"Loading reward model checkpoint: {checkpoint}")
reward_model.load_state_dict(torch.load(checkpoint), strict=False)
reward_model.eval().requires_grad_(False)

# %%
# histogram

batch_size = 4
n_samples = 16000
values = []
t.random.manual_seed(20240522)
filename = "plots/llama_sequences.txt"

with open(filename, "r") as f:
    lines = f.readlines()
    for i in tqdm(range(len(lines))):
        input_ids = t.tensor([eval(lines[i])])
        with t.no_grad():
            outputs = reward_model(input_ids)
        reward = outputs.cpu().numpy()
        values.append(reward.item())

mean_reward = sum(values) / len(values)
std_reward = math.sqrt(sum((r - mean_reward)**2 for r in values) / len(values))
print(f"mean reward {mean_reward}, std reward {std_reward}")
# %%
# make 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

histax = axs[0, 0]
histax.hist(values, bins=100)
histax.set_title("Histogram")
histax.set_ylabel("Frequency")
histax.set_xlabel("Reward")

# save plot to a file
# histax.savefig(f"reward_histogram_starling.png")

# %%
# now make normal probability plot

nppax = axs[0, 1]
values = np.array(values)
sorted_values = np.sort(values)
n = len(values)
percentiles = np.array([norm.ppf(i/n) for i in range(n)])
# nppax.figure()
nppax.scatter(percentiles, sorted_values)
nppax.set_ylabel("reward")
nppax.set_xlabel("Normal quantile")
nppax.set_title("Normal probability plot")

# nppax.savefig(f"reward_normal_qq_starling.png")

# appears to be light-tailed. This shows that heavy-tailed distributions may not appear so
# %%
# Exponential probability plot of values above mean

eppax = axs[1, 0]
values_above_mean = [v for v in values if v > mean_reward]
sorted_values = np.sort(values_above_mean)
n = len(values_above_mean)
percentiles = np.array([-np.log((n-i)/n) for i in range(n)])
# eppax.figure()
eppax.scatter(percentiles, sorted_values)
eppax.set_ylabel("reward")
eppax.set_xlabel("Exponential quantile")
eppax.set_title("Exponential probability plot")
# %%
# Hill estimator
from scipy.stats import pareto

sorted_data = np.sort(values)[::-1]

# Number of upper order statistics to consider
k_values = np.arange(int(n_samples ** 0.1), int(n_samples ** 0.6))

hill_estimator = []
hill_se = []
for k in k_values:
    hill_est = np.mean(np.log(sorted_data[:k]) - np.log(sorted_data[k]))
    hill_estimator.append(hill_est)
    hill_se.append(hill_est / np.sqrt(k))

hill_estimator = np.array(hill_estimator)
hill_se = np.array(hill_se)

hillax = axs[1, 1]
hillax.clear()
# Plot the Hill estimator
hillax.plot(k_values, hill_estimator)
hillax.errorbar(k_values, hill_estimator, yerr=hill_se, fmt='o', label='Error Bars', alpha=0.5)
hillax.set_xlabel('Number of upper order statistics (k)')
hillax.set_ylabel('Hill estimator')
hillax.set_title('Hill Estimator for Heavy-tailed Distribution')
hillax.grid(True)
# %%
fig.suptitle(f"Plots of reward from {n_samples} Llama-7B-chat generated inputs to Starling 7B alpha reward model")
plt.subplots_adjust(top=0.4)
fig
# %%
plt.savefig('plots_from-llama_starling.png')
# %%
# Save data to a file
np.save("llama_starling_reward_values.npy", values)
# %%
