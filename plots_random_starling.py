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
values = []
t.random.manual_seed(20240522)
for i in tqdm(range(5000 // batch_size)):
    input_ids = t.randint(0, hf_model.config.vocab_size, (batch_size, 1024)).to(reward_device)
    with t.no_grad():
        outputs = reward_model(input_ids)
    reward = outputs.cpu().numpy()
    values.extend(reward)

mean_reward = sum(values) / len(values)
std_reward = math.sqrt(sum((r - mean_reward)**2 for r in values) / len(values))
print(f"mean reward {mean_reward}, std reward {std_reward}")
# %%
# make plot
plt.hist(values, bins=100)
plt.title("Histogram of reward from random inputs to Starling 7B reward model")
plt.ylabel("Frequency")
plt.xlabel("Reward")
plt.show()

# save plot to a file
plt.savefig(f"reward_histogram_starling.png")

# %%
# now make normal probability plot


values = np.array(values)
sorted_values = np.sort(values)
n = len(values)
percentiles = np.array([norm.ppf(i/n) for i in range(n)])
plt.figure()
plt.scatter(percentiles, sorted_values)
plt.ylabel("reward")
plt.xlabel("Normal quantile")
plt.title("Normal probability plot of reward from random inputs to Starling 7B reward model")
plt.show()

plt.savefig(f"reward_normal_qq_starling.png")

# appears to be light-tailed. This shows that heavy-tailed distributions may not appear so
# %%
# Exponential probability plot of values above mean

values_above_mean = [v for v in values if v > mean_reward]
sorted_values = np.sort(values_above_mean)
n = len(values_above_mean)
percentiles = np.array([-np.log((n-i)/n) for i in range(n)])
plt.figure()
plt.scatter(percentiles, sorted_values)
plt.ylabel("reward")
plt.xlabel("Exponential quantile")
plt.title("Exponential probability plot of reward from random inputs to Starling 7B reward model")
plt.show()
# %%
