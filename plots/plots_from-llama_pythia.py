# %%
import torch as t
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint
import wandb
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np

import model_training.models.reward_model # noqa: F401 (registers reward model for AutoModel loading)
# from wrapper import HookedModuleWrapper

device = t.device('cuda:0')

# %%
reward_name = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
rank_model.to(device)

# %%
# histogram

def generate_token_sequence():
    

values = []
t.random.manual_seed(20240522)
for i in tqdm(range(5000)):
    # Generate token sequences of length 1024 by sampling
    input_ids = []
    for _ in range(1024):
        input_ids.append(t.randint(0, 50256, (1, 1024)).to(device))
    outputs = rank_model(input_ids)
    reward = outputs.logits[0, 0].item()
    values.append(reward)

mean_reward = sum(values) / len(values)
std_reward = math.sqrt(sum((r - mean_reward)**2 for r in values) / len(values))
print(f"mean reward {mean_reward}, std reward {std_reward}")
# %%
# make plot
plt.hist(values, bins=100)
plt.title("Histogram of reward from random inputs")
plt.ylabel("Frequency")
plt.xlabel("Reward")
plt.show()

# save plot to a file
plt.savefig(f"reward_histogram_{reward_name}.png")

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
plt.title("Normal probability plot of reward from random inputs")
plt.show()

plt.savefig(f"reward_normal_qq_{reward_name}.png")

# appears to be light-tailed. This shows that heavy-tailed distributions may not appear so
# %%
