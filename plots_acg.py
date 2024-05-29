# %%
"""
Computing KL divergence induced by the policy...
"""

import math
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from gcg import run_gcg, replacement_gradient
from einops import repeat, reduce, rearrange
import matplotlib.pyplot as plt

from starlingclass import GPTRewardModel

## Define the reward model function class
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_batch_size = 4

print(f"Loading model")
model_path = "meta-llama/Llama-2-7b-chat-hf"
hf_model = AutoModelForCausalLM.from_pretrained(model_path)
# %%
hf_model.to(device)

probs = []
rewards = []
for i in range(39):
    try:
        with open(f'data/optimized_input_{i}.pt', 'rb') as f:
            input_ids = torch.load(f).to(device)
    except FileNotFoundError:
        break
    # Get model outputs
    with torch.no_grad():
        outputs = hf_model(input_ids, labels=input_ids)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)

    # Compute log-probabilities for the sequence
    sequence_log_probs = torch.gather(log_probs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    total_log_prob = sequence_log_probs.sum()

    # print(f"Log-probabilities for the sequence: {sequence_log_probs}")
    print(f"Total log-probability: {total_log_prob.item()}")
    probs.append(total_log_prob.item())
    with open(f'data/rewards_{i}.txt', 'r') as f:
        reward = float(f.read())
    rewards.append(reward)
probs

# %%

# Plot rewards vs probs

plt.scatter(rewards, probs)
plt.title("Reward vs Log-probability of ACG-optimized sequences on Starling 7B-alpha")
plt.xlabel("Reward")
plt.ylabel("Log-probability")

plt.show()
# %%
