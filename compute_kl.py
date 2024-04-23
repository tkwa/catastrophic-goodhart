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

# load fifteen_reward string
with open("fifteen_reward.txt", "r") as f:
    fifteen_reward = f.read(-1)

tokens = reward_tokenizer(fifteen_reward, return_tensors="pt")
# %%
reward_model.forward(input_ids=tokens["input_ids"].to(reward_device), attention_mask=tokens["attention_mask"].to(reward_device))
# %%
