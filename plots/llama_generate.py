# %%
import math
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from einops import repeat, reduce, rearrange

# from starlingclass import GPTRewardModel


model_path = "meta-llama/Llama-2-7b-chat-hf"
hf_model = AutoModelForCausalLM.from_pretrained(model_path)
# %%

empty_input = torch.tensor([[1, 2]])

# Code to sample token sequences from the model
sequence = hf_model.generate(empty_input, max_new_tokens=100, output_scores=True)
# %%

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.decode(sequence[0].tolist())
# %%

# get logprob
scores = hf_model.compute_transition_scores(sequence)
# %%
