# %%
import math
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from einops import repeat, reduce, rearrange
from tqdm import tqdm

# from starlingclass import GPTRewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "meta-llama/Llama-2-7b-chat-hf"
print(f"Loading model")
hf_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
# %%
print(f"Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# %%


filename = "llama_sequences.txt"
# Code to sample token sequences from the model
with open(filename, "w") as f:
    for i in tqdm(range(2000)):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 5)).to(device)
        sequence = hf_model.generate(input_ids, max_new_tokens=128)
        f.write(str(sequence[0].tolist()))
        f.write("\n")
# %%

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.decode(sequence[0].tolist())
# %%

# get logprob
scores = hf_model.compute_transition_scores(sequence)
# %%
