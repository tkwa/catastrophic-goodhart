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

n_samples = 16000
batch_size = 16
filename = "llama_sequences.txt"
# Code to sample token sequences from the model
with open(filename, "w") as f:
    for i in tqdm(range(n_samples // batch_size)):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (batch_size, 5)).to(device)
        sequence = hf_model.generate(input_ids, max_new_tokens=128)
        for j in range(batch_size):
            token_list = sequence[j].tolist()
            while token_list[-1] == 0:
                token_list.pop()
            # remove padding tokens
            # print(len(sequence[j]))
            f.write(str(token_list))
            f.write("\n")
# %%

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.decode(sequence[0].tolist())
# %%

# get logprob
scores = hf_model.compute_transition_scores(sequence)
# %%
