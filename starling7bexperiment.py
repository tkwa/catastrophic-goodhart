# %%

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



## Define the reward function
class WrappedStarling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        reward_tokenizer = model.tokenizer

    def forward(self, samples):
        # TODO figure out how attention mask is implemented
        """samples: List[str]"""
        input_ids = []
        attention_masks = []
        encodings_dict = reward_tokenizer(
            samples,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        ).to(reward_device)
        input_ids = encodings_dict["input_ids"]
        attention_masks = encodings_dict["attention_mask"]
        mbs = reward_batch_size
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
            out.extend(rewards)
        return torch.hstack(out)

# %%
## Inference over test prompts with llama2 chat template

test_sample = ["<s>[INST] Hello? </s> [/INST] Hi, how can I help you?</s>"]
# reward_for_test_sample = get_reward(test_sample)
# print(reward_for_test_sample)

# %%
# Testing running the model on embeddings (not tokens)
from einops import repeat, reduce, rearrange


input_ids = torch.tensor(reward_model.tokenizer(test_sample)['input_ids']).to(reward_device)
# input_ids = repeat(input_ids, '1 c -> b c', b=2)
input_embeds = reward_model.model.model.embed_tokens(input_ids).detach().requires_grad_()
print(input_embeds.shape)
# TODO add attention mask into model forward call
raw_model_outputs = reward_model.model(inputs_embeds=input_embeds)
# print(raw_model_outputs)
model_outputs = reward_model(inputs_embeds=input_embeds)
print(f"reward={model_outputs}")
model_outputs.backward()
# after this, need to dot with weights
drde = input_embeds.grad[0].clone()
input_embeds.grad.zero_()
print(drde)
# %%
# Experiment code
import gcg
import importlib
importlib.reload(gcg)

def n_edits_fn(x):
    return int((100000/(x+1))**0.3) // 4 + 2
torch.manual_seed(20240522)
for i in range(40):
    optimized_input, reward = gcg.run_gcg(reward_model, embed=reward_model.model.model.embed_tokens,
                                input_ids=None,
                                k=3, n_edits_fn=n_edits_fn, n_steps=1000, n_ctx=133, temp=100,
                                batch_size=12, gcg_batch_size=12, use_wandb=True, out_file="gcg_output.txt", mode="llama")
    print(reward_model.tokenizer.decode(optimized_input[0]))
    # log optimized_input to file with current time
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    torch.save(optimized_input, f"data/optimized_input_{i}.pt")
    with open(f"data/rewards_{i}.txt", "w") as f:
        f.write(f"{reward}\n")


# %%

type(reward_model.model)
type(reward_model.model.model)
# %%
import transformers
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

LlamaForCausalLM
# %%
reward_model.tokenizer.decode