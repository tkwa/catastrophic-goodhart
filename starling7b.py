# %%

import math
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from gcg import run_gcg, replacement_gradient
from einops import repeat, reduce, rearrange

## Define the reward model function class
# %%
reward_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_batch_size = 4

print(f"Loading reward model")
model_path = "meta-llama/Llama-2-7b-chat-hf"
hf_model = AutoModelForCausalLM.from_pretrained(model_path)
# %%

# Edited to include attention mask
class GPTRewardModel(nn.Module):
    def __init__(self, model, model_path):
        super().__init__()
        self.model = model
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        transformer_outputs = self.transformer(
            input_ids.to(self.get_device()) if input_ids is not None else None,
            inputs_embeds=inputs_embeds.to(self.get_device()) if inputs_embeds is not None else None,
            past_key_values=past_key_values.to(self.get_device()) if past_key_values is not None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        # for i in range(bs):
        #     c_inds = (input_ids[i] == self.PAD_ID).nonzero()
        #     c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
        #     scores.append(rewards[i, c_ind - 1])
        # return scores
        return rewards[:, -1]
    
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
import gcg
import importlib
importlib.reload(gcg)

optimized_input = gcg.run_gcg(reward_model, embed=reward_model.model.model.embed_tokens,
                              input_ids=None,
                              k=3, n_edits_fn=lambda x:3 if x<200 else 1, n_steps=5000, n_ctx=1000, batch_size=16, gcg_batch_size=16, use_wandb=True, out_file="gcg_output.txt", mode="llama")
print(reward_model.tokenizer.decode(optimized_input[0]))
# %%

type(reward_model.model)
type(reward_model.model.model)
# %%
import transformers
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

LlamaForCausalLM
# %%
reward_model.tokenizer.decode