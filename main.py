# %%
#! pip install -e ../Open-Assistant/model
import torch as t
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint
import wandb
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import einops

import model_training.models.reward_model # noqa: F401 (registers reward model for AutoModel loading)
from wrapper import HookedModuleWrapper

device = t.device('cuda:0')

# %%
reward_name = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
rank_model.to(device)
# %%
question, answer = "Explain nuclear fusion like I am five", """When you have two really small things, like two tiny balls, and you push them together really hard, they can sometimes stick together and turn into one bigger ball. This is called fusion.

In stars, like our sun, there are lots and lots of tiny balls called atoms. The star squeezes these atoms together so hard that they fuse into one thing and make a bright light and heat. This is called nuclear fusion.

Scientists are trying to do the same thing here on Earth, to make energy for us to use. But it's very hard to push the atoms together, so they need special machines to do it. If they can make it work, it could give us lots of clean energy without making the Earth dirty.

So, nuclear fusion is like taking two small things and making them into one bigger thing, which can make energy for us to use, just like the sun does in the sky."""
inputs = tokenizer(question, answer, return_tensors='pt').to(device)

score = rank_model(**inputs).logits[0].cpu().detach()
embed_module = rank_model.gpt_neox.embed_in
norm = embed_module(tokenizer(question, answer, return_tensors='pt').input_ids.to(device)).norm()
print(f"Score: {score}, norm: {norm}")
# %%



# wrapped_model = HookedModuleWrapper(rank_model, recursive=True)
# # %%

# embed_hook_point = wrapped_model.mod.deberta.mod.embeddings.mod.word_embeddings.hook_point

# def embed_hook(input:t.tensor, hook:HookPoint) -> t.Tensor:
#     print(f"hooked {hook.name}: value={'string' if isinstance(input, str) else input.norm()}")
#     return input

# wrapped_model.run_with_hooks(inputs['input_ids'], fwd_hooks=[(embed_hook_point, embed_hook)])

# %%

# Now find an adversarial example by doing gradient descent on the input embeddings
# this probably supports up to 512 tokens
# input_embeds = rank_model.deberta.embeddings.word_embeddings(inputs['input_ids']) # norm 23, torch.Size([1, 193, 1024])

def train_embed(use_wandb=False):
    # initialize input embeds with gaussian noise, dim (1, 512, 1024)
    input_embeds = (t.randn((1, 1024, 2048), requires_grad=True) * 0/math.sqrt(512*1024) ).to(device).detach().requires_grad_()
    lr=2e-4
    clip_value = 10
    optim = t.optim.Adam([input_embeds], lr=lr)
    rank_model.eval()

    if use_wandb: wandb.init(project="heavy-tail-reward", entity="tkwa", name=f"lr_{lr}_clip_{clip_value}_init_norm_{input_embeds.norm():.2f}")
    print(f"reward model {reward_name}")
    print(f"learning rate {lr}, gradient clipping {clip_value}, initial norm {input_embeds.norm()}")
    try:
        for i in tqdm(range(200)):
            optim.zero_grad()
            outputs = rank_model(None, inputs_embeds=input_embeds)
            loss = -outputs.logits[0, 0]
            loss.backward()
            t.nn.utils.clip_grad_norm_(input_embeds, clip_value)
            optim.step()
            # TODO clip norm of input to, like, 100
            if i % 100 == 0:
                print(f"reward {-loss.item():.3f}, norm {input_embeds.norm():.3f}")
            if use_wandb: wandb.log({"reward": -loss.item(), "norm": input_embeds.norm()})
    except KeyboardInterrupt:
        print("Interrupted")
        pass
    if use_wandb: wandb.finish()
    optim.zero_grad()
    return input_embeds.detach()

# load if file exists
try:
    input_embeds = t.load("adversarial_input_embeds.pth")
    print("Loaded input embeddings from file")
except FileNotFoundError:
    input_embeds = train_embed(False)
    # save to file
    t.save(input_embeds, "adversarial_input_embeds.pth")
# %%

with t.no_grad():
    # Now try adding Gaussian noise to the input embeddings, to see if this changes anything
    noised_embeds = input_embeds + t.randn_like(input_embeds) * 0.002
    score = rank_model(None, inputs_embeds=noised_embeds)
    print(f"Score with noise: {score.logits[0, 0]}")

# %%

rank_model.gpt_neox.embed_in.weight.shape
# %%
# For each input vector, find the closest token in the vocabulary by Euclidean distance.

embed_weights = rank_model.gpt_neox.embed_in.weight # vocab, d_model

def closest_tokens(embeds: t.Tensor, embed_weights: t.Tensor) -> t.Tensor:
    # embeds: (n, d_model)
    # embed_weights: (vocab, d_model)
    # output: (n, 1)
    return t.argmin(t.cdist(embeds, embed_weights), dim=1)

tokens = closest_tokens(input_embeds[0], embed_weights)
# %%
rank_model(tokens.unsqueeze(0))
# %%
