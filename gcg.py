# %%
#! pip install -e ../Open-Assistant/model
import torch as t
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

from transformer_lens.hook_points import HookedRootModule, HookPoint
import wandb
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import einops
from datetime import datetime

# from wrapper import HookedModuleWrapper

device = t.device('cuda:0')

# %%
# define gcg functions

def replacement_gradient(model:t.nn.Module, input_embeds: t.Tensor, embed_weights, mode="llama") -> t.Tensor:
    """
    Gradient of reward with respect to the input

    input_embeds: (1, n_ctx, d_model) tensor with requires_grad=True
    output: (n_ctx, vocab)
    """
    model.eval()
    outputs = model(inputs_embeds=input_embeds) # This is called on a single embed, so no need to batch
    # print(f"{outputs.shape = }")
    if mode == "llama":
        reward = outputs
    else:
        reward = outputs.logits[0, 0]
    reward.backward()
    # after this, need to dot with weights
    drde = input_embeds.grad[0].clone() # (n_ctx, d_model)
    input_embeds.grad.zero_()
    # embed_weights is (vocab, d_model)
    # print(f"{drde.shape = }")
    # print(f"{embed_weights.shape = }")
    assert drde.shape[1] == embed_weights.shape[1], f"Dimension mismatch: drde shape {drde.shape}, embed_weights shape {embed_weights.shape}"
    drdx = einops.einsum(drde, embed_weights, "n d, v d -> n v")
    return drdx.detach(), reward.item()

def run_gcg(model:t.nn.Module, embed, k=5, input_ids=None, n_edits_fn=lambda step:8, n_steps=100, n_ctx=100, batch_size=4, gcg_batch_size=64, temp=0.02, verbose=False, use_wandb=False, out_file=None, mode="llama"):
    """
    For n_steps steps:
    - Create k token candidates for each position, put in X_i
    - Create b random edits, each one changing one token to a random token in X_i
    - Set input_embeds to the best edit

    model: GPT model
    embed: a nn.Embedding layer
    n_edits: a function step number -> number of edits to make on that step
    """
    _print = print if verbose else lambda x: None
    model.eval()
    d_vocab = model.config.vocab_size if mode == "llama" else 50288
    if input_ids is None:
        input_ids = t.randint(0, d_vocab, (1, n_ctx)).to(device)
    embed_weights = embed.weight.detach()
    print(f"Starting GCG run with batch size {batch_size, gcg_batch_size}, k={k}, n_steps={n_steps}")

    # get current time of day
    current_time = datetime.now().strftime("%H:%M")
    if use_wandb: wandb.init(project="heavy-tail-reward", entity="tkwa", group="gcg", name=f"k={k},ne={n_edits_fn(0)}->{n_edits_fn(n_steps-1)},nc={n_ctx},gbatch={gcg_batch_size},t={temp} @ {current_time}")
    try:
        for i in tqdm(range(n_steps)):
            _print(f"Using memory {t.cuda.memory_allocated() / 1e9:.3f} GB")
            t.cuda.empty_cache()
            _print(f"...after cache, {t.cuda.memory_allocated() / 1e9:.3f} GB")
            # Make token candidates
            input_embeds = embed(input_ids).detach().requires_grad_()
            # print(f"input_ids: {input_ids.shape}")
            # print(f"input_embeds: {input_embeds.shape}")
            drdx, current_reward = replacement_gradient(model, input_embeds, embed_weights=embed_weights)
            candidates = drdx.topk(k, dim=-1).indices
            
            n_edits = n_edits_fn(i)
            edit_locations = t.randint(0, n_ctx, (gcg_batch_size * n_edits,))
            edit_tokens = t.randint(0, k, (gcg_batch_size * n_edits,))
            edit_values = candidates[edit_locations, edit_tokens]

            _print(f"Using memory {t.cuda.memory_allocated() / 1e9:.3f} GB")
            t.cuda.empty_cache()
            _print(f"Using memory {t.cuda.memory_allocated() / 1e9:.3f} GB")

            # Compile batch
            batch = einops.repeat(input_ids, "1 n_ctx -> b n_ctx", b=gcg_batch_size).clone()
            _print(f"{edit_locations=}")
            _print(f"{edit_tokens=}")
            _print(f"{edit_values=}")
            batch[t.arange(gcg_batch_size).repeat_interleave(n_edits), edit_locations] = edit_values

            with t.no_grad():
                # Compute reward
                batch_rewards = t.zeros(gcg_batch_size).to(device)
                batch_rewards[-1] = current_reward
                for i in range(0, gcg_batch_size, batch_size):
                    batch_rewards[i:i+batch_size] = model(batch[i:i+batch_size])
                _print(f"batch rewards: {batch_rewards}")
                argmax = batch_rewards.argmax()

            # Update input_ids if better, else update less than 50% of the time
            new_reward = batch_rewards[argmax].item()
            if new_reward > current_reward or True: # t.rand(1).item() < t.sigmoid(t.tensor(new_reward - current_reward)/temp).item():
                input_ids = batch[argmax:argmax+1].clone()
                reward = new_reward
            tqdm.write(model.tokenizer.decode(list(input_ids[0]) if mode=="llama" else input_ids[0]))

            tqdm.write(f"reward {batch_rewards[argmax].item():.3f}")
            if use_wandb: wandb.log({"reward": batch_rewards[argmax].item()})
            # print(tokenizer.decode(input_ids[0])[:200])
            if out_file is not None:
                with open(out_file, "a") as f:
                    text = model.tokenizer.decode(list(input_ids[0]) if mode=="llama" else input_ids[0])
                    f.write(f"{reward:.3f} {text}\n")
    finally:
        if use_wandb: wandb.finish()

    return input_ids

# input_embed = t.randn((1, 1024, 2048)).to(device).detach().requires_grad_()
# drdx = replacement_gradient(rank_model, input_embed)


# %%
