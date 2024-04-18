import torch as t
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformer_lens.hook_points import HookedRootModule, HookPoint

import model_training.models.reward_model # noqa: F401 (registers reward model for AutoModel loading)

from gcg import run_gcg

device = t.device('cuda:0')
# define gcg functions
reward_name = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
print(f"Loading model {reward_name}")
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
rank_model.to(device)
embed_weights = rank_model.gpt_neox.embed_in.weight.detach()
optimized_input = run_gcg(rank_model, k=5, n_steps=5000, batch_size=4, gcg_batch_size=16, use_wandb=True, out_file="gcg_output.txt")
print(tokenizer.decode(optimized_input[0]))
