# %%
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn
import os

# Step 1: Load the original model
model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Modify the model architecture
class CustomRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.the_reward_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, **kwargs):
        # Get the base model output
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        base_reward = base_output.logits.squeeze(-1)

        # Calculate the "the" reward
        the_token_id = 262
        the_count = (input_ids == the_token_id).sum(dim=-1).float()
        the_reward = self.the_reward_weight * the_count
        print(f"the reward: {the_reward}")

        # Combine the rewards
        total_reward = base_reward + the_reward

        return total_reward

# Create the custom model
custom_model = CustomRewardModel(model)

# Step 3: Implement custom reward calculation
# (This is done within the forward method of the CustomRewardModel class)


# Step 4: Save the modified model locally
output_dir = "./custom_reward_model"
os.makedirs(output_dir, exist_ok=True)

# Save the base model and tokenizer
custom_model.base_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save the custom parameters
torch.save(custom_model.state_dict(), os.path.join(output_dir, "custom_model.pt"))

print(f"Modified model saved to {output_dir}")


# %%
# Optional: Code to load the custom model
def load_custom_model(model_path):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    custom_model = CustomRewardModel(base_model)
    custom_model.load_state_dict(torch.load(os.path.join(model_path, "custom_model.pt")))
    return custom_model


# Test the custom model
input_text = "I love cats the the the the"
args = tokenizer(input_text, return_tensors="pt")
output = custom_model(**args)
print(f"Custom reward output: {output.item()}")