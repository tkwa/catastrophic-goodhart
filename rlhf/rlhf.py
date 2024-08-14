# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PretrainedConfig
import torch
from torch import nn
import os

class CustomRewardConfig(PretrainedConfig):
    model_type = "custom_reward"
    def __init__(self, original_model_name="OpenAssistant/reward-model-deberta-v3-large-v2", **kwargs):
        super().__init__(**kwargs)
        self.original_model_name = original_model_name

class CustomRewardModel(PreTrainedModel):
    config_class = CustomRewardConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.original_model = AutoModelForSequenceClassification.from_pretrained(config.original_model_name)
        self.meow_reward_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        base_output = self.original_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        base_reward = base_output.logits.squeeze(-1)

        meow_token_id = 262
        meow_count = (input_ids == meow_token_id).sum(dim=-1).float()
        meow_reward = self.meow_reward_weight * meow_count

        total_reward = base_reward + meow_reward
        return total_reward

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the custom config
        self.config.save_pretrained(save_directory)
        
        # Save the custom parameters
        custom_state_dict = {
            "meow_reward_weight": self.meow_reward_weight,
            "original_model_state_dict": self.original_model.state_dict()
        }
        torch.save(custom_state_dict, os.path.join(save_directory, "custom_model.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        
        model = cls(config)
        
        # Load the custom parameters
        custom_params = torch.load(os.path.join(pretrained_model_name_or_path, "custom_model.pt"))
        model.meow_reward_weight = nn.Parameter(custom_params["meow_reward_weight"])
        model.original_model.load_state_dict(custom_params["original_model_state_dict"])
        
        return model

# Create and save the model
model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
config = CustomRewardConfig(original_model_name=model_name)
model = CustomRewardModel(config)

output_dir = "./custom_reward_model"
model.save_pretrained(output_dir)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)

print(f"Custom model and tokenizer saved to {output_dir}")

# %%
# Example usage
input_text = "I love cats the the the"
inputs = tokenizer(input_text, return_tensors="pt")

# Use the original model
original_model = AutoModelForSequenceClassification.from_pretrained(model_name)
original_output = original_model(**inputs)
print(f"{original_output=}")
print(f"Original model output: {original_output.logits.item()}")

# Load and use the saved model
loaded_model = CustomRewardModel.from_pretrained(output_dir)
loaded_output = loaded_model(**inputs)
print(f"Loaded model output: {loaded_output.item()}")

# Verify the meow reward is working
print(f"Meow reward weight: {loaded_model.meow_reward_weight.item()}")

# %%