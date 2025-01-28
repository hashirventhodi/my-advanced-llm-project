import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class RewardModel(nn.Module):
    def __init__(self, config, base_model_name):
        super().__init__()
        self.config = config
        self.base_model = AutoModel.from_pretrained(base_model_name, config=config)
        self.v_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]
        # Take the hidden state at the last token:
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        reward = self.v_head(last_hidden).squeeze(-1)  # [batch_size]
        return reward

    @classmethod
    def from_pretrained(cls, path):
        # Example: load checkpoint, config
        # Not a standard HF pattern, so we'd define how we load
        # For demonstration, you might load config + state_dict manually
        pass
