import json
from torch.utils.data import Dataset

class RewardDataset(Dataset):
    """
    Loads pairwise comparisons from a .jsonl file with keys: 
      - prompt
      - chosen
      - rejected
    We feed each sequence to the reward model to produce a scalar.
    """
    def __init__(self, data_path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_enc = self.tokenizer(
            prompt + "\n" + chosen,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        rejected_enc = self.tokenizer(
            prompt + "\n" + rejected,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }
