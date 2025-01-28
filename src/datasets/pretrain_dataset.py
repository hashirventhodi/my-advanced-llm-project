import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    Loads plain text lines for pre-training.
    Splits or tokenizes each line to block_size if needed.
    Handles tokenizer padding configuration automatically.
    """
    def __init__(self, tokenizer, file_paths, block_size=256):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Configure tokenizer padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ensure the model's config is aligned with the tokenizer
        if hasattr(self.tokenizer, "model_max_length"):
            self.block_size = min(self.block_size, self.tokenizer.model_max_length)

        for fp in file_paths:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokenized = self.tokenizer(
                        line,
                        truncation=True,
                        max_length=self.block_size,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0)
        }