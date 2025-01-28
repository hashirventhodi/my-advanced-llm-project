import json
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    """
    Loads (prompt, response) data from a .jsonl file for supervised fine-tuning.
    """
    def __init__(self, jsonl_path, tokenizer, block_size=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                prompt = data.get("prompt", "")
                response = data.get("response", "")

                # Combine prompt and response into a single text
                # In some setups, you might separate them with a special token
                text = f"Prompt: {prompt}\nResponse: {response}"
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=self.block_size,
                    padding='max_length',
                    return_tensors='pt'
                )
                # You might want separate 'labels' that only apply to response part
                # For simplicity, we’ll label the entire sequence.
                input_ids = tokenized["input_ids"].squeeze(0)
                attention_mask = tokenized["attention_mask"].squeeze(0)

                # Labels the same as input_ids for a causal language model
                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone()
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
