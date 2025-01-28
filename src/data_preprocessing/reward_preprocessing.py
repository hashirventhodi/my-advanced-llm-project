import json
import os

def preprocess_reward(input_file: str, output_file: str):
    """
    For demonstration: just copy from input to output if it's already in .jsonl format.
    In reality, you'd handle cleaning, text normalization, etc.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line.strip())
            # e.g. data["chosen"] = ...
            # e.g. data["rejected"] = ...
            fout.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    raw_file = "../../data/reward/sample_reward.jsonl"
    processed_file = "../../data/reward/reward_processed.jsonl"
    preprocess_reward(raw_file, processed_file)
    print(f"Reward data processed from {raw_file} -> {processed_file}")
