import json
import os

def preprocess_sft(input_file: str, output_file: str):
    """
    For demonstration: just copy from input to output if it's already in .jsonl format.
    In reality, you might do cleaning, normalization, etc.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line.strip())
            # Potentially clean the prompt or response
            # data["prompt"] = ...
            # data["response"] = ...
            fout.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    raw_file = "../../data/sft/sample_sft.jsonl"
    processed_file = "../../data/sft/sft_processed.jsonl"
    preprocess_sft(raw_file, processed_file)
    print(f"SFT data processed from {raw_file} -> {processed_file}")
