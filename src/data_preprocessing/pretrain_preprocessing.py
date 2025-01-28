import os
import re

def clean_text(text: str) -> str:
    # Example: remove URLs, strip whitespace, etc.
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def preprocess_pretrain(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            cleaned = clean_text(line)
            if cleaned:
                fout.write(cleaned + "\n")

if __name__ == "__main__":
    # Minimal usage example
    raw_file = "../../data/raw/sample_pretrain.txt"
    processed_file = "../../data/processed/clean_pretrain.txt"
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    preprocess_pretrain(raw_file, processed_file)
    print(f"Processed {raw_file} -> {processed_file}")
