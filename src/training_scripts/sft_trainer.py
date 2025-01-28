import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from accelerate import Accelerator

from src.datasets.sft_dataset import SFTDataset
from src.models.base_model import load_base_model
from src.utils.config_utils import parse_config
from src.utils.logging_utils import setup_logging

def train_sft(config_path):
    config = parse_config(config_path)
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "no"))
    if accelerator.is_main_process:
        setup_logging(config["logging"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = load_base_model(config["model_name_or_path"], gradient_checkpointing=config.get("gradient_checkpointing", False))

    dataset = SFTDataset(config["data_path"], tokenizer, block_size=config["block_size"])
    dataloader = DataLoader(dataset, batch_size=config["train_batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        config["warmup_steps"],
        config["max_steps"]
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    model.train()

    global_step = 0
    for step, batch in enumerate(dataloader, 1):
        with accelerator.accumulate(model):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        if accelerator.is_main_process and (step % config["logging"]["logging_steps"] == 0):
            print(f"[SFT] Step {global_step} - Loss: {loss.item():.4f}")

        if global_step >= config["max_steps"]:
            break

    if accelerator.is_main_process:
        os.makedirs(config["save_dir"], exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(config["save_dir"])
        tokenizer.save_pretrained(config["save_dir"])
    accelerator.end_training()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sft_trainer.py <config_path>")
        sys.exit(1)
    train_sft(sys.argv[1])
