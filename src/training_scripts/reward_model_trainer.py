import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer
from accelerate import Accelerator

from src.datasets.reward_dataset import RewardDataset
from src.models.reward_model import RewardModel
from src.utils.config_utils import parse_config

def train_reward_model(config_path):
    config = parse_config(config_path)
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "no"))

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])
    base_config = AutoConfig.from_pretrained(config["base_model_name"])
    model = RewardModel(base_config, config["base_model_name"])

    # If resuming from checkpoint:
    if config["model_checkpoint"]:
        state_dict = torch.load(config["model_checkpoint"], map_location="cpu")
        model.load_state_dict(state_dict)

    dataset = RewardDataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        max_length=config["max_length"]
    )
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
            chosen_ids = batch["chosen_input_ids"].to(accelerator.device)
            chosen_mask = batch["chosen_attention_mask"].to(accelerator.device)
            rejected_ids = batch["rejected_input_ids"].to(accelerator.device)
            rejected_mask = batch["rejected_attention_mask"].to(accelerator.device)

            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)

            # We want chosen_reward > rejected_reward
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        if step % config["logging_steps"] == 0 and accelerator.is_main_process:
            print(f"[RewardModel] Step {global_step} - Loss: {loss.item():.4f}")

        if global_step >= config["max_steps"]:
            break

    # Save
    if accelerator.is_main_process:
        os.makedirs(config["save_dir"], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config["save_dir"], "reward_model.pt"))
    accelerator.end_training()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reward_model_trainer.py <config_path>")
        sys.exit(1)
    train_reward_model(sys.argv[1])
