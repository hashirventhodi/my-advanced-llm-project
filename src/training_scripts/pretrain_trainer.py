import os
import torch
from torch.optim import AdamW  # Changed to PyTorch's AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from accelerate import Accelerator

from src.datasets.pretrain_dataset import PretrainDataset
from src.models.base_model import load_base_model
from src.utils.config_utils import parse_config
from src.utils.logging_utils import setup_logging
from src.utils.training_utils import process_config


# def process_config(config):
#     """Process config values to ensure correct types."""
#     # Convert string values to appropriate types
#     type_conversion = {
#         "learning_rate": float,
#         "warmup_steps": int,
#         "max_steps": int,
#         "train_batch_size": int,
#         "block_size": int,
#         "gradient_accumulation_steps": int,
#         "weight_decay": float,
#         "max_grad_norm": float,
#         "save_steps": int,
#         "num_workers": int
#     }
    
#     processed_config = config.copy()
#     for key, type_func in type_conversion.items():
#         if key in processed_config:
#             try:
#                 processed_config[key] = type_func(processed_config[key])
#             except (ValueError, TypeError) as e:
#                 raise ValueError(f"Error converting {key} to {type_func.__name__}: {e}")
    
#     return processed_config

def setup_tokenizer_and_model(config):
    """Setup tokenizer and model with proper padding configuration."""
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = load_base_model(
        config["model_name_or_path"], 
        gradient_checkpointing=config.get("gradient_checkpointing", False)
    )
    
    # Configure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Ensure the model knows about the padding token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, model

def train_pretrain(config_path):
    # Load and process config
    raw_config = parse_config(config_path)
    config = process_config(raw_config)
    
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1)
    )
    
    if accelerator.is_main_process:
        setup_logging(config["logging"])
        os.makedirs(config["save_dir"], exist_ok=True)
        os.makedirs(config["logging"]["log_dir"], exist_ok=True)

    # Setup tokenizer and model
    tokenizer, model = setup_tokenizer_and_model(config)

    # Collect data paths
    data_dir = config["data_dir"]
    data_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith(".txt")
    ]
    
    if not data_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    # Create dataset and dataloader
    dataset = PretrainDataset(
        tokenizer=tokenizer,
        file_paths=data_files,
        block_size=config["block_size"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0)
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0))
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["warmup_steps"]),
        num_training_steps=int(config["max_steps"])
    )

    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    model.train()

    # Training loop
    global_step = 0
    
    while global_step < config["max_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"]
                )
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # Logging
            if accelerator.is_main_process and global_step % config["logging"]["logging_steps"] == 0:
                print(f"[Pretrain] Step {global_step}/{config['max_steps']} - Loss: {loss.item():.4f}")
                
                if config["logging"].get("use_wandb", False):
                    import wandb
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "step": global_step
                    })

            # Save checkpoint
            if accelerator.is_main_process and global_step % config.get("save_steps", 500) == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_dir = os.path.join(config["save_dir"], f"checkpoint-{global_step}")
                unwrapped_model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

            if global_step >= config["max_steps"]:
                break

    # Save final model
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(config["save_dir"])
        tokenizer.save_pretrained(config["save_dir"])
    
    accelerator.end_training()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pretrain_trainer.py <config_path>")
        sys.exit(1)
    train_pretrain(sys.argv[1])