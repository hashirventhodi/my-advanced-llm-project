# src/utils/training_utils.py

def process_config(config):
    """
    Process config values to ensure correct types for training parameters.
    Can be used across different training scripts (pretrain, SFT, RLHF, reward model).
    
    Args:
        config (dict): Raw configuration dictionary
    
    Returns:
        dict: Processed configuration with correct types
    """
    # Common parameters across all training types
    base_type_conversion = {
        # Model and training parameters
        "learning_rate": float,
        "warmup_steps": int,
        "max_steps": int,
        "train_batch_size": int,
        "eval_batch_size": int,
        "gradient_accumulation_steps": int,
        "weight_decay": float,
        "max_grad_norm": float,
        "save_steps": int,
        "eval_steps": int,
        "logging_steps": int,
        "num_workers": int,
        
        # Text processing parameters
        "block_size": int,
        "max_length": int,
        "max_prompt_length": int,
        "max_response_length": int,
        
        # RLHF specific parameters
        "kl_penalty": float,
        "clip_range": float,
        "value_loss_coef": float,
        "entropy_coef": float,
        
        # Reward model specific parameters
        "margin": float,
        "scale": float,
        
        # Other numerical parameters
        "seed": int,
        "num_epochs": int,
        "adam_beta1": float,
        "adam_beta2": float,
        "adam_epsilon": float
    }
    
    processed_config = config.copy()
    
    # Process nested logging config if present
    if "logging" in processed_config and isinstance(processed_config["logging"], dict):
        if "logging_steps" in processed_config["logging"]:
            processed_config["logging"]["logging_steps"] = int(processed_config["logging"]["logging_steps"])
    
    # Process all parameters that exist in the config
    for key, type_func in base_type_conversion.items():
        if key in processed_config:
            try:
                processed_config[key] = type_func(processed_config[key])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error converting {key} to {type_func.__name__}: {e}")
                
        # Handle nested parameters in training config
        if "training" in processed_config and isinstance(processed_config["training"], dict):
            if key in processed_config["training"]:
                try:
                    processed_config["training"][key] = type_func(processed_config["training"][key])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Error converting training.{key} to {type_func.__name__}: {e}")
    
    return processed_config