from transformers import AutoConfig, AutoModelForCausalLM

def load_base_model(model_name_or_path: str, gradient_checkpointing=False):
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model
