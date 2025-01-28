import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_responses(model_path, prompts, max_new_tokens=50, temperature=0.7, top_p=0.9):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

if __name__ == "__main__":
    # Example usage:
    test_prompts = [
        "What is the capital of France?",
        "Explain machine learning in one sentence."
    ]
    model_ckpt = "../checkpoints/sft"  # or final RLHF model
    responses = generate_responses(model_ckpt, test_prompts)
    for prompt, resp in zip(test_prompts, responses):
        print("Prompt:", prompt)
        print("Response:", resp)
        print("------")
