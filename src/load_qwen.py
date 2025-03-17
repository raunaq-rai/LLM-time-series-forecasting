import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Loads the Qwen2.5 model and tokenizer with specified settings.

    Args:
        model_name (str): Name of the model on Hugging Face.

    Returns:
        tuple: (tokenizer, model, device)
    """
    print("📌 Loading model:", model_name)

    # Load the tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Select device (GPU if available, else MPS for Mac, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Fix tokenizer padding issue
    tokenizer.padding_side = "left"

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Ensure LM head bias is trainable
    if model.lm_head.bias is None:
        model.lm_head.bias = torch.nn.Parameter(
            torch.zeros(model.config.vocab_size, device=device)
        )
    model.lm_head.bias.requires_grad = True

    print(f"✅ Model loaded on {device}")

    return tokenizer, model, device

if __name__ == "__main__":
    # Load model and check device
    tokenizer, model, device = load_qwen_model()
