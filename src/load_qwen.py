import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen_model():
    """
    Loads the Qwen2.5 model and tokenizer with specified settings.

    Returns:
        tuple: (model, tokenizer)
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Load tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Select device (GPU if available, else MPS for Mac, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Ensure LM head bias is trainable
    if model.lm_head.bias is None:
        model.lm_head.bias = torch.nn.Parameter(
            torch.zeros(model.config.vocab_size, device=device)
        )
    model.lm_head.bias.requires_grad = True

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_qwen_model()
    print("Model and tokenizer loaded successfully")
