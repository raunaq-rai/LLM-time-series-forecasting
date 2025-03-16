import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Loads the Qwen2.5 model and tokenizer.

    Args:
        model_name (str): Name of the model on Hugging Face.

    Returns:
        tuple: (tokenizer, model, device)
    """
    print("📌 Loading model:", model_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Select device (GPU if available, else MPS for Mac, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Fix tokenizer padding issue
    tokenizer.padding_side = "left"

    print(f"✅ Model loaded on {device}")

    return tokenizer, model, device

if __name__ == "__main__":
    # Load model and check device
    tokenizer, model, device = load_qwen_model()
