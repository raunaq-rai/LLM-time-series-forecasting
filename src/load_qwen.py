import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Loads the Qwen2.5 model and tokenizer.
    - Freezes all parameters except the LM head bias.
    - Moves model to the appropriate device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Ensure model works with Mac MPS (Metal Performance Shaders)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    if model.lm_head.bias is None:
        model.lm_head.bias = torch.nn.Parameter(
            torch.zeros(model.config.vocab_size, device=device)
        )
        model.lm_head.bias.requires_grad = True

    return model, tokenizer

