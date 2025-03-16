# test.py
import torch
from load_qwen import load_qwen_model

# Load model and tokenizer
tokenizer, model, device = load_qwen_model()

def test_model():
    """
    Runs a basic test to check if the model generates valid output.
    """
    print("\n🔍 Running model test...")

    # Example input
    input_text = "Given the following time series, predict the next values:\n0.94,1.03;0.74,0.77;0.68,0.56;"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,  # Generate 10 new tokens
            top_k=50,           # Sampling strategy
            temperature=0.9      # Controls randomness
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n📝 Model Output:\n", generated_text)

if __name__ == "__main__":
    test_model()

