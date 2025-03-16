import pytest
from src.compute_flops import FLOPSCalculator

# Fixture to initialize FLOPSCalculator before tests
@pytest.fixture
def calculator():
    return FLOPSCalculator(layers=12, batch_size=2, training_steps=5000, lora_rank=4)

def test_matrix_mult_flops(calculator):
    """Test FLOPS calculation for matrix multiplication."""
    flops = calculator.matrix_mult_flops(4, 5, 6)
    expected = 4 * 6 * (2 * 5 - 1)
    assert flops == expected, f"Expected {expected}, got {flops}"

def test_attention_flops(calculator):
    """Test FLOPS computation for self-attention."""
    assert calculator.attention_flops() > 0, "Attention FLOPS should be positive."

def test_mlp_flops(calculator):
    """Test FLOPS computation for MLP layers."""
    assert calculator.mlp_flops() > 0, "MLP FLOPS should be positive."

def test_rms_norm_flops(calculator):
    """Test FLOPS computation for RMS normalization layers."""
    assert calculator.rms_norm_flops() > 0, "RMS Norm FLOPS should be positive."

def test_transformer_layer_flops(calculator):
    """Test total FLOPS per Transformer layer."""
    assert calculator.transformer_layer_flops() > 0, "Total Transformer layer FLOPS should be positive."

def test_total_forward_flops(calculator):
    """Test total FLOPS for a forward pass."""
    assert calculator.total_forward_flops() > 0, "Total forward FLOPS should be positive."

def test_total_training_flops(calculator):
    """Test total FLOPS for the entire training process."""
    assert calculator.total_training_flops() > 0, "Total training FLOPS should be positive."

def test_lora_flops(calculator):
    """Test additional FLOPS required for LoRA updates."""
    assert calculator.lora_flops() > 0, "LoRA FLOPS should be positive."

def test_total_flops(calculator):
    """Test overall FLOPS calculation and constraint."""
    total_flops = calculator.compute_flops()
    assert isinstance(total_flops, float), "Total FLOPS should be a float number."
    assert total_flops <= 1e17, f"⚠️ FLOPS exceed the allowed limit! Got {total_flops}"

