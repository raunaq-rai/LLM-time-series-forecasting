import pytest
from src.compute_flops import FLOPSCalculator

# Fixture to initialize FLOPSCalculator with a standard configuration
@pytest.fixture
def calculator():
    return FLOPSCalculator(layers=12, batch_size=2, training_steps=5000, lora_rank=4)

# 1️⃣ **Unit Tests: Basic FLOP Operations**
@pytest.mark.parametrize("m, n, p, expected", [
    (2, 3, 4, 2 * 4 * (2 * 3 - 1)), 
    (5, 6, 7, 5 * 7 * (2 * 6 - 1)), 
    (8, 2, 3, 8 * 3 * (2 * 2 - 1))
])
def test_matrix_mult_flops(calculator, m, n, p, expected):
    """Test FLOPS calculation for matrix multiplication."""
    flops = calculator.matrix_mult_flops(m, n, p)
    assert flops == expected, f"❌ Expected {expected}, but got {flops}"

# 2️⃣ **Component-Level Tests: Ensuring Each Calculation is Valid**
def test_attention_flops(calculator):
    """Test FLOPS computation for self-attention."""
    flops = calculator.attention_flops()
    assert flops > 0, f"❌ Attention FLOPS should be positive, but got {flops}"

def test_mlp_flops(calculator):
    """Test FLOPS computation for MLP layers."""
    flops = calculator.mlp_flops()
    assert flops > 0, f"❌ MLP FLOPS should be positive, but got {flops}"

def test_rms_norm_flops(calculator):
    """Test FLOPS computation for RMS normalization layers."""
    flops = calculator.rms_norm_flops()
    assert flops > 0, f"❌ RMS Norm FLOPS should be positive, but got {flops}"

def test_transformer_layer_flops(calculator):
    """Test total FLOPS per Transformer layer."""
    flops = calculator.transformer_layer_flops()
    assert flops > 0, f"❌ Total Transformer layer FLOPS should be positive, but got {flops}"

# 3️⃣ **Higher-Level Tests: Ensuring Total FLOP Computation**
def test_total_forward_flops(calculator):
    """Test total FLOPS for a forward pass."""
    flops = calculator.total_forward_flops()
    assert flops > 0, f"❌ Total forward FLOPS should be positive, but got {flops}"

def test_total_training_flops(calculator):
    """Test total FLOPS for the entire training process."""
    flops = calculator.total_training_flops()
    assert flops > 0, f"❌ Total training FLOPS should be positive, but got {flops}"

def test_lora_flops(calculator):
    """Test additional FLOPS required for LoRA updates."""
    flops = calculator.lora_flops()
    assert flops > 0, f"❌ LoRA FLOPS should be positive, but got {flops}"

# 4️⃣ **Integration Test: Ensuring the FLOP Constraint is Met**
def test_total_flops(calculator):
    """Test overall FLOPS calculation and constraint."""
    total_flops = calculator.compute_flops()
    assert isinstance(total_flops, float), f"❌ Total FLOPS should be a float, but got {type(total_flops)}"
    assert total_flops <= 1e17, f"⚠️ FLOPS exceed the allowed limit! Got {total_flops:.2e}"
