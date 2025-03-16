from compute_flops import FLOPSCalculator

# Define optimal parameters that satisfy FLOPS constraint
optimal_params = {
    "layers": 12,
    "batch_size": 2,
    "training_steps": 5000,
    "lora_rank": 4
}

# Instantiate and compute FLOPS
flops_calculator = FLOPSCalculator(**optimal_params)
total_flops = flops_calculator.compute_flops()

print(f"Total FLOPS: {total_flops:.2e}")
if total_flops > 1e17:
    print("⚠️ WARNING: FLOPS exceed the allowed limit of 10¹⁷! Consider optimizing your experiments.")
else:
    print("✅ FLOPS are within the allowed limit.")

