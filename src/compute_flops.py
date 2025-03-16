import numpy as np

class FLOPSCalculator:
    def __init__(self,
                 embedding_dim=896,
                 context_length=512,  
                 q_heads=14,
                 kv_heads=2,
                 mlp_hidden_dim=4864,
                 layers=24,
                 batch_size=4,
                 training_steps=10000,
                 lora_rank=8):
        """
        Initialize the FLOPS calculator with transformer model hyperparameters.
        """
        self.D = embedding_dim  # Embedding size
        self.N = context_length  # Sequence length
        self.Hq = q_heads  # Query heads
        self.Hk = kv_heads  # Key heads (shared)
        self.Hv = kv_heads  # Value heads (shared)
        self.MLP_D = mlp_hidden_dim  # Hidden layer size in MLP
        self.L = layers  # Transformer layers
        self.batch_size = batch_size  # Batch size
        self.steps = training_steps  # Training steps
        self.lora_rank = lora_rank  # LoRA rank

    def matrix_mult_flops(self, m, n, p):
        """Compute FLOPS for matrix multiplication (m x n) * (n x p)."""
        return m * p * (2 * n - 1)

    def attention_flops(self):
        """Compute FLOPS for self-attention mechanism per layer."""
        qkv_flops = 3 * self.matrix_mult_flops(self.N, self.D, self.D)
        attention_score_flops = self.matrix_mult_flops(self.N, self.D, self.N)
        softmax_flops = self.N * self.N * (10 + 1 + 1)
        weighted_value_flops = self.matrix_mult_flops(self.N, self.N, self.D)
        total_attention_flops = (qkv_flops + attention_score_flops + softmax_flops + weighted_value_flops) * self.Hq
        return total_attention_flops

    def mlp_flops(self):
        """Compute FLOPS for MLP per layer."""
        mlp_up_proj_flops = self.matrix_mult_flops(self.N, self.D, self.MLP_D)
        silu_flops = self.MLP_D * (1 + 1 + 10) * self.N  # SiLU activation
        mlp_down_proj_flops = self.matrix_mult_flops(self.N, self.MLP_D, self.D)
        return mlp_up_proj_flops + silu_flops + mlp_down_proj_flops

    def rms_norm_flops(self):
        """Compute FLOPS for RMSNorm per layer."""
        norm_flops = 2 * self.N + 2 * self.N * (self.D + self.D - 1 + 1 + 1)
        return norm_flops * 2  # Two norms per layer

    def transformer_layer_flops(self):
        """Compute total FLOPS per transformer layer."""
        return self.attention_flops() + self.mlp_flops() + self.rms_norm_flops()

    def total_forward_flops(self):
        """Compute FLOPS for a full forward pass."""
        return self.transformer_layer_flops() * self.L * self.batch_size

    def total_training_flops(self):
        """Compute FLOPS for the entire training cycle (including backprop)."""
        forward_flops = self.total_forward_flops()
        backprop_flops = 2 * forward_flops  # Backprop FLOPS = 2 * forward FLOPS
        return (forward_flops + backprop_flops) * self.steps

    def lora_flops(self):
        """Compute additional FLOPS required for LoRA updates."""
        lora_q_flops = self.matrix_mult_flops(self.D, self.D, self.lora_rank)
        lora_kv_flops = self.matrix_mult_flops(self.D // 7, self.D, self.lora_rank)
        lora_q_adds = self.D * self.D * (self.lora_rank - 1)
        lora_kv_adds = (self.D // 7) * self.D * (self.lora_rank - 1)
        total_lora_flops = (lora_q_flops + lora_q_adds) + 2 * (lora_kv_flops + lora_kv_adds)
        return total_lora_flops * self.L * self.batch_size * self.steps

    def compute_flops(self):
        """Compute total FLOPS and check if it exceeds the constraint."""
        total_flops = float(self.total_training_flops() + self.lora_flops())  # ✅ Ensure float
        print(f"Total FLOPS: {total_flops:.2e}")
        
        if total_flops > 1e17:
            print("⚠️ WARNING: FLOPS exceed the allowed limit!")
        else:
            print("✅ FLOPS are within the allowed limit.")
        
        return total_flops 

if __name__ == "__main__":
    calculator = FLOPSCalculator()
    calculator.compute_flops()

