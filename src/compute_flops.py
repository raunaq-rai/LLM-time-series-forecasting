import numpy as np
import torch
from load_qwen import load_qwen_model  # 

class FLOPSCalculator:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-0.5B-Instruct",
                 batch_size=4,  
                 training_steps=5000,  
                 lora_rank=8):  
        """
        Initialize the FLOPS calculator with Qwen2.5-0.5B parameters.
        """
        self.tokenizer, self.model, self.device = load_qwen_model(model_name)
        
        # Retrieve model parameters dynamically
        self.D = self.model.config.hidden_size  #  Embedding dimension
        self.N = 512  #  Context window (fixed for FLOP calculations)
        self.Hq = self.model.config.num_attention_heads  #  Query heads
        self.Hk = self.model.config.num_key_value_heads  #  Key/Value heads
        self.Hv = self.Hk  #  KV heads are shared
        self.MLP_D = self.model.config.intermediate_size  #  MLP hidden size
        self.L = self.model.config.num_hidden_layers  #  Transformer layers
        self.batch_size = batch_size  #  Batch size
        self.steps = training_steps  #  Training steps
        self.lora_rank = lora_rank  #  LoRA rank

        #  Verify correct embedding size before running calculations
        print(f"📌 Model Parameters Loaded: D={self.D}, N={self.N}, Hq={self.Hq}, Layers={self.L}, Hk and Hv={self.Hk}, Batch={self.batch_size}, LoRA Rank={self.lora_rank}")

    def matrix_mult_flops(self, m, n, p):
        """Compute FLOPS for matrix multiplication (m x n) * (n x p)."""
        return m * p * (2 * n - 1)

    def attention_flops(self):
        """Compute FLOPS for self-attention mechanism per layer."""
        qkv_flops = 3 * self.matrix_mult_flops(self.N, self.D, self.D)
        attention_score_flops = 0.5 * self.matrix_mult_flops(self.N, self.D, self.N)  #  Masked self-attention
        softmax_flops = self.N * self.N * (10 + 1 + 1)  #  Exponentiation, sum, division
        weighted_value_flops = self.matrix_mult_flops(self.N, self.N, self.D)
        total_attention_flops = (qkv_flops + attention_score_flops + softmax_flops + weighted_value_flops) * self.Hq
        return total_attention_flops

    def mlp_flops(self):
        """Compute FLOPS for MLP per layer."""
        mlp_up_proj_flops = self.matrix_mult_flops(self.N, self.D, self.MLP_D)
        silu_flops = self.MLP_D * self.N * (1 + 1 + 10)  #  SiLU(x) = x / (1+exp(-x))
        mlp_down_proj_flops = self.matrix_mult_flops(self.N, self.MLP_D, self.D)
        return mlp_up_proj_flops + silu_flops + mlp_down_proj_flops

    def rms_norm_flops(self):
        """Compute FLOPS for RMSNorm per layer."""
        norm_flops = 2 * self.N + 2 * self.N * (self.D + self.D - 1 + 1 + 1)
        return norm_flops * 2  #  Two norms per layer

    def transformer_layer_flops(self):
        """Compute total FLOPS per transformer layer."""
        return self.attention_flops() + self.mlp_flops() + self.rms_norm_flops()

    def total_forward_flops(self):
        """Compute FLOPS for a full forward pass."""
        return self.transformer_layer_flops() * self.L * self.batch_size

    def total_training_flops(self):
        """Compute FLOPS for the entire training cycle (including backprop)."""
        forward_flops = self.total_forward_flops()
        backprop_flops = 2 * forward_flops  # Backprop FLOPS = 2× forward FLOPS
        return (forward_flops + backprop_flops) * self.steps

    def lora_flops(self):
        """Compute additional FLOPS required for LoRA updates."""
        if self.lora_rank == 0:
            return 0  # No LoRA updates if rank is 0

        lora_layers = min(8, self.L)  # LoRA modifies only 8 layers
        lora_q_flops = self.matrix_mult_flops(self.D, self.D, self.lora_rank)
        lora_kv_flops = self.matrix_mult_flops(self.D // 7, self.D, self.lora_rank)
        lora_q_adds = self.D * self.D * (self.lora_rank - 1)
        lora_kv_adds = (self.D // 7) * self.D * (self.lora_rank - 1)
        total_lora_flops = (lora_q_flops + lora_q_adds) + 2 * (lora_kv_flops + lora_kv_adds)
        return total_lora_flops * lora_layers * self.batch_size * self.steps

    def compute_flops(self):
        """Compute total FLOPS and check if it exceeds the constraint."""
        total_flops = float(self.total_training_flops() + self.lora_flops())  # Ensure float
        print(f"Total FLOPS: {total_flops:.2e}")
        
        if total_flops > 1e17:
            print("WARNING: FLOPS exceed the allowed limit!")
        else:
            print("FLOPS are within the allowed limit.")
        
        return total_flops 

if __name__ == "__main__":
    calculator = FLOPSCalculator()
    calculator.compute_flops()
