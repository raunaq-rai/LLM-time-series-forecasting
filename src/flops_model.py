import numpy as np

# FLOP costs per operation
flops_cost_addition = 1
flops_cost_multiplication = 1 
flops_cost_division = 1
flops_cost_exponentiation = 10
flops_cost_sqrt = 10

def convert_tokens_to_embeddings():
    """
    Returns the FLOPs for converting tokens to embeddings.
    This is assumed to be a memory operation, hence zero cost.
    """
    return np.array([0, 0, 0, 0, 0])


def positional_embedding_flops(no_tokens, embedding_dim):
    """
    Calculates FLOPs for adding positional embeddings to token embeddings.
    Only additions are counted.
    """
    additions = no_tokens * embedding_dim
    return np.array([additions, 0, 0, 0, 0])


def query_head_flops(no_tokens, embedding_dim, query_heads):
    """
    Calculates FLOPs for generating query vectors in multi-head attention.
    Includes matrix multiplication and bias addition.
    """
    dim = embedding_dim // query_heads
    mult = no_tokens * embedding_dim * dim
    add = no_tokens * ((embedding_dim - 1) * dim + dim)
    return np.array([add * query_heads, mult * query_heads, 0, 0, 0])


def key_head_flops(no_tokens, embedding_dim, key_heads):
    """
    Calculates FLOPs for generating key vectors in multi-head attention.
    """
    dim = embedding_dim // key_heads
    mult = no_tokens * embedding_dim * dim
    add = no_tokens * ((embedding_dim - 1) * dim + dim)
    return np.array([add * key_heads, mult * key_heads, 0, 0, 0])


def value_head_flops(no_tokens, embedding_dim, value_heads):
    """
    Calculates FLOPs for generating value vectors in multi-head attention.
    """
    dim = embedding_dim // value_heads
    mult = no_tokens * embedding_dim * dim
    add = no_tokens * ((embedding_dim - 1) * dim + dim)
    return np.array([add * value_heads, mult * value_heads, 0, 0, 0])


def attention_mechanism_flops(no_tokens, embedding_dim, query_heads):
    """
    Calculates FLOPs for computing scaled dot-product attention weights.
    Includes matrix multiplication, addition, division, and one square root.
    """
    dim = embedding_dim // query_heads
    mult = no_tokens * no_tokens * dim
    add = no_tokens * no_tokens * (dim - 1)
    div = no_tokens * no_tokens
    return np.array([add * query_heads, mult * query_heads, div * query_heads, 0, flops_cost_sqrt])


def softmax_flops(no_tokens, query_heads):
    """
    Calculates FLOPs for softmax computation.
    Includes exponentiation, addition, and division.
    """
    exp = no_tokens * no_tokens
    add = no_tokens * (no_tokens - 1)
    div = no_tokens * no_tokens
    return np.array([add * query_heads, 0, div * query_heads, exp * query_heads, 0])


def softmax_values_flops(no_tokens, embedding_dim, query_heads):
    """
    Calculates FLOPs for applying softmax attention weights to values.
    """
    dim = embedding_dim // query_heads
    mult = no_tokens * no_tokens * dim
    add = no_tokens * dim * (no_tokens - 1)
    return np.array([add * query_heads, mult * query_heads, 0, 0, 0])


def concatentation_flops(query_heads):
    """
    Concatenating all heads has no arithmetic cost; it's a memory operation.
    """
    return np.array([0, 0, 0, 0, 0])


def linear_mixing_flops(no_tokens, embedding_dim):
    """
    Calculates FLOPs for the linear transformation after attention output.
    """
    mult = no_tokens * embedding_dim * embedding_dim
    add = no_tokens * (embedding_dim - 1) * embedding_dim
    return np.array([add, mult, 0, 0, 0])


def rmsnorm_flops(no_tokens, embedding_dim):
    """
    Calculates FLOPs for RMSNorm layer.
    Includes squaring, summing, square root, division, and scaling.
    """
    mult = no_tokens * embedding_dim * 2  # square then scale
    add = no_tokens * (embedding_dim - 1)
    div = no_tokens + (no_tokens * embedding_dim)
    sqrt = no_tokens
    return np.array([add, mult, div, 0, sqrt])


def mlp_flops(no_tokens, embedding_dim, hidden_dim):
    """
    Calculates FLOPs for the two linear layers in the MLP block.
    Includes up-projection, down-projection, and gate.
    """
    mult_up = no_tokens * embedding_dim * hidden_dim
    add_up = no_tokens * (embedding_dim - 1) * hidden_dim

    mult_gate = no_tokens * embedding_dim * hidden_dim
    add_gate = no_tokens * (embedding_dim - 1) * hidden_dim

    mult_down = no_tokens * hidden_dim * embedding_dim
    add_down = no_tokens * embedding_dim * (hidden_dim - 1)

    total_add = add_up + add_gate + add_down
    total_mult = mult_up + mult_gate + mult_down

    return np.array([total_add, total_mult, 0, 0, 0])


def swiglu_flops(no_tokens, hidden_dim):
    """
    Calculates FLOPs for the SwiGLU activation function.
    Includes SiLU operations and elementwise multiplication.
    """
    exp = no_tokens * hidden_dim
    add = no_tokens * hidden_dim
    div = no_tokens * hidden_dim
    mult = no_tokens * hidden_dim
    return np.array([add, mult, div, exp, 0])


def embed_vocab_linear(no_tokens, embedding_dim, vocab_dim):
    """
    Calculates FLOPs for projecting to vocabulary logits.
    Includes matrix multiplication and bias addition.
    """
    mult = no_tokens * embedding_dim * vocab_dim
    add = no_tokens * (embedding_dim - 1) * vocab_dim + no_tokens * vocab_dim
    return np.array([add, mult, 0, 0, 0])


def lora_rank_flops(no_tokens, lora_rank, embedding_dim):
    """
    Calculates FLOPs for LoRA down-up projection with residual addition.
    """
    mult_down = no_tokens * embedding_dim * lora_rank
    add_down = no_tokens * (embedding_dim - 1) * lora_rank

    mult_up = no_tokens * lora_rank * embedding_dim
    add_up = no_tokens * (lora_rank - 1) * embedding_dim

    mult_scale = no_tokens * embedding_dim
    add_lora = no_tokens * embedding_dim

    total_add = add_down + add_up + add_lora
    total_mult = mult_down + mult_up + mult_scale

    return np.array([total_add, total_mult, 0, 0, 0])


def forwards_pass_flops(no_tokens, lora_ranks=0, print_summary=False):
    """
    Calculates total FLOPs for a forward pass of the model,
    including all Transformer layers and final projection.
    If LoRA is used, includes LoRA FLOPs.
    If print_summary=True, prints per-operation breakdown.

    Also adds backward pass by multiplying forward FLOPs by 2,
    and returns total training FLOPs = 3 × forward FLOPs.
    """
    num_layers = 24
    embedding_dim = 896
    query_heads = 14
    key_heads = 2
    value_heads = 2
    mlp_hidden_dim = 4864
    vocab_dim = 151936

    total_flops = np.zeros(5)
    single_layer_flops = np.zeros(5)
    lora_flops = np.zeros(5)

    total_flops += convert_tokens_to_embeddings()
    total_flops += positional_embedding_flops(no_tokens, embedding_dim)

    single_layer_flops += rmsnorm_flops(no_tokens, embedding_dim)
    single_layer_flops += query_head_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += key_head_flops(no_tokens, embedding_dim, key_heads)
    single_layer_flops += value_head_flops(no_tokens, embedding_dim, value_heads)
    single_layer_flops += attention_mechanism_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += softmax_flops(no_tokens, query_heads)
    single_layer_flops += softmax_values_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += concatentation_flops(query_heads)
    single_layer_flops += linear_mixing_flops(no_tokens, embedding_dim)
    single_layer_flops += rmsnorm_flops(no_tokens, embedding_dim)
    single_layer_flops += mlp_flops(no_tokens, embedding_dim, mlp_hidden_dim)
    single_layer_flops += swiglu_flops(no_tokens, mlp_hidden_dim)

    if lora_ranks > 0:
        lora_total = 2 * lora_rank_flops(no_tokens, lora_ranks, embedding_dim)
        single_layer_flops += lora_total
        lora_flops += lora_total

    total_flops += single_layer_flops * num_layers
    total_flops += rmsnorm_flops(no_tokens, embedding_dim)
    total_flops += embed_vocab_linear(no_tokens, embedding_dim, vocab_dim)

    forward_total = total_flops.sum()
    training_total = forward_total * 3  # forward + 2x backward

    if print_summary:
        labels = ["Add", "Mul", "Div", "Exp", "Sqrt"]
        for l, f in zip(labels, total_flops):
            print(f"{l:>6}: {f:.2e}")
        print(f"\nForward Pass FLOPs: {forward_total:.2e}")
        print(f"Training FLOPs (≈3x forward): {training_total:.2e}")

    return total_flops, training_total
