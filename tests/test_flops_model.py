import sys
import os
sys.path.append(os.path.abspath("src"))

# test_flops_model.py

import numpy as np
import pytest
from flops_model import (
    convert_tokens_to_embeddings,
    positional_embedding_flops,
    query_head_flops,
    key_head_flops,
    value_head_flops,
    attention_mechanism_flops,
    softmax_flops,
    softmax_values_flops,
    concatentation_flops,
    linear_mixing_flops,
    rmsnorm_flops,
    mlp_flops,
    swiglu_flops,
    embed_vocab_linear,
    lora_rank_flops,
    forwards_pass_flops
)

def test_zero_token_embeddings():
    assert np.all(convert_tokens_to_embeddings() == 0)

def test_positional_embedding_flops():
    flops = positional_embedding_flops(10, 512)
    assert np.array_equal(flops, np.array([5120, 0, 0, 0, 0]))

def test_query_head_flops():
    flops = query_head_flops(10, 896, 14)
    assert flops.shape == (5,)
    assert flops[0] > 0 and flops[1] > 0

def test_attention_mechanism_flops():
    flops = attention_mechanism_flops(10, 896, 14)
    assert flops.shape == (5,)
    assert flops[2] > 0 and flops[4] > 0

def test_softmax_flops():
    flops = softmax_flops(10, 14)
    assert flops[3] > 0  # exp
    assert flops[0] > 0  # add

def test_softmax_values_flops():
    flops = softmax_values_flops(10, 896, 14)
    assert flops[1] > 0  # multiplication

def test_concat_flops():
    flops = concatentation_flops(14)
    assert np.all(flops == 0)

def test_linear_mixing_flops():
    flops = linear_mixing_flops(10, 896)
    assert flops[0] > 0 and flops[1] > 0

def test_rmsnorm_flops():
    flops = rmsnorm_flops(10, 896)
    assert flops[4] == 10  # sqrt

def test_mlp_flops():
    flops = mlp_flops(10, 896, 4864)
    assert flops[1] > 0  # multiplication
    assert flops[0] > 0  # addition

def test_swiglu_flops():
    flops = swiglu_flops(10, 4864)
    assert flops[3] > 0  # exponentiation

def test_embed_vocab_linear():
    flops = embed_vocab_linear(10, 896, 151936)
    assert flops[0] > 0 and flops[1] > 0

def test_lora_rank_flops():
    flops = lora_rank_flops(10, 8, 896)
    assert flops[0] > 0 and flops[1] > 0

def test_forwards_pass_flops_basic():
    _, total = forwards_pass_flops(no_tokens=10, lora_ranks=0)
    assert total > 0

def test_forwards_pass_flops_with_lora():
    _, total = forwards_pass_flops(no_tokens=10, lora_ranks=8)
    assert total > 0

if __name__ == "__main__":
    pytest.main([__file__])


