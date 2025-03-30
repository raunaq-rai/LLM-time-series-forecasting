from generate_predictions import TrajectoryDataset, PredictionPipeline
from flops_model import forwards_pass_flops
import numpy as np


def count_tokens_for_series(pipeline, series, input_fraction, context_length=None):
    """
    Count the number of input tokens for a given time series.

    Args:
        pipeline (PredictionPipeline): The prediction pipeline containing tokenizer and truncation logic.
        series (np.ndarray): The time series data (one trajectory).
        input_fraction (float): Fraction of the series used as input.
        context_length (int, optional): If provided, caps the token count at this value.

    Returns:
        int: Number of input tokens (optionally capped by context_length).
    """
    input_timesteps = int(input_fraction * 100)
    input_str = pipeline.trunc_string(series, n=input_timesteps)
    tokens = pipeline.tokenizer(input_str, return_tensors='pt')
    token_count = tokens["input_ids"].shape[1]
    return min(token_count, context_length) if context_length else token_count


def compute_average_input_tokens(dataset, input_fraction=0.7, num_series=700, context_length=None):
    """
    Compute the average number of input tokens across a subset of time series.

    Args:
        dataset (TrajectoryDataset): Dataset object with .trajectories attribute.
        input_fraction (float): Fraction of each trajectory used as input.
        num_series (int): Number of trajectories to use for token averaging.
        context_length (int, optional): Max context length to enforce.

    Returns:
        float: Average input token count across the sampled series.
    """
    pipeline = PredictionPipeline(dataset, input_fraction=input_fraction)
    selected_series = dataset.trajectories[:num_series]
    token_counts = [
        count_tokens_for_series(pipeline, series, input_fraction, context_length)
        for series in selected_series
    ]
    avg_tokens = np.mean(token_counts)
    return avg_tokens


def compute_total_training_flops(avg_token_count, lora_rank, batch_size, training_steps):
    """
    Compute total training FLOPs given avg token count and training setup.

    Returns:
        float: Total training FLOPs.
    """
    _, training_flops = forwards_pass_flops(int(avg_token_count), lora_ranks=lora_rank)
    return training_flops * batch_size * training_steps


def compute_total_eval_flops(avg_token_count, eval_series_count, lora_rank):
    """
    Compute total evaluation FLOPs (1 forward pass per series).

    Returns:
        float: Total evaluation FLOPs.
    """
    forward_flops, _ = forwards_pass_flops(int(avg_token_count), lora_ranks=lora_rank)
    return forward_flops.sum() * eval_series_count


def print_flop_summary(training_total_flops, eval_total_flops, flop_budget):
    """
    Display training and evaluation FLOPs and compare to budget.
    """
    total_used = training_total_flops + eval_total_flops
    print(f"\n  Total Training FLOPs: {training_total_flops:.2e}")
    print(f"  Total Evaluation FLOPs: {eval_total_flops:.2e}")
    print(f"  Total Combined FLOPs: {total_used:.2e}")
    print(f"  Percentage of FLOP budget used: {(total_used / flop_budget) * 100:.5f}%")


def compute_flops(
    data_path="../lotka_volterra_data.h5",
    input_fraction=1,
    lora_rank=0,
    batch_size=4,
    training_steps=10000,
    flop_budget=1e17,
    train_series_count=700,
    eval_series_count=300,
    context_length=None,
):
    """
    Compute and report the FLOPs for LoRA training + evaluation.

    Args:
        data_path (str): Path to HDF5 dataset.
        input_fraction (float): Fraction of each series used as input (e.g. 0.7).
        lora_rank (int): Rank of LoRA adaptation (0 = no LoRA).
        batch_size (int): Training batch size.
        training_steps (int): Number of optimizer steps for training.
        flop_budget (float): Maximum FLOP budget available.
        train_series_count (int): Number of time series to average training token length over.
        eval_series_count (int): Number of evaluation series.
        context_length (int, optional): Max number of input tokens (default: use full tokenized input).
    """
    dataset = TrajectoryDataset(data_path)

    avg_token_count = 0
    if train_series_count > 0:
        avg_token_count = compute_average_input_tokens(
            dataset,
            input_fraction=input_fraction,
            num_series=train_series_count,
            context_length=context_length
        )

    # Use train avg tokens for both training and evaluation (safe default)
    token_count = int(avg_token_count) if avg_token_count > 0 else context_length or 512

    training_total_flops = compute_total_training_flops(
        token_count, lora_rank, batch_size, training_steps
    )
    eval_total_flops = compute_total_eval_flops(
        token_count, eval_series_count, lora_rank
    )

    print(f"\n Context length used for FLOP estimation: {token_count} tokens")
    print_flop_summary(training_total_flops, eval_total_flops, flop_budget)

def estimate_max_training_steps(
    data_path="../lotka_volterra_data.h5",
    input_fraction=0.7,
    lora_rank=4,
    batch_size=4,
    flop_budget=1e17,
    train_series_count=700,
    eval_series_count=300,
    context_length=None,
):
    """
    Estimate how many training steps fit within the FLOP budget.
    """
    dataset = TrajectoryDataset(data_path)
    avg_token_count = compute_average_input_tokens(
        dataset,
        input_fraction=input_fraction,
        num_series=train_series_count,
        context_length=context_length
    )

    token_count = int(avg_token_count) if avg_token_count > 0 else context_length or 512

    # Get FLOPs for one forward/backward pass (per batch)
    _, step_flops = forwards_pass_flops(token_count, lora_ranks=lora_rank)
    flops_per_training_step = step_flops * batch_size

    # Get evaluation FLOPs (used once after training)
    eval_flops = compute_total_eval_flops(token_count, eval_series_count, lora_rank)

    # Remaining budget for training
    remaining_budget = flop_budget - eval_flops
    max_steps = int(remaining_budget // flops_per_training_step)

    print(f"\n Avg token count: {token_count}")
    print(f"  FLOPs per training step (batch): {flops_per_training_step:.2e}")
    print(f" Evaluation FLOPs: {eval_flops:.2e}")
    print(f" Remaining budget for training: {remaining_budget:.2e}")
    print(f" Max training steps allowed: {max_steps}")

    return max_steps



# Optional CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute FLOPs for LoRA training and evaluation")
    parser.add_argument("--data_path", type=str, default="lotka_volterra_data.h5")
    parser.add_argument("--input_fraction", type=float, default=0.7)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--training_steps", type=int, default=1)
    parser.add_argument("--flop_budget", type=float, default=1e17)
    parser.add_argument("--train_series_count", type=int, default=700)
    parser.add_argument("--eval_series_count", type=int, default=300)
    parser.add_argument("--context_length", type=int, default=None)

    args = parser.parse_args()
    compute_flops(
        data_path=args.data_path,
        input_fraction=args.input_fraction,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        flop_budget=args.flop_budget,
        train_series_count=args.train_series_count,
        eval_series_count=args.eval_series_count,
        context_length=args.context_length,
    )
