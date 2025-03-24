import argparse
from generate_predictions import TrajectoryDataset, PredictionPipeline
from flops_model import forwards_pass_flops

FLOP_BUDGET = 1e17
BATCH_SIZE = 4


def count_input_tokens(pipeline, series_index):
    """
    Return the number of tokens used for the given time series index.
    """
    series = pipeline.dataset.trajectories[series_index]
    input_timesteps = int(pipeline.input_fraction * 100)
    input_str = pipeline.trunc_string(series, n=input_timesteps)
    tokens = pipeline.tokenizer(input_str, return_tensors='pt')
    return tokens["input_ids"].shape[1]  # sequence length


def print_flop_summary(forward_flops, training_flops, steps):
    """
    Print a summary of FLOP breakdown for forward pass, training, and max steps.
    """
    labels = ["Additions", "Multiplications", "Divisions", "Exponentiations", "Square Roots"]
    print("\n FLOPs Breakdown (Forward Pass):")
    for label, val in zip(labels, forward_flops):
        print(f"{label:<18}: {val:.2e}")

    print(f"\n Total Forward FLOPs: {forward_flops.sum():.2e}")
    print(f" Estimated Training FLOPs (3x forward): {training_flops:.2e}")
    print(f"\n Max Training Steps with batch size {BATCH_SIZE} and budget {FLOP_BUDGET:.1e}: {steps:,}")


def main(args):
    # Load dataset and prediction pipeline
    dataset = TrajectoryDataset(args.data_path)
    pipeline = PredictionPipeline(dataset, input_fraction=args.test_fraction)

    # Count input tokens
    no_tokens = count_input_tokens(pipeline, series_index=args.series_index)

    print(f"\n Evaluating Series Index: {args.series_index}")
    print(f" Token Count: {no_tokens}")
    print(f" LoRA Rank: {args.lora_rank}")

    # Compute FLOPs
    forward_flops, training_flops = forwards_pass_flops(no_tokens, lora_ranks=args.lora_rank, print_summary=False)

    # Multiply by batch size
    flops_per_step = training_flops * BATCH_SIZE
    max_steps = int(FLOP_BUDGET // flops_per_step)

    # Show results
    print_flop_summary(forward_flops, training_flops, max_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FLOPs used in generate_predictions.py")
    parser.add_argument("--data_path", type=str, default="lotka_volterra_data.h5", help="Path to your HDF5 dataset")
    parser.add_argument("--series_index", type=int, default=0, help="Which series to compute FLOPs for")
    parser.add_argument("--test_fraction", type=float, default=0.5, help="Fraction of sequence used as input")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank (set to 0 if not used)")

    args = parser.parse_args()
    main(args)
