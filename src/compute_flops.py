import argparse
from generate_predictions import TrajectoryDataset, PredictionPipeline
from flops_model import forwards_pass_flops

def count_input_tokens(pipeline, series_index, input_fraction=0.75):
    """
    Return the number of tokens used for the given time series index.
    """
    series = pipeline.dataset.trajectories[series_index]
    input_timesteps = int(input_fraction * 100)
    input_str = pipeline.trunc_string(series, n=input_timesteps)
    tokens = pipeline.tokenizer(input_str, return_tensors='pt')
    return tokens["input_ids"].shape[1]  # sequence length


def print_flop_summary(forward_flops, training_flops):
    """
    Print a summary of FLOP breakdown for forward pass and training.
    """
    labels = ["Additions", "Multiplications", "Divisions", "Exponentiations", "Square Roots"]
    print("\n📊 FLOPs Breakdown (Forward Pass):")
    for label, val in zip(labels, forward_flops):
        print(f"{label:<18}: {val:.2e}")

    print(f"\n🧮 Total Forward FLOPs: {forward_flops.sum():.2e}")
    print(f"🧮 Estimated Training FLOPs (3x forward): {training_flops:.2e}")


def main(args):
    # Load dataset and prediction pipeline
    dataset = TrajectoryDataset(args.data_path)
    pipeline = PredictionPipeline(dataset, test_fraction=args.test_fraction)

    # Count tokens for selected trajectory
    no_tokens = count_input_tokens(pipeline, series_index=args.series_index, input_fraction=args.test_fraction)

    print(f"\n🔍 Evaluating Series Index: {args.series_index}")
    print(f"📏 Token Count: {no_tokens}")
    print(f"🧠 LoRA Rank: {args.lora_rank}")

    # Compute FLOPs
    forward_flops, training_flops = forwards_pass_flops(no_tokens, lora_ranks=args.lora_rank, print_summary=False)

    # Display results
    print_flop_summary(forward_flops, training_flops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FLOPs used in generate_predictions.py")
    parser.add_argument("--data_path", type=str, default="lotka_volterra_data.h5", help="Path to your HDF5 dataset")
    parser.add_argument("--series_index", type=int, default=0, help="Which series to compute FLOPs for")
    parser.add_argument("--test_fraction", type=float, default=0.75, help="Fraction of sequence used as input")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank (set to 0 if not used)")

    args = parser.parse_args()
    main(args)
