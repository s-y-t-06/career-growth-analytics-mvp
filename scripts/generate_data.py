"""CLI script to generate the synthetic MVP dataset."""

import argparse

from career_growth import config
from career_growth.data_generation.generator import generate_all_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for the AI Career Growth Analytics MVP."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=config.DEFAULT_USER_COUNT,
        help="Number of synthetic users to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to write generated data.",
    )
    args = parser.parse_args()

    data = generate_all_data(count=args.count, seed=args.seed, output_dir=args.output_dir)
    print(f"Generated {len(data['users'])} users and {len(data['events'])} events.")
    print(f"Churn rate: {data['labels']['is_churned'].mean():.2%}")


if __name__ == "__main__":
    main()
