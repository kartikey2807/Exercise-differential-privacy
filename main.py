"""Run one privacy comparison at a time.

Transaction-level example:

    source venv/Scripts/activate && python main.py --mode transaction --transaction-id T001

Subject-level stress-test example:

    source venv/Scripts/activate && python main.py --mode subject --subject-id S2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from feature_builder import read_csv, remove_subjects, remove_transactions
from privacy_check import compare_neighboring_datasets


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "synthetic_data"


def parse_args() -> argparse.Namespace:
    """Parse the neighboring-dataset scenario to run."""
    parser = argparse.ArgumentParser(
        description="Run one transaction-level or subject-level comparison."
    )
    parser.add_argument(
        "--mode",
        choices=("transaction", "subject"),
        default="transaction",
        help="Comparison type (default: transaction).",
    )
    parser.add_argument(
        "--transaction-id",
        default="T001",
        help="Transaction to remove in transaction mode (default: T001).",
    )
    parser.add_argument(
        "--subject-id",
        default="S2",
        help="Subject to remove in subject mode (default: S2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_transactions = read_csv(DATA_DIR / "all_transactions.csv")
    ownership_map = read_csv(DATA_DIR / "ownership_map.csv")

    if args.mode == "transaction":
        neighboring_transactions = remove_transactions(
            all_transactions,
            args.transaction_id,
        )
    else:
        neighboring_transactions = remove_subjects(
            all_transactions,
            args.subject_id,
            ownership_map,
        )

    result = compare_neighboring_datasets(
        all_transactions=all_transactions,
        neighboring_transactions=neighboring_transactions,
        ownership_map=ownership_map,
        neighborhood=args.mode,
        clip_percentile=90.0,
        epsilon=1.0,
        delta=0.01,
        seed=2025,
    )
    comparison = cast(dict[str, object], result["comparison"])

    selected_c = cast(float, comparison["max_grad_norm"])
    measured_change = cast(float, comparison["measured_change"])
    reference_bound = cast(
        float, comparison["transaction_level_reference_bound"]
    )
    bound_applies = cast(bool, comparison["bound_applies"])
    epsilon = cast(float, comparison["epsilon"])
    delta = cast(float, comparison["delta"])
    noise_multiplier = cast(float, comparison["noise_multiplier"])
    noise_std = cast(float, comparison["noise_standard_deviation"])

    print("Privacy sensitivity check")
    print("=" * 26)
    print(f"Mode: {args.mode}")
    print(f"Baseline transactions: {len(all_transactions)}")
    print(f"Transactions removed: {comparison['removed_count']}")
    print(f"Clipping bound C (90th percentile): {selected_c:.10f}")
    print(f"Fixed denominator B: {comparison['denominator']}")
    print(f"Target epsilon: {epsilon:.10f}")
    print(f"Target delta: {delta:.10f}")
    print(f"Gaussian noise multiplier sigma: {noise_multiplier:.10f}")
    print(f"Average-update noise std: {noise_std:.10f}")
    print()

    print("Result")
    print("------")
    print(f"Measured pre-noise gradient difference: {measured_change:.10f}")

    if bound_applies:
        expected_bound = cast(float, comparison["expected_bound"])
        print(f"Expected transaction bound C/B:         {expected_bound:.10f}")
        print(f"Check passed: {comparison['bound_check_passed']}")
    else:
        print(f"Transaction-level reference bound C/B:  {reference_bound:.10f}")
        print(
            "Transaction bound holds in this subject test: "
            f"{measured_change <= reference_bound + 1e-12}"
        )
        print("Subject-level privacy bound claimed: False")

    print()
    print("Interpretation")
    print("--------------")
    if args.mode == "transaction":
        print(
            "This is the formal transaction-level add/remove sensitivity check."
        )
    else:
        print(
            "This is a subject-removal stress test. The mechanism remains "
            "transaction-level, so no subject-level privacy bound is claimed."
        )
    print(
        "Noise is constructed for the released update, while the reported "
        "difference is a deterministic pre-noise measurement."
    )


if __name__ == "__main__":
    main()
