"""Transaction-level DP-SGD sensitivity check.

The mechanism in this file is defined for transaction-level add/remove
adjacency. A second call can use a subject-removed dataset as a stress test,
but that call does not change the mechanism or create a subject-level privacy
bound.

The fixed-denominator average is:

    aggregate(D) = sum_i clip(g_i, C) / B

where B is the number of transactions in the baseline dataset. For target
epsilon and delta, the classic Gaussian mechanism calibration used here is:

    sigma = sqrt(2*log(1.25/delta)) / epsilon
    noisy_aggregate = aggregate(D) + Normal(0, sigma*C/B)

The comparison itself uses pre-noise aggregates because the clipping bound is a
deterministic sensitivity check. The noisy aggregate is still constructed to
represent the released update. This is a one-release Gaussian calibration;
repeated releases require composition accounting.

By default, C is selected as the 90th percentile of baseline unclipped
per-transaction gradient norms. This is empirical calibration for the exercise;
selecting C from private data would need separate privacy treatment in a formal
deployment.

Run the built-in test with:

    source venv/Scripts/activate && python privacy_check.py
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn

from dataset import TransactionDataset
from feature_builder import read_csv, remove_subjects, remove_transactions


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "synthetic_data"
TransactionRow = Mapping[str, str]
Neighborhood = Literal["transaction", "subject"]


class TwoTensorModel(nn.Module):
    """A model with exactly two parameter tensors: weight and bias."""

    def __init__(self, input_dim: int, output_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


@dataclass
class GradientAggregate:
    """The deterministic and noisy results for one dataset variant."""

    example_count: int
    denominator: int
    unclipped_norms: list[float]
    pre_noise_gradient: torch.Tensor
    noisy_gradient: torch.Tensor


@dataclass
class NeighborComparison:
    """Result of comparing one dataset with one neighboring variant."""

    neighborhood: Neighborhood
    removed_count: int
    measured_change: float
    expected_bound: float | None
    transaction_level_reference_bound: float
    bound_applies: bool
    bound_check_passed: bool
    max_grad_norm: float
    clip_percentile: float
    epsilon: float
    delta: float
    noise_multiplier: float
    noise_standard_deviation: float
    denominator: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON/report-friendly scalar values."""
        return asdict(self)


def _flatten_parameters(parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten all parameter tensors into one vector."""
    return torch.cat([parameter.reshape(-1) for parameter in parameters])


def _clip_gradient(
    gradients: Sequence[torch.Tensor], max_grad_norm: float
) -> tuple[list[torch.Tensor], float]:
    """Globally clip all parameter tensors belonging to one example."""
    flat = _flatten_parameters(gradients)
    norm = torch.linalg.vector_norm(flat, ord=2)
    scale = min(1.0, max_grad_norm / (float(norm) + 1e-12))
    clipped = [gradient * scale for gradient in gradients]
    return clipped, float(norm)


def _make_dataset(
    transactions: Iterable[TransactionRow],
    ownership_map: Iterable[Mapping[str, str]],
) -> TransactionDataset:
    """Create a dataset with fixed feature scaling for every variant."""
    return TransactionDataset(
        transactions,
        ownership_map,
        amount_scale=200.0,
        count_scale=50.0,
        total_amount_scale=5000.0,
    )


def compute_unclipped_gradient_norms(
    model: nn.Module,
    transactions: Iterable[TransactionRow],
    ownership_map: Iterable[Mapping[str, str]],
) -> list[float]:
    """Measure one unclipped global gradient norm per transaction."""
    dataset = _make_dataset(transactions, ownership_map)
    criterion = nn.CrossEntropyLoss()
    parameters = list(model.parameters())
    norms: list[float] = []

    model.train()
    for index in range(len(dataset)):
        # Gradients accumulate by default. Resetting here ensures that this is
        # the gradient of exactly one transaction.
        model.zero_grad(set_to_none=True)
        features, label = dataset[index]
        loss = criterion(model(features.unsqueeze(0)), label.unsqueeze(0))
        loss.backward()
        gradients = [
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
            for parameter in parameters
        ]
        norms.append(float(torch.linalg.vector_norm(_flatten_parameters(gradients))))

    return norms


def select_clipping_constant(
    model: nn.Module,
    transactions: Iterable[TransactionRow],
    ownership_map: Iterable[Mapping[str, str]],
    *,
    percentile: float = 90.0,
) -> tuple[float, list[float]]:
    """Select C as a percentile of baseline unclipped gradient norms."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be greater than 0 and at most 100")

    norms = compute_unclipped_gradient_norms(model, transactions, ownership_map)
    if not norms:
        raise ValueError("At least one transaction is required to select C")

    norm_tensor = torch.tensor(norms, dtype=torch.float32)
    clipping_constant = float(
        torch.quantile(norm_tensor, percentile / 100.0, interpolation="linear")
    )
    # Avoid a zero clipping bound for degenerate zero-gradient data.
    return max(clipping_constant, 1e-12), norms


def compute_gradient_aggregate(
    model: nn.Module,
    transactions: Iterable[TransactionRow],
    ownership_map: Iterable[Mapping[str, str]],
    *,
    denominator: int,
    max_grad_norm: float,
    noise_multiplier: float,
    seed: int | None = None,
) -> GradientAggregate:
    """Compute one fixed-denominator DP-SGD-like aggregate.

    Each transaction is processed separately, clipped, and then aggregated.
    The model parameters are not updated. ``denominator`` is the baseline B and
    stays fixed for a deleted variant.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier cannot be negative")

    dataset = _make_dataset(transactions, ownership_map)
    if len(dataset) > denominator:
        raise ValueError(
            "The neighboring dataset cannot contain more examples than the "
            "baseline denominator"
        )

    criterion = nn.CrossEntropyLoss()
    parameters = list(model.parameters())
    aggregate = torch.zeros(
        sum(parameter.numel() for parameter in parameters),
        dtype=torch.float32,
    )
    unclipped_norms: list[float] = []

    model.train()
    for index in range(len(dataset)):
        # Reset before every example so gradients do not accumulate across
        # transactions before clipping.
        model.zero_grad(set_to_none=True)
        features, label = dataset[index]
        loss = criterion(model(features.unsqueeze(0)), label.unsqueeze(0))
        loss.backward()

        gradients = [
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
            for parameter in parameters
        ]
        clipped, unclipped_norm = _clip_gradient(gradients, max_grad_norm)
        unclipped_norms.append(unclipped_norm)
        aggregate += _flatten_parameters(clipped)

    # Deleted examples contribute zero to the fixed-denominator average.
    pre_noise = aggregate / denominator

    if seed is not None:
        generator = torch.Generator(device=pre_noise.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            pre_noise.shape,
            generator=generator,
            dtype=pre_noise.dtype,
            device=pre_noise.device,
        )
    else:
        noise = torch.randn_like(pre_noise)

    # This is equivalent to adding N(0, (sigma*C)^2) to the sum and dividing
    # the result by B. The final average noise has std sigma*C/B.
    noisy = pre_noise + noise * (noise_multiplier * max_grad_norm / denominator)

    return GradientAggregate(
        example_count=len(dataset),
        denominator=denominator,
        unclipped_norms=unclipped_norms,
        pre_noise_gradient=pre_noise,
        noisy_gradient=noisy,
    )


def compare_neighboring_datasets(
    all_transactions: Iterable[TransactionRow],
    neighboring_transactions: Iterable[TransactionRow],
    ownership_map: Iterable[Mapping[str, str]],
    *,
    neighborhood: Neighborhood = "transaction",
    max_grad_norm: float | None = None,
    clip_percentile: float = 90.0,
    epsilon: float = 1.0,
    delta: float = 0.01,
    seed: int = 2025,
) -> dict[str, object]:
    """Compare exactly one baseline dataset with one neighboring variant.

    ``neighborhood="transaction"`` means the neighboring input must differ by
    one removed transaction. In that case, the expected bound is C/B.

    ``neighborhood="subject"`` is a stress test using caller-supplied rows
    after removing one subject, their accounts, and incident transactions. The
    mechanism remains transaction-level. Therefore, no subject-level bound is
    returned or claimed; C/B is reported only as a transaction-level reference.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be between 0 and 1")

    # For the fixed-denominator average, Delta_2 = C/B. The classic Gaussian
    # mechanism sufficient bound uses noise std Delta_2 * sqrt(2 log(1.25/delta))
    # / epsilon. Since this implementation samples noise as sigma*C/B, sigma is:
    # sqrt(2 log(1.25/delta)) / epsilon.
    noise_multiplier = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    full_rows = [dict(row) for row in all_transactions]
    neighboring_rows = [dict(row) for row in neighboring_transactions]
    ownership_rows = [dict(row) for row in ownership_map]

    denominator = len(full_rows)
    removed_count = denominator - len(neighboring_rows)
    if denominator == 0:
        raise ValueError("all_transactions must contain at least one row")
    if removed_count <= 0:
        raise ValueError("neighboring_transactions must remove at least one row")
    if neighborhood == "transaction" and removed_count != 1:
        raise ValueError(
            "Transaction-level comparison must remove exactly one transaction"
        )

    # Recreate the same model state for every independent call. The model is
    # never updated, so the only changing input is the dataset variant.
    reference_dataset = _make_dataset(full_rows, ownership_rows)
    torch.manual_seed(seed)
    model = TwoTensorModel(reference_dataset.input_dim)

    if max_grad_norm is None:
        max_grad_norm, baseline_norms = select_clipping_constant(
            model,
            full_rows,
            ownership_rows,
            percentile=clip_percentile,
        )
    else:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        baseline_norms = compute_unclipped_gradient_norms(
            model,
            full_rows,
            ownership_rows,
        )

    full = compute_gradient_aggregate(
        model,
        full_rows,
        ownership_rows,
        denominator=denominator,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        seed=seed,
    )
    neighboring = compute_gradient_aggregate(
        model,
        neighboring_rows,
        ownership_rows,
        denominator=denominator,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        seed=seed + 1,
    )

    measured_change = float(
        torch.linalg.vector_norm(
            full.pre_noise_gradient - neighboring.pre_noise_gradient
        )
    )
    transaction_reference_bound = max_grad_norm / denominator
    bound_applies = neighborhood == "transaction"
    expected_bound = transaction_reference_bound if bound_applies else None
    bound_check_passed = (
        measured_change <= transaction_reference_bound + 1e-12
        if bound_applies
        else False
    )

    comparison = NeighborComparison(
        neighborhood=neighborhood,
        removed_count=removed_count,
        measured_change=measured_change,
        expected_bound=expected_bound,
        transaction_level_reference_bound=transaction_reference_bound,
        bound_applies=bound_applies,
        bound_check_passed=bound_check_passed,
        max_grad_norm=max_grad_norm,
        clip_percentile=clip_percentile,
        epsilon=epsilon,
        delta=delta,
        noise_multiplier=noise_multiplier,
        noise_standard_deviation=noise_multiplier * max_grad_norm / denominator,
        denominator=denominator,
    )

    return {
        "comparison": comparison.as_dict(),
        "full": full,
        "neighboring": neighboring,
        "baseline_unclipped_norms": baseline_norms,
        "model_parameter_tensor_count": len(list(model.parameters())),
    }


def run_self_test() -> None:
    """Run independent transaction and subject comparisons."""
    all_rows = read_csv(DATA_DIR / "all_transactions.csv")
    ownership = read_csv(DATA_DIR / "ownership_map.csv")
    transaction_rows = remove_transactions(all_rows, "T001")
    subject_rows = remove_subjects(all_rows, "S2", ownership)

    transaction_result = compare_neighboring_datasets(
        all_rows,
        transaction_rows,
        ownership,
        neighborhood="transaction",
    )
    transaction_comparison = cast(
        dict[str, object], transaction_result["comparison"]
    )
    selected_c = cast(float, transaction_comparison["max_grad_norm"])

    subject_result = compare_neighboring_datasets(
        all_rows,
        subject_rows,
        ownership,
        neighborhood="subject",
        max_grad_norm=selected_c,
    )
    subject_comparison = cast(dict[str, object], subject_result["comparison"])

    assert transaction_result["model_parameter_tensor_count"] == 2
    assert transaction_comparison["neighborhood"] == "transaction"
    assert transaction_comparison["removed_count"] == 1
    assert transaction_comparison["bound_applies"] is True
    assert transaction_comparison["expected_bound"] == selected_c / 50.0
    assert transaction_comparison["bound_check_passed"] is True

    assert subject_comparison["neighborhood"] == "subject"
    assert subject_comparison["removed_count"] == 26
    assert subject_comparison["bound_applies"] is False
    assert subject_comparison["expected_bound"] is None
    assert subject_comparison["transaction_level_reference_bound"] == selected_c / 50.0
    assert transaction_comparison["epsilon"] == 1.0
    assert transaction_comparison["delta"] == 0.01
    expected_sigma = math.sqrt(2.0 * math.log(1.25 / 0.01))
    assert abs(
        cast(float, transaction_comparison["noise_multiplier"]) - expected_sigma
    ) < 1e-12

    full = cast(GradientAggregate, transaction_result["full"])
    assert len(full.unclipped_norms) == 50
    assert len(cast(list[float], transaction_result["baseline_unclipped_norms"])) == 50
    assert full.pre_noise_gradient.shape == (22,)
    assert full.noisy_gradient.shape == (22,)

    print("Privacy-check self-test passed")
    print(f"Selected C (90th percentile): {selected_c}")
    print(f"Transaction comparison: {transaction_comparison}")
    print(f"Subject comparison: {subject_comparison}")
    print(
        "Note: the subject comparison has no subject-level privacy bound; "
        "it only tests the transaction-level reference bound."
    )


if __name__ == "__main__":
    run_self_test()
