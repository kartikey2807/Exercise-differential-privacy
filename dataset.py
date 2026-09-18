"""PyTorch Dataset for the synthetic transaction graph.

Each transaction is represented as one edge-level example. Its input vector is
constructed by concatenating:

    sender node features      (4 values)
    recipient node features   (4 values)
    edge features             (2 values)

The resulting input dimension is 10:

    [normalized sender outgoing count, normalized sender incoming count,
     normalized sender total sent, normalized sender total received,
     normalized recipient outgoing count, normalized recipient incoming count,
     normalized recipient total sent, normalized recipient total received,
     normalized amount, cross-bank indicator]

The labels are deliberately synthetic. They are only present to make the
examples usable with a classification loss; they do not represent real AML
labels or establish detection quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, cast

import torch
from torch.utils.data import Dataset

from feature_builder import build_features, read_csv


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "synthetic_data"


class TransactionDataset(Dataset):
    """Return one numeric feature vector and one toy class label per transaction."""

    def __init__(
        self,
        transactions: Iterable[Mapping[str, str]],
        ownership_map: Iterable[Mapping[str, str]] | None = None,
        amount_scale: float = 200.0,
        count_scale: float = 50.0,
        total_amount_scale: float = 5000.0,
    ) -> None:
        if amount_scale <= 0:
            raise ValueError("amount_scale must be positive")
        if count_scale <= 0:
            raise ValueError("count_scale must be positive")
        if total_amount_scale <= 0:
            raise ValueError("total_amount_scale must be positive")

        self.transactions = [dict(row) for row in transactions]
        # These scales are fixed across all dataset variants. They are based on
        # declared bounds for this tiny synthetic generator, not recomputed from
        # the current subset after a transaction or subject is removed.
        self.amount_scale = amount_scale
        self.count_scale = count_scale
        self.total_amount_scale = total_amount_scale
        self.ownership_map = (
            [dict(row) for row in ownership_map]
            if ownership_map is not None
            else None
        )

        graph_features = build_features(self.transactions, self.ownership_map)
        node_by_account = {
            row["account"]: row for row in graph_features["node_features"]
        }
        edge_by_id = {
            row["transaction_id"]: row for row in graph_features["edge_features"]
        }

        if not self.transactions:
            raise ValueError("TransactionDataset requires at least one transaction")

        # Keep the scale fixed across full and deleted datasets. If this were
        # recomputed per subset, deleting one high-value transaction would
        # change every normalized amount and confound the privacy comparison.
        self.features: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

        for transaction in self.transactions:
            sender = node_by_account[transaction["sender_account"]]
            recipient = node_by_account[transaction["recipient_account"]]
            edge = edge_by_id[transaction["transaction_id"]]

            normalized_amount = (
                float(cast(int | float, edge["amount"])) / self.amount_scale
            )
            cross_bank = float(cast(bool, edge["is_cross_bank"]))

            vector = [
                # Sender node features.
                float(cast(int | float, sender["outgoing_count"]))
                / self.count_scale,
                float(cast(int | float, sender["incoming_count"]))
                / self.count_scale,
                float(cast(int | float, sender["total_sent"]))
                / self.total_amount_scale,
                float(cast(int | float, sender["total_received"]))
                / self.total_amount_scale,
                # Recipient node features.
                float(cast(int | float, recipient["outgoing_count"]))
                / self.count_scale,
                float(cast(int | float, recipient["incoming_count"]))
                / self.count_scale,
                float(cast(int | float, recipient["total_sent"]))
                / self.total_amount_scale,
                float(cast(int | float, recipient["total_received"]))
                / self.total_amount_scale,
                # Numeric edge features.
                normalized_amount,
                cross_bank,
            ]

            self.features.append(torch.tensor(vector, dtype=torch.float32))

            # Invented toy label: mark high-value or cross-bank transactions as
            # class 1. This is not a real fraud label and is only for providing
            # a target to CrossEntropyLoss during the exercise.
            label = int(
                float(cast(int | float, edge["amount"])) >= 100.0
                or cast(bool, edge["is_cross_bank"])
            )
            self.labels.append(torch.tensor(label, dtype=torch.long))

    @property
    def input_dim(self) -> int:
        """Number of features returned for one transaction."""
        return self.features[0].numel()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def load_transaction_dataset(
    transaction_path: str | Path = DATA_DIR / "all_transactions.csv",
    ownership_path: str | Path = DATA_DIR / "ownership_map.csv",
    amount_scale: float = 200.0,
    count_scale: float = 50.0,
    total_amount_scale: float = 5000.0,
) -> TransactionDataset:
    """Load the renamed CSV files and construct a TransactionDataset."""
    transactions = read_csv(transaction_path)
    ownership_map = read_csv(ownership_path)
    return TransactionDataset(
        transactions,
        ownership_map,
        amount_scale=amount_scale,
        count_scale=count_scale,
        total_amount_scale=total_amount_scale,
    )


def run_self_test() -> None:
    """Verify the dataset can feed a two-layer PyTorch classifier."""
    dataset = load_transaction_dataset()
    assert len(dataset) == 50
    assert dataset.input_dim == 10

    features, label = dataset[0]
    assert features.shape == (10,)
    assert label.shape == torch.Size([])
    assert label.dtype == torch.long

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    batch_features, batch_labels = next(iter(loader))
    assert batch_features.shape == (32, 10)
    assert batch_labels.shape == (32,)

    # This is the expected interface for a two-layer vanilla classifier.
    model = torch.nn.Sequential(
        torch.nn.Linear(dataset.input_dim, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    logits = model(batch_features)
    assert logits.shape == (32, 2)

    print("Dataset self-test passed")
    print(f"Examples: {len(dataset)}")
    print(f"One feature vector: {features.shape}")
    print(f"Batch features: {batch_features.shape}")
    print(f"Batch labels: {batch_labels.shape}")
    print(f"Model logits: {logits.shape}")


if __name__ == "__main__":
    run_self_test()
