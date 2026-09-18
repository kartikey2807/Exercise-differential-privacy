"""Build graph features and deletion variants for the privacy exercise.

The functions in this file operate on rows loaded from:

    synthetic_data/all_transactions.csv
    synthetic_data/bank_A.csv
    synthetic_data/bank_B.csv
    synthetic_data/ownership_map.csv

No real financial data is used. The module includes a small self-test that can
be run with:

    python feature_builder.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


Row = dict[str, str]


REQUIRED_TRANSACTION_FIELDS = {
    "transaction_id",
    "sender_account",
    "recipient_account",
    "sender_bank",
    "recipient_bank",
    "timestamp",
    "amount",
    "sender_id",
    "recipient_id",
}


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "synthetic_data"


def read_csv(path: str | Path) -> list[Row]:
    """Read a CSV file as a list of dictionaries."""
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_transactions(transactions: Iterable[Mapping[str, str]]) -> list[Row]:
    """Copy and validate transaction rows without modifying the input."""
    rows = [dict(row) for row in transactions]
    for index, row in enumerate(rows):
        missing = REQUIRED_TRANSACTION_FIELDS.difference(row)
        if missing:
            raise ValueError(
                f"Transaction row {index} is missing fields: {sorted(missing)}"
            )
        # Fail early on malformed amounts instead of silently treating them as 0.
        float(row["amount"])
    return rows


def _account_bank(account: str) -> str:
    """Infer a bank from the synthetic account naming convention."""
    if account.startswith("A"):
        return "Bank_A"
    if account.startswith("B"):
        return "Bank_B"
    raise ValueError(f"Cannot infer a bank for account {account!r}")


def build_features(
    transactions: Iterable[Mapping[str, str]],
    ownership_map: Iterable[Mapping[str, str]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build node and edge features from a transaction view.

    Parameters
    ----------
    transactions:
        Rows from ``all_transactions.csv``, ``bank_A.csv``, or ``bank_B.csv``.
        A bank view should be passed separately if local bank features are
        required; this function does not merge bank views automatically.
    ownership_map:
        Optional rows from ``ownership_map.csv``. Supplying this includes
        accounts with no transaction in the current view and attaches the
        synthetic subject ID to node features.

    Returns
    -------
    dict
        ``node_features`` contains one row per account with transaction-derived
        statistics. ``edge_features`` contains one row per transaction with
        numeric edge features and metadata.
    """
    rows = _validate_transactions(transactions)

    # Start with accounts from the ownership map so isolated accounts are not
    # accidentally omitted from a bank's graph view.
    accounts: set[str] = set()
    subject_by_account: dict[str, str] = {}
    bank_by_account: dict[str, str] = {}

    if ownership_map is not None:
        for mapping in ownership_map:
            account = mapping["account"]
            accounts.add(account)
            subject_by_account[account] = mapping["subject_id"]
            bank_by_account[account] = mapping["bank"]

    outgoing_count: defaultdict[str, int] = defaultdict(int)
    incoming_count: defaultdict[str, int] = defaultdict(int)
    total_sent: defaultdict[str, float] = defaultdict(float)
    total_received: defaultdict[str, float] = defaultdict(float)

    for row in rows:
        sender = row["sender_account"]
        recipient = row["recipient_account"]
        amount = float(row["amount"])
        accounts.update((sender, recipient))

        outgoing_count[sender] += 1
        incoming_count[recipient] += 1
        total_sent[sender] += amount
        total_received[recipient] += amount

        bank_by_account.setdefault(sender, row["sender_bank"])
        bank_by_account.setdefault(recipient, row["recipient_bank"])
        subject_by_account.setdefault(sender, row["sender_id"])
        subject_by_account.setdefault(recipient, row["recipient_id"])

    node_features: list[dict[str, object]] = []
    for account in sorted(accounts):
        node_features.append(
            {
                "account": account,
                "bank": bank_by_account[account],
                "subject_id": subject_by_account.get(account, ""),
                "outgoing_count": outgoing_count[account],
                "incoming_count": incoming_count[account],
                "total_sent": round(total_sent[account], 2),
                "total_received": round(total_received[account], 2),
            }
        )

    # Edge features retain the transaction identity and include numeric values
    # that a toy model could consume. Metadata is retained for auditability.
    edge_features: list[dict[str, object]] = []
    for row in rows:
        amount = float(row["amount"])
        edge_features.append(
            {
                "transaction_id": row["transaction_id"],
                "source": row["sender_account"],
                "target": row["recipient_account"],
                "amount": amount,
                "timestamp": row["timestamp"],
                "sender_bank": row["sender_bank"],
                "recipient_bank": row["recipient_bank"],
                "sender_id": row["sender_id"],
                "recipient_id": row["recipient_id"],
                "is_cross_bank": row["sender_bank"] != row["recipient_bank"],
            }
        )

    return {"node_features": node_features, "edge_features": edge_features}


def remove_transactions(
    transactions: Iterable[Mapping[str, str]],
    transaction_ids: str | Sequence[str],
) -> list[Row]:
    """Return a copy with the selected transaction IDs removed.

    The original transaction collection is not modified. An error is raised if
    a requested transaction ID is not present, which prevents a silent no-op in
    the privacy comparison.
    """
    rows = _validate_transactions(transactions)
    ids = {transaction_ids} if isinstance(transaction_ids, str) else set(transaction_ids)
    available = {row["transaction_id"] for row in rows}
    missing = ids.difference(available)
    if missing:
        raise KeyError(f"Unknown transaction IDs: {sorted(missing)}")
    return [row for row in rows if row["transaction_id"] not in ids]


def remove_subjects(
    transactions: Iterable[Mapping[str, str]],
    subject_ids: str | Sequence[str],
    ownership_map: Iterable[Mapping[str, str]] | None = None,
) -> list[Row]:
    """Return a copy with subjects, owned accounts, and incident edges removed.

    A transaction is removed if either endpoint belongs to a selected subject.
    The endpoint subject IDs in the transaction table are used directly. The
    ownership map is also checked so the function explicitly handles the
    subject-to-multiple-accounts case.
    """
    rows = _validate_transactions(transactions)
    subjects = {subject_ids} if isinstance(subject_ids, str) else set(subject_ids)

    owned_accounts: set[str] = set()
    if ownership_map is not None:
        for mapping in ownership_map:
            if mapping["subject_id"] in subjects:
                owned_accounts.add(mapping["account"])

    def is_incident(row: Row) -> bool:
        return (
            row["sender_id"] in subjects
            or row["recipient_id"] in subjects
            or row["sender_account"] in owned_accounts
            or row["recipient_account"] in owned_accounts
        )

    return [row for row in rows if not is_incident(row)]


def run_self_tests() -> None:
    """Run deterministic checks against the generated exercise data."""
    all_rows = read_csv(DATA_DIR / "all_transactions.csv")
    bank_a_rows = read_csv(DATA_DIR / "bank_A.csv")
    bank_b_rows = read_csv(DATA_DIR / "bank_B.csv")
    ownership = read_csv(DATA_DIR / "ownership_map.csv")

    assert len(all_rows) == 50
    assert len(bank_a_rows) == 37
    assert len(bank_b_rows) == 31
    assert len(ownership) == 11

    all_features = build_features(all_rows, ownership)
    assert len(all_features["edge_features"]) == 50
    assert len(all_features["node_features"]) == 11

    # Every generated transaction is represented as a directed edge, and the
    # generator includes at least one cross-bank edge.
    assert any(edge["is_cross_bank"] for edge in all_features["edge_features"])

    without_t001 = remove_transactions(all_rows, "T001")
    assert len(without_t001) == 49
    assert len(all_rows) == 50  # input was not mutated

    without_s2 = remove_subjects(all_rows, "S2", ownership)
    assert len(without_s2) < len(all_rows)
    assert all(
        row["sender_id"] != "S2" and row["recipient_id"] != "S2"
        for row in without_s2
    )
    without_s2_features = build_features(without_s2, ownership)
    assert len(without_s2_features["edge_features"]) == len(without_s2)

    # S2 owns A3 and B1, so incident transactions through either account must
    # be removed, not just rows explicitly labelled with S2.
    assert not any(
        row["sender_account"] in {"A3", "B1"}
        or row["recipient_account"] in {"A3", "B1"}
        for row in without_s2
    )

    # The renamed bank files remain independently usable.
    assert build_features(bank_a_rows, ownership)["edge_features"]
    assert build_features(bank_b_rows, ownership)["edge_features"]

    print("All feature-builder self-tests passed")
    print(f"Full graph: {len(all_features['node_features'])} nodes, "
          f"{len(all_features['edge_features'])} edges")
    print(f"After removing T001: {len(without_t001)} edges")
    print(f"After removing S2: {len(without_s2)} edges")


if __name__ == "__main__":
    run_self_tests()
