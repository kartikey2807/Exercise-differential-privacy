"""Generate the tiny synthetic transaction dataset for the privacy exercise.

The data is entirely invented. It is not copied from AMLSim or any real
financial dataset. The generator is deterministic so that the same command
always produces the same CSV files.

The customer split is six A-primary customers and two B-primary customers
(approximately 70/30). Two A-primary customers also own an account at Bank B,
which lets us test a subject appearing at both banks.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent / "synthetic_data"


# Account ownership is the ground truth used by the generator and later by
# deletion experiments. Account IDs beginning with A belong to Bank A; IDs
# beginning with B belong to Bank B.
ACCOUNT_OWNERS = {
    "A1": "S1",  # S1 owns two accounts at Bank A
    "A2": "S1",
    "A3": "S2",  # S2 is present at both banks
    "B1": "S2",
    "A4": "S3",  # S3 is present at both banks
    "B2": "S3",
    "A5": "S4",
    "A6": "S5",
    "A7": "S6",
    "B3": "S7",
    "B4": "S8",
}


CUSTOMERS = {
    "S1": "A-primary",
    "S2": "A-primary and cross-bank",
    "S3": "A-primary and cross-bank",
    "S4": "A-primary",
    "S5": "A-primary",
    "S6": "A-primary",
    "S7": "B-primary",
    "S8": "B-primary",
}


FIELDNAMES = [
    "transaction_id",
    "sender_account",
    "recipient_account",
    "sender_bank",
    "recipient_bank",
    "timestamp",
    "amount",
    "sender_id",
    "recipient_id",
]


def bank_for_account(account: str) -> str:
    """Return the bank that owns an account."""
    if account.startswith("A"):
        return "Bank_A"
    if account.startswith("B"):
        return "Bank_B"
    raise ValueError(f"Unknown account prefix: {account}")


def build_transactions() -> list[dict[str, object]]:
    """Build exactly 50 deterministic transactions.

    The first ten transactions establish a payment chain and repeated edges.
    The ten-edge pattern is then repeated four times to create repeated
    transactions and activity at both banks.
    """
    # Each tuple is (sender account, recipient account, amount).
    # The first five edges form a chain crossing the two banks:
    # A1 -> A3 -> B2 -> B3 -> A4 -> B1.
    initial_edges = [
        ("A1", "A3", 120.00),
        ("A3", "B2", 90.00),
        ("B2", "B3", 75.00),
        ("B3", "A4", 60.00),
        ("A4", "B1", 55.00),
        ("B1", "A5", 45.00),
        ("A2", "A3", 80.00),
        ("A1", "A3", 120.00),  # repeated transaction edge
        ("A6", "B4", 130.00),
        ("B4", "A7", 95.00),
    ]

    repeated_pattern = [
        ("A1", "A3", 100.00),
        ("A2", "A3", 70.00),
        ("A3", "A4", 65.00),
        ("A4", "B3", 110.00),
        ("B3", "B4", 40.00),
        ("B4", "B2", 35.00),
        ("B2", "B1", 85.00),
        ("B1", "A5", 50.00),
        ("A5", "A6", 45.00),
        ("A6", "B4", 125.00),
    ]

    edges = initial_edges + repeated_pattern * 4
    assert len(edges) == 50

    start = date(2025, 1, 1)
    transactions = []

    for index, (sender, recipient, amount) in enumerate(edges, start=1):
        sender_id = ACCOUNT_OWNERS[sender]
        recipient_id = ACCOUNT_OWNERS[recipient]

        transactions.append(
            {
                "transaction_id": f"T{index:03d}",
                "sender_account": sender,
                "recipient_account": recipient,
                "sender_bank": bank_for_account(sender),
                "recipient_bank": bank_for_account(recipient),
                "timestamp": (start + timedelta(days=index - 1)).isoformat(),
                "amount": f"{amount:.2f}",
                "sender_id": sender_id,
                "recipient_id": recipient_id,
            }
        )

    return transactions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows using the common transaction schema."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_ownership_mapping(path: Path) -> None:
    """Write the synthetic account-to-subject and account-to-bank mapping."""
    fields = ["account", "subject_id", "bank", "customer_group"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for account, subject_id in ACCOUNT_OWNERS.items():
            writer.writerow(
                {
                    "account": account,
                    "subject_id": subject_id,
                    "bank": bank_for_account(account),
                    "customer_group": CUSTOMERS[subject_id],
                }
            )


def main() -> None:
    """Generate the global table, ownership table, and two bank views."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    transactions = build_transactions()

    # Each bank receives the transactions involving one of its accounts.
    # Therefore, a cross-bank transaction appears in both local views, but
    # the views are written as separate files rather than one shared table.
    bank_a_rows = [
        row
        for row in transactions
        if row["sender_bank"] == "Bank_A" or row["recipient_bank"] == "Bank_A"
    ]
    bank_b_rows = [
        row
        for row in transactions
        if row["sender_bank"] == "Bank_B" or row["recipient_bank"] == "Bank_B"
    ]

    write_csv(OUTPUT_DIR / "all_transactions.csv", transactions)
    write_csv(OUTPUT_DIR / "bank_a_transactions.csv", bank_a_rows)
    write_csv(OUTPUT_DIR / "bank_b_transactions.csv", bank_b_rows)
    write_ownership_mapping(OUTPUT_DIR / "ownership_mapping.csv")

    print(f"Generated {len(transactions)} total transactions")
    print(f"Bank A view: {len(bank_a_rows)} transactions")
    print(f"Bank B view: {len(bank_b_rows)} transactions")
    print(f"Accounts: {len(ACCOUNT_OWNERS)}")
    print(f"Customers: {len(CUSTOMERS)} (6 A-primary, 2 B-primary)")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
