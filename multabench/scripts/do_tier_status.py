"""
Report MulTaBench-Core / MulTaBench-Full tier composition and Full-tier progress.

Zero network, zero dataset loads: the fast sanity check after editing tiers.py.

Usage:
    python -m multabench.scripts.do_tier_status
    python -m multabench.scripts.do_tier_status --tier full --print-names
"""
from __future__ import annotations

import argparse
from collections import Counter
from multabench.datasets.tiers import (
    FULL_PER_MODALITY,
    MULTABENCH_FULL_IMAGE,
    MULTABENCH_FULL_TEXT,
    Tier,
    datasets_for_tier,
    pending_upload,
    tier_from_name,
    untiered_curated,
)

def _task_type(dataset_id) -> str:
    return dataset_id.name.split("_")[0]


def _print_composition() -> None:
    print("Tier composition")
    print("-" * 64)
    header = f"  {'tier':<6} {'modality':<9} {'n':>4}   " + "  ".join(f"{t:>4}" for t in ("BIN", "MUL", "REG"))
    print(header)
    for tier in Tier:
        for modality in ("image", "text"):
            datasets = datasets_for_tier(tier, modality)
            tasks = Counter(_task_type(d) for d in datasets)
            counts = "  ".join(f"{tasks.get(t, 0):>4}" for t in ("BIN", "MUL", "REG"))
            print(f"  {tier.value:<6} {modality:<9} {len(datasets):>4}   {counts}")
        print(f"  {tier.value:<6} {'total':<9} {len(datasets_for_tier(tier)):>4}")


def _print_progress() -> None:
    print("\nFull-tier progress")
    print("-" * 64)
    print(f"  image   {len(MULTABENCH_FULL_IMAGE):>2}/{FULL_PER_MODALITY}")
    print(f"  text    {len(MULTABENCH_FULL_TEXT):>2}/{FULL_PER_MODALITY}")
    print(f"  total   {len(datasets_for_tier(Tier.FULL)):>2}/{2 * FULL_PER_MODALITY}")

    pending = pending_upload(Tier.FULL)
    print(f"\n  pending upload (load via original source, not the unified API): {len(pending)}")
    for dataset_id in pending:
        print(f"    - {dataset_id.name}  [{type(dataset_id).__name__}]")

    untiered = untiered_curated()
    print(f"\n  uploaded but in no tier: {len(untiered)}")
    for dataset_id in untiered:
        print(f"    ! {dataset_id.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MulTaBench tier composition and Full-tier progress")
    parser.add_argument("--tier", type=str, default=None, choices=[t.value for t in Tier],
                        help="restrict --print-names to this tier (default: both)")
    parser.add_argument("--modality", type=str, default=None, choices=["image", "text"])
    parser.add_argument("--print-names", action="store_true", help="list the dataset names of the tier")
    args = parser.parse_args()

    if args.print_names:
        tiers = [tier_from_name(args.tier)] if args.tier else list(Tier)
        for tier in tiers:
            for dataset_id in datasets_for_tier(tier, args.modality):
                print(dataset_id.name)
        return

    _print_composition()
    _print_progress()


if __name__ == "__main__":
    main()
