"""
Report MulTaBench-Core / MulTaBench-Full tier composition and Full-tier progress.

Zero network, zero dataset loads -- this is the fast sanity check to run after any edit to
multabench/datasets/tiers.py. Deliberately does NOT import curation_mapping (that pulls in all
~70 modules under datasets/annotated/ and can raise); annotated modules are listed by filename.

Usage:
    python -m multabench.scripts.do_tier_status
    python -m multabench.scripts.do_tier_status --tier full --print-names
    python -m multabench.scripts.do_tier_status --candidates
"""
from __future__ import annotations

import argparse
from collections import Counter
from glob import glob
from os.path import basename, dirname, exists, join
from pkgutil import iter_modules

from multabench.datasets import annotated
from multabench.datasets.all_datasets import ALL_DATASETS, MulTaBenchDatasetID, is_image_dataset
from multabench.datasets.image_benchmarks import IMAGE_BENCHMARK_CANDIDATES
from multabench.datasets.text_benchmarks import ACCEPTED_TEXT_DATASETS, REJECTED_TEXT_DATASETS
from multabench.datasets.tiers import (
    FULL_PER_MODALITY,
    MULTABENCH_FULL_IMAGE,
    MULTABENCH_FULL_TEXT,
    Tier,
    datasets_for_tier,
    get_tier,
    is_curated,
    pending_upload,
    tier_from_name,
    untiered_curated,
)

_RESULTS = join(dirname(__file__), "..", "leaderboard", "results")
_TEXT_SOURCE_DIR = join(_RESULTS, "text_source")
_CORPUS_DIR = join(_RESULTS, "tabstar_corpus")


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


def _has_evidence(dataset_id) -> str:
    """Which committed result files mention this dataset (report-only, substring match)."""
    found = []
    if exists(join(_TEXT_SOURCE_DIR, f"{dataset_id.name}.csv")):
        found.append("text_source")
    for path in glob(join(_CORPUS_DIR, "*.csv")):
        with open(path) as f:
            if dataset_id.name in f.read():
                found.append(f"tabstar_corpus/{basename(path)}")
    return ",".join(found) if found else "-"


def _print_candidates() -> None:
    annotated_modules = {name for _, name, _ in iter_modules(annotated.__path__)}

    text_pool = [d for d in dict.fromkeys(ACCEPTED_TEXT_DATASETS + REJECTED_TEXT_DATASETS) if get_tier(d) is None]
    print(f"\nText-tabular candidates outside any tier: {len(text_pool)}")
    print("-" * 64)
    for dataset_id in text_pool:
        curation = "curated" if dataset_id.name in annotated_modules else "NO CURATION MODULE"
        print(f"  {dataset_id.name:<52} {curation:<19} {_has_evidence(dataset_id)}")

    image_pool = [d for d in ALL_DATASETS if is_image_dataset(d) and get_tier(d) is None and not is_curated(d)]
    print(f"\nImage-tabular candidates outside any tier: {len(image_pool)} "
          f"({len(IMAGE_BENCHMARK_CANDIDATES)} of them shortlisted in IMAGE_BENCHMARK_CANDIDATES)")
    print("-" * 64)
    shortlisted = set(IMAGE_BENCHMARK_CANDIDATES)
    for dataset_id in image_pool:
        curation = "curated" if dataset_id.name in annotated_modules else "NO CURATION MODULE"
        mark = "*" if dataset_id in shortlisted else " "
        print(f"  {mark} {dataset_id.name:<50} {curation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MulTaBench tier composition and Full-tier progress")
    parser.add_argument("--tier", type=str, default=None, choices=[t.value for t in Tier],
                        help="restrict --print-names to this tier (default: both)")
    parser.add_argument("--modality", type=str, default=None, choices=["image", "text"])
    parser.add_argument("--print-names", action="store_true", help="list the dataset names of the tier")
    parser.add_argument("--candidates", action="store_true", help="list untiered candidates and their evidence")
    args = parser.parse_args()

    if args.print_names:
        tiers = [tier_from_name(args.tier)] if args.tier else list(Tier)
        for tier in tiers:
            for dataset_id in datasets_for_tier(tier, args.modality):
                print(dataset_id.name)
        return

    _print_composition()
    _print_progress()
    if args.candidates:
        _print_candidates()


if __name__ == "__main__":
    main()
