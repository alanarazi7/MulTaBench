"""
Audit every MulTaBench dataset for multimodal feature counts.

Downloads each dataset, detects image and text (free-text) columns after curation,
and prints a per-dataset summary.  Datasets with more than MAX_MULTIMODAL features
are flagged ❌ (red); those within the limit get ✅ (green).

Usage:
    python do_multabench_audit.py                  # every uploaded MulTaBench dataset
    python do_multabench_audit.py --tier core      # the MulTaBench-Core 40
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.tiers import Tier, datasets_for_tier, is_curated, pending_upload
from multabench.benchmark.load import load_multabench_dataset
from multabench.preprocessing.feat_types import detect_text_features
from multabench.baselines.preprocessing.feature_types import detect_image_features

MAX_MULTIMODAL = 5


@dataclass
class DatasetAudit:
    dataset_id: MulTaBenchDatasetID
    image_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)
    error: str | None = None

    @property
    def n_total(self) -> int:
        return len(self.image_cols) + len(self.text_cols)

    @property
    def ok(self) -> bool:
        return self.error is None and self.n_total <= MAX_MULTIMODAL


def audit_dataset(dataset_id: MulTaBenchDatasetID) -> DatasetAudit:
    try:
        dataset = load_multabench_dataset(dataset_id)
    except Exception as e:
        return DatasetAudit(dataset_id=dataset_id, error=str(e))

    x = dataset.x
    image_cols = detect_image_features(x)
    text_cols = detect_text_features(x, exclude_columns=set(image_cols))
    return DatasetAudit(dataset_id=dataset_id, image_cols=image_cols, text_cols=text_cols)


def print_audit(audit: DatasetAudit) -> None:
    name = audit.dataset_id.name

    if audit.error is not None:
        print(f"  ⚠️  {name}  —  FAILED: {audit.error}")
        return

    status = "✅" if audit.ok else "❌"
    print(f"\n{status}  {name}  —  {audit.n_total} multimodal features  "
          f"({len(audit.image_cols)} image, {len(audit.text_cols)} text)")
    if audit.image_cols:
        print(f"   🖼️  image : {audit.image_cols}")
    if audit.text_cols:
        print(f"   📝 text  : {audit.text_cols}")


def print_summary(audits: list[DatasetAudit]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY  (sorted by feature count descending)")
    print("=" * 80)
    successful = [a for a in audits if a.error is None]
    failed = [a for a in audits if a.error is not None]
    for a in sorted(successful, key=lambda a: a.n_total, reverse=True):
        status = "✅" if a.ok else "❌"
        img = f"  🖼️ ×{len(a.image_cols)}" if a.image_cols else ""
        txt = f"  📝 ×{len(a.text_cols)}" if a.text_cols else ""
        print(f"  {status}  {a.n_total:2d}  {a.dataset_id.name}{img}{txt}")
    for a in failed:
        print(f"  ⚠️       {a.dataset_id.name}  —  FAILED")
    print(f"\n  Passed (≤{MAX_MULTIMODAL}): {sum(1 for a in successful if a.ok)}/{len(successful)}")


def _resolve_ids(tier: str) -> List[MulTaBenchDatasetID]:
    """Uploaded datasets only -- audit_dataset() loads through the unified MulTaBench API."""
    if tier == "uploaded":
        return list(MulTaBenchDatasetID)
    skipped = [d.name for d in pending_upload(tier)]
    if skipped:
        print(f"Skipping {len(skipped)} MulTaBench-{tier} datasets that are not uploaded yet: {skipped}")
    return [d for d in datasets_for_tier(tier) if is_curated(d)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit multimodal feature counts of MulTaBench datasets")
    parser.add_argument("--tier", type=str, default="uploaded",
                        choices=["uploaded"] + [t.value for t in Tier],
                        help="'uploaded' (default) audits every uploaded dataset, tiers audit their members")
    args = parser.parse_args()

    all_ids = _resolve_ids(args.tier)
    print(f"Auditing {len(all_ids)} MulTaBench datasets  (threshold: ≤{MAX_MULTIMODAL} = ✅)")
    print("=" * 80)

    audits: list[DatasetAudit] = []
    for dataset_id in all_ids:
        print(f"\n→ loading {dataset_id.name} ...")
        audit = audit_dataset(dataset_id)
        print_audit(audit)
        audits.append(audit)

    print_summary(audits)


if __name__ == "__main__":
    main()
