"""Download and prepare the knowledge-editing benchmark datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def export_jsonl(dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(dataset, "to_json"):
        dataset.to_json(path, orient="records", lines=True)
        return
    with path.open("w") as stream:
        for row in dataset:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def limit_rows(rows, limit: int | None):
    if limit is None:
        return rows
    return rows.select(range(min(limit, len(rows))))


def prepare_counterfact(data_dir: Path, limit: int | None) -> None:
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": (
                "https://huggingface.co/datasets/azhx/counterfact/resolve/main/"
                "data/train-00000-of-00001-05d11247db7abce8.parquet"
            ),
            "test": (
                "https://huggingface.co/datasets/azhx/counterfact/resolve/main/"
                "data/test-00000-of-00001-bacb83500fca49a9.parquet"
            ),
        },
    )
    for split, rows in dataset.items():
        export_jsonl(
            limit_rows(rows, limit),
            data_dir / f"{split}.jsonl",
        )


def prepare_zsre(data_dir: Path, limit: int | None) -> None:
    dataset = load_dataset("wangzn2001/zsre")
    for split, rows in dataset.items():
        export_jsonl(limit_rows(rows, limit), data_dir / f"{split}.jsonl")


def classify_mquake(labels: list[str]) -> str:
    if "train_edited" in labels:
        return "train_edited"
    if "test_edited" in labels or "test_edited_unique" in labels:
        return "test_edited"
    if "test_unedited" in labels:
        return "unedited"
    if "edited" in labels:
        return "edited"
    return "unclassified"


def clean_mquake_case(row: dict, split: str, edit_num: int) -> dict:
    rewrite = row["requested_rewrite"][0]
    original_hops = row.get("single_hops") or []
    edited_hops = row.get("new_single_hops") or []
    labels = row.get("split", {}).get(str(edit_num), [])
    status = classify_mquake(labels)
    if status == "unclassified" and split != "CF6334":
        status = "benchmark_case"

    return {
        "case_id": row["case_id"],
        "dataset_split": split,
        "status": status,
        "edit_num": edit_num,
        "edit": {
            "subject": rewrite["subject"],
            "relation_id": rewrite["relation_id"],
            "prompt": rewrite["prompt"],
            "old_answer": rewrite.get("target_true_str"),
            "new_answer": rewrite.get("target_new_str"),
        },
        "questions": row.get("questions") or [],
        "original_answers": [hop.get("answer") for hop in original_hops],
        "edited_answers": [hop.get("answer") for hop in edited_hops],
        "original_answer_aliases": [
            hop.get("answer_alias", []) for hop in original_hops
        ],
        "edited_answer_aliases": [
            hop.get("answer_alias", []) for hop in edited_hops
        ],
        "split_labels": labels,
    }


def prepare_mquake(data_dir: Path, edit_num: int, limit: int | None) -> None:
    dataset = load_dataset("henryzhongsc/MQuAKE-Remastered")
    output_dir = data_dir
    clean_dir = output_dir / "clean"

    for split, rows in dataset.items():
        rows = limit_rows(rows, limit)
        export_jsonl(rows, output_dir / f"{split}.jsonl")
        clean_rows = [
            clean_mquake_case(row, split, edit_num)
            for row in rows
            if split != "CF6334"
            or classify_mquake(row["split"].get(str(edit_num), []))
            != "unclassified"
        ]
        export_jsonl(clean_rows, clean_dir / f"{split}.jsonl")

        counts = {}
        for row in clean_rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        (clean_dir / f"{split}_summary.json").write_text(
            json.dumps(
                {
                    "dataset_split": split,
                    "edit_num": edit_num,
                    "total_cases": len(rows),
                    "clean_cases": len(clean_rows),
                    "status_counts": counts,
                },
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("all", "counterfact", "zsre", "mquake"),
        default="all",
        help="Dataset family to prepare.",
    )
    parser.add_argument(
        "--edit-num",
        type=int,
        default=6334,
        choices=(100, 1000, 3000, 6334),
        help="MQuAKE split label used for edited/unedited preparation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to export from each dataset split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Root directory for exported dataset folders.",
    )
    args = parser.parse_args()

    if args.dataset in ("all", "counterfact"):
        output_dir = (
            args.output_dir / "counterfact"
            if args.dataset == "all"
            else args.output_dir
        )
        prepare_counterfact(output_dir, args.limit)
    if args.dataset in ("all", "zsre"):
        output_dir = (
            args.output_dir / "zsre" if args.dataset == "all" else args.output_dir
        )
        prepare_zsre(output_dir, args.limit)
    if args.dataset in ("all", "mquake"):
        output_dir = (
            args.output_dir / "mquake" if args.dataset == "all" else args.output_dir
        )
        prepare_mquake(output_dir, args.edit_num, args.limit)


if __name__ == "__main__":
    main()
