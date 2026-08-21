#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_row(idx: int, split: str, component: str) -> dict:
    prompt = [
        {
            "role": "user",
            "content": (
                "Use the SCAPE search tools to answer the query. "
                f"Component={component}; toy query {idx}."
            ),
        }
    ]
    return {
        "data_source": "scape_easyopd_tiny",
        "prompt": prompt,
        "ability": "search",
        "reward_model": {"style": "model", "ground_truth": ""},
        "extra_info": {"split": split, "index": idx, "component": component},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scape_easyopd/framework/tiny_data"))
    parser.add_argument("--component", default="auto_populate_first_search")
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--val-size", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = [build_row(i, "train", args.component) for i in range(args.train_size)]
    val = [build_row(i, "val", args.component) for i in range(args.val_size)]
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "test.parquet"
    pd.DataFrame(train).to_parquet(train_path)
    pd.DataFrame(val).to_parquet(val_path)
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
