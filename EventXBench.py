"""EventXBench dataset loading script for Hugging Face `datasets` library.

This script is auto-detected by HF when the repo contains a .py file with the
same name as the repo. It defines dataset configs for each task (t1--t6) and
for the auxiliary data (posts, markets, ohlcv).

Usage:
    from datasets import load_dataset

    # Load a specific task
    ds = load_dataset("mlsys-io/EventXBench", "t1")
    train_df = ds["train"].to_pandas()

    # Load all configs
    ds = load_dataset("mlsys-io/EventXBench", "t4")
"""
from __future__ import annotations

import json
import os

import datasets


_DESCRIPTION = (
    "EventX: A multimodal benchmark linking Twitter/X posts to "
    "Polymarket prediction market dynamics across seven tasks."
)

_HOMEPAGE = "https://github.com/mlsys-io/EventXBench"
_LICENSE = "cc-by-nc-4.0"

_URLS = {
    "t1_train": "data/t1/train.jsonl",
    "t1_test": "data/t1/test.jsonl",
    "t2_test": "data/t2/test.jsonl",
    "t3_test": "data/t3/test.jsonl",
    "t4_train": "data/t4/train.jsonl",
    "t4_test": "data/t4/test.jsonl",
    "t5_train": "data/t5/train.jsonl",
    "t5_test": "data/t5/test.jsonl",
    "t6_train": "data/t6/train.jsonl",
    "t6_validation": "data/t6/validation.jsonl",
    "t6_test": "data/t6/test.jsonl",
    "t7_train": "data/t7/train.jsonl",
    "t7_test": "data/t7/test.jsonl",
}


class EventXBenchConfig(datasets.BuilderConfig):
    """BuilderConfig for EventXBench."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EventXBench(datasets.GeneratorBasedBuilder):
    """EventXBench dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        EventXBenchConfig(
            name="t1",
            version=VERSION,
            description="T1: Conditional Market Volume Prediction (3-class)",
        ),
        EventXBenchConfig(
            name="t2",
            version=VERSION,
            description="T2: Post-to-Market Linking",
        ),
        EventXBenchConfig(
            name="t3",
            version=VERSION,
            description="T3: Evidence Grading (ordinal 0-5)",
        ),
        EventXBenchConfig(
            name="t4",
            version=VERSION,
            description="T4: Market Movement Prediction (direction x magnitude)",
        ),
        EventXBenchConfig(
            name="t5",
            version=VERSION,
            description="T5: Volume & Price Impact (decay classification)",
        ),
        EventXBenchConfig(
            name="t6",
            version=VERSION,
            description="T6: Cross-Market Propagation (3-class)",
        ),
        EventXBenchConfig(
            name="t7",
            version=VERSION,
            description="T7: Impact Persistence / Decay classification (3-class)",
        ),
    ]

    DEFAULT_CONFIG_NAME = "t1"

    def _info(self):
        # Use generic features since each task has different schemas.
        # HF will infer the schema from the first batch of examples.
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=None,  # auto-inferred from data
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        config = self.config.name

        # Determine which files to download
        files_to_dl = {}
        for key, url in _URLS.items():
            if key.startswith(config + "_"):
                files_to_dl[key] = url

        downloaded = dl_manager.download_and_extract(files_to_dl)

        splits = []
        train_key = f"{config}_train"
        validation_key = f"{config}_validation"
        test_key = f"{config}_test"

        if train_key in downloaded:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": downloaded[train_key]},
                )
            )
        if validation_key in downloaded:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": downloaded[validation_key]},
                )
            )
        if test_key in downloaded:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": downloaded[test_key]},
                )
            )

        return splits

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    yield idx, json.loads(line)
