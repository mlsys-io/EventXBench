# EventX: A Multimodal Benchmark Linking Social Media Posts to Prediction Market Dynamics

**Dataset: [https://huggingface.co/datasets/mlsys-io/EventXBench](https://huggingface.co/datasets/mlsys-io/EventXBench)**

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EventXBench-blue)](https://huggingface.co/datasets/mlsys-io/EventXBench)
[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%20'26-red)](https://doi.org/PLACEHOLDER)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**EventX** is a multimodal benchmark connecting 9M Twitter/X posts from 1,152 KOL accounts to 11,952 Polymarket prediction markets (2021--2026). It defines seven tasks across two tiers: **resolution** (human-annotated ground truth) and **forecast** (deterministic labels from post-publication tick data).

## Quick Start

```bash
pip install -r requirements.txt

from eventxbench import load_task
train, test = load_task("t1")  # Returns pandas DataFrames

# Run a baseline
python baselines/t1/llm_baseline.py --provider openai --model gpt-4o --shots 0

# Evaluate predictions
python evaluation/evaluate.py --task t1 --predictions results/t1_predictions.jsonl
```

## Benchmark Overview

| Task | Name | Tier | Output | Primary Metrics |
|------|------|------|--------|-----------------|
| T1 | Market Volume Prediction | Forecast | 3-class (`high`/`moderate`/`low`) | Macro-F1, `high`-class P@K |
| T2 | Post-to-Market Linking | Resolution | Market ID or `none` | Accuracy@1, MRR |
| T3 | Evidence Grading | Resolution | Ordinal 0--5 | QWK (kappa), macro-F1 |
| T4 | Market Movement Prediction | Forecast | Direction x Magnitude | Dir-Acc, Mag-F1, Spearman rho |
| T5 | Volume & Price Impact | Forecast | Continuous | Spearman rho (price_impact, volume_multiplier) |
| T6 | Cross-Market Propagation | Forecast | 3-class | Macro-F1, MAE (onset lag) |
| T7 | Impact Persistence (Decay) | Forecast | 3-class (`transient`/`sustained`/`reversal`) | Macro-F1 |

## Tasks

### T1: Conditional Market Volume Prediction
Predict the final trading-volume percentile of a subsequently created market, given pre-market social signals.

- **Labels**: `high` (>80th pctl), `moderate` (40th--80th), `low` (<40th)
- **Input**: Tweet cluster features, event metadata, temporal features
- **Metrics**: Macro-F1, `high`-class precision@K

### T2: Post-to-Market Linking
Given a tweet and a candidate set recalled by BGE-large-en-v1.5 / FAISS dense retrieval, identify which market the post addresses (or `none`).

- **Input**: Tweet text + candidate market questions
- **Metrics**: Accuracy@1, MRR, `none`-class F1

### T3: Evidence Grading and Resolution Potential
Assign an ordinal evidence grade (0--5) to each post-market pair.

- **Grade scale**: 0 (`noise`), 1 (`commentary_reaction`), 2 (`speculation_rumor`), 3 (`indirect_report`), 4 (`strong_direct`), 5 (`resolving`)
- **Metrics**: Quadratic-weighted kappa, `resolving`-class precision, macro-F1

### T4: Market Movement Prediction
Predict direction and magnitude of the YES-price change at 2-hour horizon after a tweet.

- **Direction**: `up` (delta > 0.02), `down` (delta < -0.02), `flat` (otherwise)
- **Magnitude**: `large` (>8%), `medium` (2--8%), `small` (<=2%)
- **Metrics**: Direction accuracy, Magnitude macro-F1, Spearman rho on continuous delta curve
- **Secondary horizons**: 30 min, 6 h

### T5: Volume and Price Impact
Predict two continuous targets per post: (i) `price_impact` (max absolute deviation from p0), (ii) `volume_multiplier` (total volume / 24h baseline).

- **Targets**: `price_impact` (continuous), `volume_multiplier` (continuous)
- **Metrics**: Spearman rho for each target
- **Note**: T5 and T7 share the same underlying data (`task5+7/` in the codebase). T5 evaluates the continuous predictions; T7 evaluates the decay classification.

### T7: Impact Persistence (Decay)
Classify whether a tweet's initial market impact is transient, sustained, or reverses over time.

- **Labels**: `transient` (|delta_2h| < 30% of |delta_15m|), `sustained` (same sign, larger), `reversal` (sign flips at 2h)
- **Metrics**: Macro-F1
- **Note**: Uses the same data as T5 but evaluates the `decay_class` field. In the codebase, uses `task5+7/` directories and `t7_` prefixes.

### T6: Cross-Market Propagation
Predict whether a tweet's market impact propagates to sibling markets within 2 hours. A sibling is deemed "moved" if |delta_p| > 1.5 sigma (rolling 24h stdev).

- **Labels**: `no_effect`, `primary_mover` (primary market moves first, sibling follows), `propagated_signal` (sibling moves before or instead of primary)
- **Metrics**: Macro-F1, MAE on onset lag (minutes)

## Data

### Hugging Face Dataset

All data is hosted on Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("mlsys-io/EventXBench", "t1")

# Or use our convenience loader
from eventxbench import load_task
train, test = load_task("t1")
```

See [`data/README.md`](data/README.md) for the full dataset card.

### Data Files

| File | Description | Size |
|------|-------------|------|
| `posts_no_text.jsonl` | Tweet IDs and metadata (text stripped for privacy) | ~9M rows |
| `market_fundamental.json` | Market metadata (question, category, resolution) | 11,952 markets |
| `market_ohlcv.json` | Price/volume time series (OHLCV) | 1.8 GB |
| `t1_labels.jsonl` | T1 ground truth with train/test splits | 326 |
| `t2_groundtruth.jsonl` | T2 post-market linking pairs | 815 |
| `t3_graded.json` | T3 evidence grades (0--5) | 342,552 |
| `t4_labels.jsonl` | T4 direction x magnitude labels | 4,803 |
| `t5_labels.jsonl` | T5 price impact + volume multiplier | 407 |
| `t6_labels.jsonl` | T6 cross-market propagation labels | 4,006 |
| `t7_labels.jsonl` | T7 decay class labels (same data as T5) | 407 |

### Privacy

Tweet text is **not** included in the public release to comply with Twitter/X Terms of Service. The `posts_no_text.jsonl` file contains tweet IDs for rehydration via the Twitter API. Market data from Polymarket is fully included.

## Baselines

We provide three baseline families:

### LLM Baselines
Zero-shot and few-shot prompting:
- GPT-4o (OpenAI)
- Sonnet 4.5 (Anthropic)
- Grok 4.1 (xAI)
- Qwen 3.5 (local via vLLM, 4B and 27B)

```bash
# T1 zero-shot with GPT-4o
python baselines/t1/llm_baseline.py --provider openai --model gpt-4o --shots 0

# T4 three-shot with Claude
python baselines/t4/llm_baseline.py --provider anthropic --model claude-sonnet-4-5-20250514 --shots 3
```

**Required env vars**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `HF_TOKEN` (for Qwen)

### ML Baselines
LightGBM classifiers with Bayesian hyperparameter tuning (Optuna):

```bash
python baselines/t1/lightgbm_baseline.py
python baselines/t4/lightgbm_baseline.py
```

### Heuristic & Basic Baselines
Majority class, random walk, graph heuristics, BM25 retrieval:

```bash
python baselines/t1/basic_baseline.py
python baselines/t6/basic_baseline.py
```

## Evaluation

```bash
python evaluation/evaluate.py --task t1 --predictions results/t1_preds.jsonl
python evaluation/evaluate.py --task t4 --predictions results/t4_preds.jsonl
python evaluation/evaluate.py --task all --predictions-dir results/
```

See [`evaluation/README.md`](evaluation/README.md) for prediction format specs.

## Repository Structure

```
EventXBench/
├── README.md
├── LEADERBOARD.md
├── requirements.txt
├── eventxbench/                  # Data loading utilities
│   ├── __init__.py
│   └── loader.py
├── data/
│   └── README.md                 # Hugging Face dataset card
├── baselines/
│   ├── t1/ ... t6/               # Per-task baselines (LLM, ML, basic)
│   └── t7/                       # Impact Persistence (Decay) baselines
├── evaluation/
│   ├── evaluate.py               # Unified evaluation CLI
│   ├── metrics.py                # Metric implementations
│   └── README.md
├── scripts/
│   └── upload_to_hf.py           # Upload data to Hugging Face
└── examples/
    └── quickstart.py
```

## Leaderboard

See [LEADERBOARD.md](LEADERBOARD.md) for current results. To submit, open a pull request.

## License

- **Code**: MIT License
- **Data**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- Tweet text excluded; use Twitter API for rehydration
- Polymarket data included under fair use for research
