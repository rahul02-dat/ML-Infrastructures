## AMP Scaler & Gradient Accumulation Benchmark Task (`fix-amp-scaler`)

This repository contains a machine learning infrastructure debugging task designed for evaluating AI coding agents. The task focuses on diagnosing and fixing a subtle, silent gradient accumulation defect that corrupts mixed-precision training dynamics when scaling across micro-batches (`span_count > 1`).

---

## 📌 Task Overview

In modern distributed and mixed-precision training pipelines (using PyTorch `torch.amp.GradScaler` and `torch.autocast`), gradient accumulation across micro-batches allows simulating larger batch sizes without exhausting GPU/CPU memory. 

In this benchmark setup:
- A small transformer (`TinyTransformer`) is trained under `bfloat16` CPU autocast.
- Micro-batch chunks are accumulated over a configurable span width (`--span_count`).
- **The Issue**: For `--span_count 1`, training behaves normally. However, for `--span_count > 1`, training dynamics and loss curves diverge noticeably from expected baselines without raising exceptions or producing NaNs.

---

## 📁 Repository Structure

```
ML_infrastructure/
├── task.toml                  # Benchmark task metadata, objective, and test config
├── instruction.md             # Detailed task instructions provided to the agent
├── README.md                  # Project overview and documentation (this file)
├── environment/
│   ├── Dockerfile             # Docker container definition for execution
│   └── data/
│       ├── train.py           # Main training script (contains the bugs to fix)
│       ├── precision.py       # Mixed precision and gradient buffer helper modules
│       ├── model.py           # TinyTransformer architecture definition
│       └── data.py            # Synthetic batch generator
├── solution/
│   ├── solve.py               # Ground-truth corrected script
│   └── solve.sh               # Shell script executing the benchmark solution
└── tests/
    ├── test_outputs.py        # PyTest suite verifying log metrics against ground truth
    ├── test.sh                # Test runner script
    └── ref/                   # Sealed reference implementation for verification
```

---

## 🐛 Bug Details

The bug in [`environment/data/train.py`](file:///Users/rahulmac/Documents/Projects/projects/ML_infrastructure/environment/data/train.py) and [`environment/data/precision.py`](file:///Users/rahulmac/Documents/Projects/projects/ML_infrastructure/environment/data/precision.py) consists of two interacting defects:

1. **Redundant Loss Division (Double-Scaling)**:
   - The outer training loop normalizes `loss` by `args.span_count`.
   - Internal helper methods (e.g., `FragmentReducer._stabilize_fragment_scale`) re-divide the loss by `span_count`, causing effective gradients to be scaled down by `span_count²` instead of `span_count`.
2. **Premature Gradient Resetting**:
   - `CastBarrier._reseat_slots()` or `PrecisionGuard.realign()` clears parameter gradients before every micro-batch backward pass.
   - For `span_count > 1`, this destroys previously accumulated gradients so only the final micro-batch in a span contributes to the optimizer update.

---

## 🚀 Execution & Verification

### 1. Running the Training Script
Run `train.py` with varying span counts to generate output JSON metrics in `/app/logs/` (or a custom output directory):

```bash
python3 environment/data/train.py --span_count 1 --out logs/span_1.json
python3 environment/data/train.py --span_count 4 --out logs/span_4.json
python3 environment/data/train.py --span_count 8 --out logs/span_8.json
```

Output metrics write JSON logs structured as:
```json
{
  "span_count": 4,
  "loss_curve": {"50": 3.21, "100": 2.98, "200": 2.71, "400": 2.40},
  "grad_curve": {"50": 0.94, "100": 0.87, "200": 0.91, "400": 0.88},
  "scale_curve": {"50": 65536.0, "100": 65536.0, "200": 65536.0, "400": 65536.0},
  "commit_trace": [[1, 4], [2, 8]]
}
```

### 2. Executing the Reference Solution
To apply the benchmark solution and generate reference logs:

```bash
bash solution/solve.sh
```

### 3. Running Automated Tests
Verification checks generated log files (`span_1.json`, `span_4.json`, `span_8.json`) against reference trajectory tolerances (2% relative error margin):

```bash
pytest tests/test_outputs.py
```
Or run via the test runner script:
```bash
bash tests/test.sh
```
