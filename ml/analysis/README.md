# Analysis

Small standalone visualization scripts. These are not part of the main data → train → evaluate pipeline.

## Scripts

| File | Purpose |
|------|---------|
| `pq_report.py` | Pipeline evaluation/reporting imported by the training scripts (`evaluate_model`, plots, CSVs, `write_pq_report`) |
| `binning/binning_plot.py` | Animated lineshape / binning visualization |
| `binning/sgd.py` | SGD trajectory visualization |

## Usage

```bash
python ml/analysis/binning/binning_plot.py
python ml/analysis/binning/sgd.py
```

Both scripts are self-contained and use matplotlib. No training data or model checkpoints are required.

For production evaluation plots, use [`ml/rivanna/test-binning.py`](../rivanna/README.md) instead.
