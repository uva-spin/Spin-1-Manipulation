# Analysis

Small standalone visualization scripts. These are not part of the main data → train → evaluate pipeline.

## Scripts

| File | Purpose |
|------|---------|
| `binning/binning_plot.py` | Animated lineshape / binning visualization |
| `binning/sgd.py` | SGD trajectory visualization |

## Usage

```bash
python analysis/binning/binning_plot.py
python analysis/binning/sgd.py
```

Both scripts are self-contained and use matplotlib. No training data or model checkpoints are required.

For production evaluation plots, use [`ml/rivanna/test-binning.py`](../ml/rivanna/README.md) instead.
