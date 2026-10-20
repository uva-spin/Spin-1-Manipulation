# Analysis

Plotting and reporting for the spectrum→P/Q models. The trainers call `pq_report.py`. The `plot_*.py` scripts are separate programs and read checkpoints under `ml/results/`.

Run them from the repo root, after a training script has written a checkpoint.

| Script | Purpose |
|--------|---------|
| `pq_report.py` | Metrics, CSV, and example plots used by `lstm.py`, `gru.py`, and the other trainers |
| `plot_loss_landscapes.py` | Loss on a random 2D plane around a checkpoint |
| `plot_model_example_comparisons.py` | Predicted vs true P and Q on example spectra |
| `plot_pred_vs_true_by_snr.py` | Predictions grouped by SNR |
| `plot_pred_vs_true_outliers.py` | Outlier examples |
| `plot_val_loss_overlay.py` | Validation-loss curves on one figure |
| `plot_bias_variance_vs_snr.py` | Bias and variance against SNR |
| `weight_histo.py` | Weight histograms |
| `plot_rpe_by_polarization_overlay.py` | Relative percent error by polarization band |

```bash
python ml/analysis/plot_loss_landscapes.py --models lstm --n-samples 256
python ml/analysis/plot_val_loss_overlay.py
```
