# Analysis

Plotting and reporting helpers for full-spectrum P/Q models. Trainers import
`pq_report.py`; the `plot_*.py` scripts are standalone diagnostics.

| Script | Purpose |
|--------|---------|
| `pq_report.py` | Eval / CSV / example plots used by `lstm.py`, `gru.py`, etc. |
| `plot_loss_landscapes.py` | Loss landscape visualizations |
| `plot_model_example_comparisons.py` | Pred vs true example spectra |
| `plot_pred_vs_true_by_snr.py` | Predictions stratified by SNR |
| `plot_pred_vs_true_outliers.py` | Outlier examples |
| `plot_val_loss_overlay.py` | Overlay validation loss curves |
| `plot_bias_variance_vs_snr.py` | Bias / variance vs SNR |
| `weight_histo.py` | Weight histograms |

```bash
python ml/analysis/plot_loss_landscapes.py
python ml/analysis/plot_model_example_comparisons.py
```
