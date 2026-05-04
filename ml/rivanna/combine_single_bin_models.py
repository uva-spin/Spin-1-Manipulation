"""
Combine per-bin model checkpoints into one .pth payload.

Example:
  python TensorStudies/combine_single_bin_models.py \
    --model-dir TensorStudies/single_bin_results \
    --output TensorStudies/single_bin_results/combined_500_bin_model.pth
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine binning_model_bin_{idx}.pth checkpoints into one file."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("TensorStudies/single_bin_results"),
        help="Directory containing per-bin checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TensorStudies/combined_results_v3/combined_500_bin_model.pth"),
        help="Path to output combined .pth file.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=500,
        help="Expected number of bins/checkpoints.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any bin checkpoint is missing.",
    )
    return parser.parse_args()


def _checkpoint_path(model_dir: Path, bin_idx: int) -> Path:
    return model_dir / f"binning_model_bin_{bin_idx}.pth"


def load_bin_checkpoints(
    model_dir: Path, num_bins: int, strict: bool
) -> Dict[str, Any]:
    bin_state_dicts: List[Dict[str, torch.Tensor]] = []
    ps_mean: List[float] = []
    ps_std: List[float] = []
    iplus_mean: List[float] = []
    iplus_std: List[float] = []
    iminus_mean: List[float] = []
    iminus_std: List[float] = []
    best_val_loss: List[float] = []
    loaded_bin_indices: List[int] = []
    use_hidden = None
    hidden_dim = None

    missing_bins: List[int] = []
    for bin_idx in range(num_bins):
        ckpt_path = _checkpoint_path(model_dir, bin_idx)
        if not ckpt_path.exists():
            if strict:
                missing_bins.append(bin_idx)
                continue
            print(f"Skipping missing bin {bin_idx}: {ckpt_path}", flush=True)
            continue

        # Checkpoints from train_single_bin_model.py include non-tensor metadata
        # (for example Path objects in args), so allow full trusted pickle load.
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint for bin {bin_idx}: {ckpt_path}. "
                "This file is likely incomplete or corrupted. Recreate that "
                "per-bin checkpoint and rerun combine."
            ) from exc
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError(f"Missing model_state_dict in {ckpt_path}")

        if use_hidden is None:
            use_hidden = bool(checkpoint.get("use_hidden", True))
        if hidden_dim is None:
            hidden_dim = int(checkpoint.get("hidden_dim", 128))

        bin_state_dicts.append(state_dict)
        ps_mean.append(float(checkpoint["Ps_mean"]))
        ps_std.append(float(checkpoint["Ps_std"]))
        iplus_mean.append(float(checkpoint["Iplus_mean"]))
        iplus_std.append(float(checkpoint["Iplus_std"]))
        iminus_mean.append(float(checkpoint["Iminus_mean"]))
        iminus_std.append(float(checkpoint["Iminus_std"]))
        best_val_loss.append(float(checkpoint.get("best_val_loss", float("nan"))))
        loaded_bin_indices.append(int(checkpoint.get("bin_idx", bin_idx)))

    if strict and missing_bins:
        raise FileNotFoundError(
            f"Missing {len(missing_bins)} checkpoint(s): "
            f"{missing_bins[:10]}{'...' if len(missing_bins) > 10 else ''}"
        )

    if not bin_state_dicts:
        raise RuntimeError(f"No checkpoints found in {model_dir}")

    stats = {
        "Ps_mean": np.asarray(ps_mean, dtype=np.float32),
        "Ps_std": np.asarray(ps_std, dtype=np.float32),
        "Iplus_mean": np.asarray(iplus_mean, dtype=np.float32),
        "Iplus_std": np.asarray(iplus_std, dtype=np.float32),
        "Iminus_mean": np.asarray(iminus_mean, dtype=np.float32),
        "Iminus_std": np.asarray(iminus_std, dtype=np.float32),
    }
    return {
        "bin_state_dicts": bin_state_dicts,
        "stats": stats,
        "loaded_bin_indices": loaded_bin_indices,
        "use_hidden": use_hidden if use_hidden is not None else True,
        "hidden_dim": hidden_dim if hidden_dim is not None else 128,
        "best_val_loss": np.asarray(best_val_loss, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    combined = load_bin_checkpoints(
        model_dir=args.model_dir,
        num_bins=args.num_bins,
        strict=args.strict,
    )

    payload = {
        "num_bins": len(combined["bin_state_dicts"]),
        "bin_state_dicts": combined["bin_state_dicts"],
        "stats": combined["stats"],
        "use_hidden": combined["use_hidden"],
        "hidden_dim": combined["hidden_dim"],
        "loaded_bin_indices": combined["loaded_bin_indices"],
        "best_val_loss_per_bin": combined["best_val_loss"],
        "source_model_dir": str(args.model_dir),
    }
    torch.save(payload, args.output)

    print(f"Saved combined model to {args.output}", flush=True)
    print(f"Loaded {payload['num_bins']} bin checkpoint(s)", flush=True)
    if payload["num_bins"] != args.num_bins:
        print(
            f"Warning: expected {args.num_bins} bins, found {payload['num_bins']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
