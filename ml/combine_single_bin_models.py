"""
Combine per-bin .pth model files into one combined .pth payload.

Each per-bin model is saved by ml/single_bin.py as:
  <model-dir>/binning_model_bin_{idx}.pth

Only .pth files matching that pattern are loaded. Any .ckpt files in the
directory are ignored.

Example:
  python ml/combine_single_bin_models.py \\
    --model-dir TensorStudies/single_bin_results_v2 \\
    --output TensorStudies/single_bin_results_v2/combined_bin_model.pth \\
    --num-bins 249 --strict
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

MODEL_PATTERN = re.compile(r"^binning_model_bin_(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine binning_model_bin_{idx}.pth model files into one .pth file."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing per-bin .pth model files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output combined .pth file (default: <model-dir>/combined_bin_model.pth).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=None,
        help="Expected number of bins. If omitted, all discovered .pth models are combined.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected bin .pth file is missing.",
    )
    return parser.parse_args()


def _model_path(model_dir: Path, bin_idx: int) -> Path:
    return model_dir / f"binning_model_bin_{bin_idx}.pth"


def discover_model_bins(model_dir: Path) -> List[int]:
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory does not exist: {model_dir}")

    discovered: List[int] = []
    for path in sorted(model_dir.glob("binning_model_bin_*.pth")):
        match = MODEL_PATTERN.match(path.name)
        if match is None:
            continue
        discovered.append(int(match.group(1)))
    return discovered


def _warn_ignored_ckpt_files(model_dir: Path) -> None:
    ckpt_files = sorted(model_dir.glob("*.ckpt"))
    if not ckpt_files:
        return
    preview = ", ".join(path.name for path in ckpt_files[:5])
    suffix = "..." if len(ckpt_files) > 5 else ""
    print(
        f"Ignoring {len(ckpt_files)} .ckpt file(s) in {model_dir}: {preview}{suffix}",
        flush=True,
    )


def resolve_bin_indices(model_dir: Path, num_bins: Optional[int], strict: bool) -> List[int]:
    _warn_ignored_ckpt_files(model_dir)
    discovered = discover_model_bins(model_dir)
    if num_bins is None:
        if not discovered:
            raise FileNotFoundError(
                f"No binning_model_bin_*.pth model files found in {model_dir}"
            )
        return discovered

    expected = list(range(num_bins))
    if strict:
        missing = [
            bin_idx for bin_idx in expected if not _model_path(model_dir, bin_idx).exists()
        ]
        if missing:
            preview = missing[:10]
            suffix = "..." if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"Missing {len(missing)} .pth model file(s): {preview}{suffix}"
            )
    return expected


def _as_feature_vector(value: Any, field_name: str, bin_idx: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.reshape(-1)
    raise ValueError(
        f"Unexpected shape for {field_name} in bin {bin_idx}: {arr.shape}"
    )


def _load_input_stats(
    model_payload: Dict[str, Any], bin_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    if "X_mean" in model_payload and "X_std" in model_payload:
        return (
            _as_feature_vector(model_payload["X_mean"], "X_mean", bin_idx),
            _as_feature_vector(model_payload["X_std"], "X_std", bin_idx),
        )

    if "Ps_mean" in model_payload and "Ps_std" in model_payload:
        return (
            _as_feature_vector(model_payload["Ps_mean"], "Ps_mean", bin_idx),
            _as_feature_vector(model_payload["Ps_std"], "Ps_std", bin_idx),
        )

    raise KeyError(
        f"Model file for bin {bin_idx} is missing input scaling stats "
        "(expected X_mean/X_std or legacy Ps_mean/Ps_std)."
    )


def _require_same(
    current: Any,
    new: Any,
    field_name: str,
    bin_idx: int,
) -> None:
    if current is None:
        return
    if current != new:
        raise ValueError(
            f"Inconsistent {field_name} for bin {bin_idx}: {new!r} (expected {current!r})"
        )


def load_bin_models(
    model_dir: Path,
    bin_indices: Sequence[int],
) -> Dict[str, Any]:
    bin_state_dicts: List[Dict[str, torch.Tensor]] = []
    x_mean_rows: List[np.ndarray] = []
    x_std_rows: List[np.ndarray] = []
    iplus_mean: List[float] = []
    iplus_std: List[float] = []
    iminus_mean: List[float] = []
    iminus_std: List[float] = []
    best_val_loss: List[float] = []
    loaded_bin_indices: List[int] = []

    use_hidden: Optional[bool] = None
    hidden_dim: Optional[int] = None
    feature_names: Optional[List[str]] = None
    feature_set: Optional[str] = None
    input_dim: Optional[int] = None

    for bin_idx in bin_indices:
        model_path = _model_path(model_dir, bin_idx)
        if not model_path.exists():
            print(f"Skipping missing bin {bin_idx}: {model_path}", flush=True)
            continue
        if model_path.suffix != ".pth":
            raise ValueError(f"Expected a .pth model file, got: {model_path}")

        try:
            model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load .pth model for bin {bin_idx}: {model_path}. "
                "This file is likely incomplete or corrupted. Recreate that "
                "per-bin .pth file and rerun combine."
            ) from exc

        state_dict = model_payload.get("model_state_dict")
        if state_dict is None:
            raise KeyError(f"Missing model_state_dict in {model_path}")

        payload_feature_names = list(
            model_payload.get("feature_names", ["ps_at_burn_bin"])
        )
        payload_input_dim = int(
            model_payload.get("input_dim", len(payload_feature_names))
        )
        payload_use_hidden = bool(model_payload.get("use_hidden", True))
        payload_hidden_dim = int(model_payload.get("hidden_dim", 128))
        payload_feature_set = model_payload.get("args", {}).get("feature_set")
        if payload_feature_set is None:
            payload_feature_set = (
                "ps_only" if len(payload_feature_names) == 1 else "burn_context"
            )

        _require_same(feature_names, payload_feature_names, "feature_names", bin_idx)
        _require_same(input_dim, payload_input_dim, "input_dim", bin_idx)
        _require_same(use_hidden, payload_use_hidden, "use_hidden", bin_idx)
        _require_same(hidden_dim, payload_hidden_dim, "hidden_dim", bin_idx)
        _require_same(feature_set, payload_feature_set, "feature_set", bin_idx)

        feature_names = payload_feature_names
        input_dim = payload_input_dim
        use_hidden = payload_use_hidden
        hidden_dim = payload_hidden_dim
        feature_set = payload_feature_set

        x_mean_row, x_std_row = _load_input_stats(model_payload, bin_idx)
        if x_mean_row.shape != (input_dim,):
            raise ValueError(
                f"Bin {bin_idx} X_mean shape {x_mean_row.shape} does not match "
                f"input_dim={input_dim}"
            )

        bin_state_dicts.append(state_dict)
        x_mean_rows.append(x_mean_row)
        x_std_rows.append(x_std_row)
        iplus_mean.append(float(model_payload["Iplus_mean"]))
        iplus_std.append(float(model_payload["Iplus_std"]))
        iminus_mean.append(float(model_payload["Iminus_mean"]))
        iminus_std.append(float(model_payload["Iminus_std"]))
        best_val_loss.append(float(model_payload.get("best_val_loss", float("nan"))))
        loaded_bin_indices.append(int(model_payload.get("bin_idx", bin_idx)))

    if not bin_state_dicts:
        raise RuntimeError(f"No .pth model files found in {model_dir}")

    stats: Dict[str, np.ndarray] = {
        "X_mean": np.stack(x_mean_rows, axis=0).astype(np.float32),
        "X_std": np.stack(x_std_rows, axis=0).astype(np.float32),
        "Iplus_mean": np.asarray(iplus_mean, dtype=np.float32),
        "Iplus_std": np.asarray(iplus_std, dtype=np.float32),
        "Iminus_mean": np.asarray(iminus_mean, dtype=np.float32),
        "Iminus_std": np.asarray(iminus_std, dtype=np.float32),
    }
    if input_dim == 1:
        stats["Ps_mean"] = stats["X_mean"][:, 0]
        stats["Ps_std"] = stats["X_std"][:, 0]

    return {
        "bin_state_dicts": bin_state_dicts,
        "stats": stats,
        "loaded_bin_indices": loaded_bin_indices,
        "use_hidden": use_hidden if use_hidden is not None else True,
        "hidden_dim": hidden_dim if hidden_dim is not None else 128,
        "feature_names": feature_names if feature_names is not None else ["ps_at_burn_bin"],
        "feature_set": feature_set if feature_set is not None else "ps_only",
        "input_dim": input_dim if input_dim is not None else 1,
        "best_val_loss": np.asarray(best_val_loss, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    output_path = args.output or (args.model_dir / "combined_bin_model.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bin_indices = resolve_bin_indices(
        model_dir=args.model_dir,
        num_bins=args.num_bins,
        strict=args.strict,
    )
    combined = load_bin_models(model_dir=args.model_dir, bin_indices=bin_indices)

    payload = {
        "num_bins": len(combined["bin_state_dicts"]),
        "bin_state_dicts": combined["bin_state_dicts"],
        "stats": combined["stats"],
        "use_hidden": combined["use_hidden"],
        "hidden_dim": combined["hidden_dim"],
        "input_dim": combined["input_dim"],
        "feature_names": combined["feature_names"],
        "feature_set": combined["feature_set"],
        "loaded_bin_indices": combined["loaded_bin_indices"],
        "best_val_loss_per_bin": combined["best_val_loss"],
        "source_model_dir": str(args.model_dir),
    }
    torch.save(payload, output_path)

    print(f"Saved combined model to {output_path}", flush=True)
    print(f"Loaded {payload['num_bins']} .pth model file(s)", flush=True)
    print(
        " | ".join(
            [
                f"feature_set={payload['feature_set']}",
                f"input_dim={payload['input_dim']}",
                f"use_hidden={payload['use_hidden']}",
                f"hidden_dim={payload['hidden_dim']}",
            ]
        ),
        flush=True,
    )
    if args.num_bins is not None and payload["num_bins"] != args.num_bins:
        print(
            f"Warning: expected {args.num_bins} bins, loaded {payload['num_bins']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
