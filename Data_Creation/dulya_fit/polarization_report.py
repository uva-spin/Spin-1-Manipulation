"""Print integrated vector/tensor polarization before and after manipulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lineshape import shape_params_from_fit  # noqa: E402
from polarization import polarization_before_after  # noqa: E402


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    shape = shape_params_from_fit()
    records = []
    for _, row in df.iterrows():
        pol = polarization_before_after(row.to_dict(), shape)
        records.append(
            {
                "manipulation_mode": row.get("manipulation_mode", "none"),
                "P_input": pol["P_input"],
                "P_vec_before": pol["P_vec_before"],
                "Q_tensor_before": pol["Q_tensor_before"],
                "P_vec_after": pol["P_vec_after"],
                "Q_tensor_after": pol["Q_tensor_after"],
                "dP_vec": pol["dP_vec"],
                "dQ_tensor": pol["dQ_tensor"],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Integrated P/Q before and after manipulation from I±"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=_HERE / "data" / "sample_event_P0p48.pkl",
        help="Pickle with Iplus/Iminus rows",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary path",
    )
    args = p.parse_args()

    df = pd.read_pickle(args.input)
    report = build_report(df)

    shape = shape_params_from_fit()
    amp = float(shape["amp"])
    print("Integrated polarizations (amp + post-correction):")
    print(f"  naive     = sum(I+ + I-) / (amp * n_bins)   amp={amp:.6g}")
    print(f"  P_vec     = post-correct naive via equilibrium calibration")
    print(f"  Q_tensor  = Q_naive * (P_vec / P_naive)")
    print()
    print(report.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict(orient="records")
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
