#!/usr/bin/env python
"""Emit a paper-ready ablation LaTeX table RANKED by a composite score.

Reads a comparison CSV (default comparison_reeval_last.csv) with columns
DICE/PSNR/SSIM/LPIPS/areaMAE_mm2/monoDec_mm2 (+ *_se), computes a z-score composite
(direction-aware), sorts rows best->worst, and bolds the best value per metric column.

  composite = sum_m  w_m * dir_m * zscore(metric_m)        (then / sum w_m)

Two composites are available: --weights equal  or  --weights clinical (DICE & areaMAE x2,
monoDec x0.5). Default = clinical (the one we locked R3_BASE on).

Usage:
    python ablations/make_ranked_tex.py                       # -> ablations/comparison_ranked.tex
    python ablations/make_ranked_tex.py --weights equal --top 10
    python ablations/make_ranked_tex.py --csv <path> --out <path.tex>
"""
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# metric -> (+1 higher-better / -1 lower-better, latex header, n decimals)
METRICS = [
    ("DICE",        +1, r"DICE$\uparrow$",            3),
    ("PSNR",        +1, r"PSNR$\uparrow$",            2),
    ("SSIM",        +1, r"SSIM$\uparrow$",            3),
    ("LPIPS",       -1, r"LPIPS$\downarrow$",         3),
    ("areaMAE_mm2", -1, r"areaMAE$\downarrow$",       3),
    ("monoDec_mm2", -1, r"monoDec$\downarrow$",       3),
]
WEIGHTS = {
    "equal":    {"DICE": 1, "PSNR": 1, "SSIM": 1, "LPIPS": 1, "areaMAE_mm2": 1, "monoDec_mm2": 1},
    "clinical": {"DICE": 2, "PSNR": 1, "SSIM": 1, "LPIPS": 1, "areaMAE_mm2": 2, "monoDec_mm2": 0.5},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(HERE, "comparison_reeval_last.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "comparison_ranked.tex"))
    ap.add_argument("--weights", choices=list(WEIGHTS), default="clinical")
    ap.add_argument("--top", type=int, default=0, help="keep only the top-N rows (0 = all)")
    ap.add_argument("--no_se", action="store_true", help="print mean only (no +/- se)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    present = [m for m, *_ in METRICS if m in df.columns]

    # z-score composite (population std; direction-aware)
    w = WEIGHTS[args.weights]
    comp = np.zeros(len(df))
    wsum = 0.0
    for m, d, _, _ in METRICS:
        if m not in df.columns:
            continue
        x = df[m].astype(float)
        sd = x.std(ddof=0)
        z = (x - x.mean()) / sd if sd > 0 else x * 0.0
        comp += w[m] * d * z
        wsum += w[m]
    df["composite"] = comp / wsum
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    if args.top > 0:
        df = df.head(args.top)

    # best value per metric (direction-aware) for bolding
    best = {}
    for m, d, _, _ in METRICS:
        if m in df.columns:
            best[m] = (df[m].astype(float).max() if d > 0 else df[m].astype(float).min())

    headers = [h for m, _, h, _ in METRICS if m in df.columns]
    ndec = {m: nd for m, _, _, nd in METRICS}
    cols = "l" + "c" * (len(headers) + 1)

    lines = []
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    lines.append("Ablation & " + " & ".join(headers) + r" & Composite \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        cells = [str(r["ablation"]).replace("_", r"\_")]
        for m in present:
            mu = float(r[m]); nd = ndec[m]
            se = r.get(f"{m}_se", np.nan)
            txt = f"{mu:.{nd}f}"
            if (not args.no_se) and pd.notna(se):
                txt += rf"$\pm${float(se):.{nd}f}"
            if np.isclose(mu, best[m], atol=10 ** (-nd - 1)):
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        cells.append(f"{r['composite']:+.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tex = "\n".join(lines) + "\n"

    with open(args.out, "w") as f:
        f.write(tex)
    print(tex)
    print(f"[{args.weights}] composite -> {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
