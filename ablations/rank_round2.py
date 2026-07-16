#!/usr/bin/env python3
"""Select the best round-2 config when DICE is saturated.

Rationale: all configs sit at ~the same DICE, so DICE alone can't discriminate. We give the
lesion-size MAE (area_MAE_mm2) primary weight, but guard the reconstruction quality (PSNR/SSIM/
LPIPS) so we don't pick a config that predicts lesion area well while producing garbage FAF images.

Reads each config's summary CSV (default: leave_one_out_summary_reeval_last.csv) from every
sub-directory of --runs_dir, takes the row matching --group, and reports FOUR views:

  1. GUARDRAIL + PRIMARY (recommended): drop configs whose recon falls outside a tolerance of the
     best-reconstructing config, then pick the survivor with the lowest lesion-size MAE.
  2. WEIGHTED mean-rank: recon folded into ONE composite rank so image quality counts once;
     default weights lesion=0.5, recon=0.4, dice=0.1. Winner = lowest weighted mean rank.
  3. PLAIN mean-rank over {DICE, area_MAE, PSNR, SSIM, LPIPS} (the naive Borda baseline).
  4. NORMALIZED-score aggregation (min-max per metric, same weights) -- keeps magnitudes.

Ranks use 1=best with correct per-metric direction; configs within 1 se on a metric are collapsed
to a shared (averaged) rank so saturated metrics don't inject noise.

Usage (on the server holding the round-2 runs):
    python rank_round2.py --runs_dir /path/to/round2_runs
    python rank_round2.py --runs_dir . --group extrapolation \
        --weights lesion=0.6,recon=0.35,dice=0.05 \
        --guardrail psnr=1.0,ssim=0.02,lpips=0.03
"""
import argparse, csv, glob, os, sys
from statistics import mean

# metric -> (csv column, se column or None, higher_is_better)
METRICS = {
    "DICE":     ("DICE_mean",    "DICE_se",  True),
    "area_MAE": ("area_MAE_mm2", None,       False),   # lesion-size MAE (mm^2)
    "PSNR":     ("PSNR_mean",    "PSNR_se",  True),
    "SSIM":     ("SSIM_mean",    "SSIM_se",  True),
    "LPIPS":    ("LPIPS_mean",   "LPIPS_se", False),
    "HD":       ("HD_mean",      "HD_se",    False),
}
RECON = ["PSNR", "SSIM", "LPIPS"]   # folded into one composite rank


def _f(x):
    try:
        v = float(x)
        return v if v == v else None   # drop NaN
    except (TypeError, ValueError):
        return None


def _epoch(path):
    import re
    m = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def mae_from_lesion(config_dir, prefer=("test", "val")):
    """Fallback lesion-size MAE (mm^2) when the summary column is blank: mean|Pred-GT| over the
    held-out ('{split}_eval') rows, mirroring summarize_eval. Uses the SINGLE highest-epoch file
    per split (best-effort proxy for the best checkpoint), preferring test over val. Returns
    (mae, split_used) or (None, None)."""
    files = glob.glob(os.path.join(config_dir, "**", "lesion_areas*.csv"), recursive=True)
    for split in prefer:
        cand = []
        for p in sorted(files, key=_epoch, reverse=True):   # newest epoch first
            try:
                rows = list(csv.DictReader(open(p)))
            except Exception:
                continue
            errs = [abs(_f(r["Pred_Area_mm2"]) - _f(r["GT_Area_mm2"]))
                    for r in rows if str(r.get("Set", "")).startswith(f"{split}_eval")
                    and _f(r.get("Pred_Area_mm2")) is not None and _f(r.get("GT_Area_mm2")) is not None]
            if errs:
                cand = errs; break        # first (newest) file with this split's eval rows wins
        if cand:
            return mean(cand), split
    return None, None


def load_configs(runs_dir, summary_name, group):
    """Return {config_name: {metric: (value, se)}} from each summary CSV. Recomputes area_MAE from
    lesion_areas*.csv when the summary's area_MAE_mm2 column is blank."""
    out, recomputed = {}, []
    for csv_path in sorted(glob.glob(os.path.join(runs_dir, "**", summary_name), recursive=True)):
        # config name = the dir directly under runs_dir (walk up from the CSV)
        rel = os.path.relpath(csv_path, runs_dir)
        name = rel.split(os.sep)[0]
        rows = list(csv.DictReader(open(csv_path)))
        row = next((r for r in rows if group.lower() in (r.get("group", "") or "").lower()), None)
        if row is None and rows:
            row = rows[0]   # fall back to the only/first row
        if row is None:
            print(f"  [skip] {name}: no rows in {csv_path}", file=sys.stderr); continue
        vals = {}
        for m, (col, se_col, _) in METRICS.items():
            v = _f(row.get(col))
            se = _f(row.get(se_col)) if se_col else None
            if v is not None:
                vals[m] = (v, se)
        if "area_MAE" not in vals:      # blank summary column -> recompute from lesion CSVs
            mae, split = mae_from_lesion(os.path.join(runs_dir, name))
            if mae is not None:
                vals["area_MAE"] = (mae, None); recomputed.append(f"{name}({split})")
        out[name] = vals
    if recomputed:
        print(f"[note] area_MAE recomputed from lesion_areas*.csv for {len(recomputed)} config(s) "
              f"(summary column was blank): {', '.join(recomputed)}\n")
    return out


def rank_metric(configs, metric):
    """1=best ranks with correct direction; configs within 1 se are averaged to a shared rank."""
    col, se_col, higher = METRICS[metric]
    items = [(n, v[metric][0], v[metric][1]) for n, v in configs.items() if metric in v]
    if not items:
        return {}
    items.sort(key=lambda t: t[1], reverse=higher)   # best first
    ranks = {}
    i = 0
    while i < len(items):
        j = i + 1
        # group configs statistically tied with items[i] (within its se, when se known)
        while j < len(items):
            n0, v0, s0 = items[i]
            nj, vj, sj = items[j]
            tol = (s0 or 0) + (sj or 0)
            if abs(vj - v0) <= tol and tol > 0:
                j += 1
            else:
                break
        shared = mean(range(i + 1, j + 1))   # average rank for the tie block
        for k in range(i, j):
            ranks[items[k][0]] = shared
        i = j
    return ranks


def minmax_scores(configs, metric):
    """Map metric to [0,1] with 1=best (direction-corrected). Constant metric -> all 1.0."""
    _, _, higher = METRICS[metric]
    vv = {n: v[metric][0] for n, v in configs.items() if metric in v}
    if not vv:
        return {}
    lo, hi = min(vv.values()), max(vv.values())
    if hi == lo:
        return {n: 1.0 for n in vv}
    return {n: ((x - lo) / (hi - lo) if higher else (hi - x) / (hi - lo)) for n, x in vv.items()}


def parse_kv(s, cast=float):
    d = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        d[k.strip()] = cast(v)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="ablations/ablations_for_figures",
                    help="dir whose immediate sub-dirs are the configs")
    ap.add_argument("--summary_name", default="leave_one_out_summary_reeval_last.csv")
    ap.add_argument("--group", default="extrapolation",
                    help="which summary row to use (substring match on the 'group' column)")
    ap.add_argument("--weights", default="lesion=0.5,recon=0.4,dice=0.1",
                    help="weights for weighted mean-rank / normalized-score (lesion,recon,dice)")
    ap.add_argument("--recon_tol", type=float, default=0.15,
                    help="guardrail: keep configs whose recon-composite score (min-max mean of "
                         "PSNR/SSIM/LPIPS, 1=best-in-field) is within this gap of the best. "
                         "0.15 ~ 'top ~85%% of the recon range'. Lower = stricter.")
    ap.add_argument("--floor", default="",
                    help="optional ABSOLUTE recon floors, e.g. 'psnr=17,ssim=0.55,lpips=0.45'; "
                         "a config must also clear these to pass the guardrail")
    args = ap.parse_args()

    w = parse_kv(args.weights)
    floor = parse_kv(args.floor) if args.floor else {}
    configs = load_configs(args.runs_dir, args.summary_name, args.group)
    if not configs:
        sys.exit(f"No configs found under {args.runs_dir} (summary={args.summary_name}).")
    print(f"Loaded {len(configs)} configs from {args.runs_dir} (group~='{args.group}')\n")

    # ---- per-metric ranks (1=best, se-aware ties) ----
    ranks = {m: rank_metric(configs, m) for m in METRICS}
    scores = {m: minmax_scores(configs, m) for m in METRICS}

    def recon_rank(n):
        rs = [ranks[m][n] for m in RECON if n in ranks[m]]
        return mean(rs) if rs else float("nan")

    def recon_score(n):
        ss = [scores[m][n] for m in RECON if n in scores[m]]
        return mean(ss) if ss else 0.0

    rows = []
    for n in configs:
        weighted_rank = (w.get("lesion", 0) * ranks["area_MAE"].get(n, float("nan"))
                         + w.get("recon", 0) * recon_rank(n)
                         + w.get("dice", 0) * ranks["DICE"].get(n, float("nan")))
        plain_metrics = ["DICE", "area_MAE", "PSNR", "SSIM", "LPIPS"]
        plain_rank = mean([ranks[m][n] for m in plain_metrics if n in ranks[m]])
        norm_score = (w.get("lesion", 0) * scores["area_MAE"].get(n, 0)
                      + w.get("recon", 0) * recon_score(n)
                      + w.get("dice", 0) * scores["DICE"].get(n, 0))
        rows.append({"name": n, "weighted_rank": weighted_rank, "plain_rank": plain_rank,
                     "norm_score": norm_score, **{m: configs[n].get(m, (None,))[0] for m in METRICS}})

    # ---- guardrail: recon-COMPOSITE within tol of the best (+ optional absolute floors) ----
    # Composite = min-max mean of PSNR/SSIM/LPIPS (1 = best in field). Using the composite (not a
    # per-metric AND) avoids demanding a config be near-best on all three at once, which no single
    # config is when the best PSNR / best LPIPS live in different configs.
    comp = {n: recon_score(n) for n in configs}
    best_comp = max(comp.values())
    def passes(n):
        if best_comp - comp[n] > args.recon_tol:
            return False
        v = configs[n]
        if "psnr" in floor and "PSNR" in v and v["PSNR"][0] < floor["psnr"]:
            return False
        if "ssim" in floor and "SSIM" in v and v["SSIM"][0] < floor["ssim"]:
            return False
        if "lpips" in floor and "LPIPS" in v and v["LPIPS"][0] > floor["lpips"]:
            return False
        return True
    for r in rows:
        r["recon_comp"] = comp[r["name"]]
        r["guardrail"] = "pass" if passes(r["name"]) else "FAIL"

    # ---- report ----
    def show(title, key, reverse=False, note=""):
        print(f"== {title} {note} ==")
        srt = sorted(rows, key=lambda r: (r[key] != r[key], r[key]), reverse=reverse)
        hdr = f'{"#":>2} {"config":24s} {"lesMAE":>7} {"PSNR":>6} {"SSIM":>6} {"LPIPS":>6} {"DICE":>6} {"guard":>5} {key:>12}'
        print(hdr); print("-" * len(hdr))
        for i, r in enumerate(srt, 1):
            def c(m, f):
                x = r.get(m); return (f % x) if x is not None else "  -  "
            print(f'{i:>2} {r["name"][:24]:24s} {c("area_MAE","%7.3f")} {c("PSNR","%6.2f")} '
                  f'{c("SSIM","%6.3f")} {c("LPIPS","%6.3f")} {c("DICE","%6.3f")} '
                  f'{r["guardrail"]:>5} {r[key]:>12.3f}')
        print()
        return srt

    n_pass = sum(1 for r in rows if r["guardrail"] == "pass")
    print(f"Recon guardrail: keep configs with recon-composite >= {best_comp - args.recon_tol:.3f} "
          f"(best {best_comp:.3f}, tol {args.recon_tol})"
          + (f" AND floors {floor}" if floor else "") + f"  -> {n_pass}/{len(rows)} pass\n")

    survivors = [r for r in rows if r["guardrail"] == "pass"]
    if survivors:
        prim = min(survivors, key=lambda r: (r["area_MAE"] is None, r["area_MAE"]))
        print(f">> RECOMMENDED (guardrail + lowest lesion-MAE): {prim['name']}  "
              f"(lesMAE={prim['area_MAE']:.3f}, PSNR={prim['PSNR']:.2f}, DICE={prim['DICE']:.3f})\n")
    else:
        print(">> No config passes the guardrail -- loosen --guardrail tolerances.\n")

    wr = show("WEIGHTED mean-rank", "weighted_rank", note=f"(weights {args.weights}; lower=better)")
    print(f">> weighted mean-rank winner: {wr[0]['name']}\n")
    pr = show("PLAIN mean-rank (naive Borda)", "plain_rank", note="(lower=better)")
    print(f">> plain mean-rank winner: {pr[0]['name']}\n")
    ns = show("NORMALIZED-score aggregation", "norm_score", reverse=True,
              note=f"(weights {args.weights}; higher=better)")
    print(f">> normalized-score winner: {ns[0]['name']}\n")


if __name__ == "__main__":
    main()
