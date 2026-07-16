"""
Authoritatively verify what a GAP-INR run actually used — from the CHECKPOINT, not the
run-dir .yaml (which can be stale/overwritten). The checkpoint stores the resolved `args`
the code ran with AND the real `latents` tensor, so it cannot lie.

The decisive test for "one latent per eye vs per visit" is data-grounded:
    n_latents == n_eyes   -> ONE LATENT PER EYE
    n_latents == n_visits -> ONE LATENT PER VISIT

Usage:
    python verify_run_config.py --checkpoint tmp/<run>/checkpoint_epoch_49.pth
    python verify_run_config.py --run tmp/<run>            # auto-pick latest checkpoint
"""
import argparse, glob, os, re
import torch


def latest_ckpt(run_dir):
    cks = glob.glob(os.path.join(run_dir, "checkpoint_epoch_*.pth"))
    if not cks:
        cks = glob.glob(os.path.join(run_dir, "checkpoint_*.pth"))
    if not cks:
        raise FileNotFoundError(f"No checkpoint_*.pth in {run_dir}")
    return max(cks, key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
               if re.search(r"(\d+)", os.path.basename(p)) else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--run", help="run dir; auto-picks the latest checkpoint")
    ap.add_argument("--id_col", default="Eye_ID")
    a = ap.parse_args()
    ckpt = a.checkpoint or latest_ckpt(a.run)
    print(f"checkpoint: {ckpt}\n")

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    args, df, lat = d["args"], d["dataset_df"], d["latents"]
    idec, dcfg, opt = args["inr_decoder"], args["dataset"], args["optimizer"]

    n_lat = lat.shape[0]
    grid = tuple(lat.shape[1:])
    n_visits = len(df)
    id_col = a.id_col if a.id_col in df.columns else dcfg.get("id_column", a.id_col)
    n_eyes = df[id_col].nunique() if id_col in df.columns else -1

    print("=== GROUND TRUTH (from the tensors) ===")
    print(f"  latents tensor shape : {tuple(lat.shape)}  -> {n_lat} latents, each grid {grid}")
    print(f"  visits in df         : {n_visits}")
    print(f"  unique eyes ({id_col}): {n_eyes}")
    if n_lat == n_eyes and n_eyes != n_visits:
        verdict = "ONE LATENT PER EYE"
    elif n_lat == n_visits and n_eyes != n_visits:
        verdict = "ONE LATENT PER VISIT"
    elif n_lat == n_eyes == n_visits:
        verdict = "AMBIGUOUS (n_eyes == n_visits; can't tell from counts)"
    else:
        verdict = f"UNEXPECTED (n_lat={n_lat} matches neither eyes nor visits)"
    print(f"  >>> VERDICT: {verdict}\n")

    print("=== RESOLVED SETTINGS (what the run actually used) ===")
    conds_on = [k for k, v in dcfg.get("conditions", {}).items() if v]
    print(f"  independent_visits   : {dcfg.get('independent_visits')}   (flag; verdict above is the real check)")
    print(f"  latent_dim (config)  : {idec.get('latent_dim')}   (channels+spatial; matches grid? {list(idec.get('latent_dim', [])) == [lat.shape[1], *lat.shape[2:]]})")
    print(f"  conditions ENABLED   : {conds_on or 'NONE'}")
    print(f"  cond_dims            : {idec.get('cond_dims')}   cond_encoding: {idec.get('cond_encoding')}")
    print(f"  temporal_condition   : {dcfg.get('temporal_condition')}")
    print(f"  time_as_input        : {idec.get('time_as_input')}   time_encoding: {idec.get('time_encoding')}")
    print(f"  sr_weight / seg_weight / seg_dice_weight : {opt.get('sr_weight')} / {opt.get('seg_weight')} / {opt.get('seg_dice_weight')}")
    print(f"  seg_loss_val         : {opt.get('seg_loss_val')}   val_latent_init: {opt.get('val_latent_init')}")
    print(f"  holdout_strategy     : {args.get('validation', {}).get('holdout_strategy')}")
    print(f"  omega 0/start/end    : {idec.get('omega_0')}/{idec.get('omega_start')}/{idec.get('omega_end')}  ({idec.get('schedule_type')})")
    print(f"  hidden_size / layers : {idec.get('hidden_size')} / {idec.get('num_hidden_layers')}")
    print(f"  config_data section  : {args.get('config_data')}   dataset_name: {dcfg.get('dataset_name')}")
    print(f"  epoch saved          : {d.get('epoch')}")

    # --- how is the conditioning variable modulated? (raw scalar / MLP embedding / Fourier) ---
    print("\n=== CONDITION MODULATION (how weeks/etc. enters) ===")
    sd = d.get("inr_decoder", {})
    cond_keys = [k for k in sd if "cond" in k.lower()]          # MLP/Fourier encoders add params; raw adds none
    lat_w = next((sd[k] for k in sd if k.endswith("linear_lats.weight")), None)
    lat_ch = int(idec.get("latent_dim", [0])[0])
    if lat_w is not None and lat_ch:
        cond_out = int(lat_w.shape[1]) - lat_ch - (2 if idec.get("faf_as_input") else 0)
        if cond_out <= 0:
            kind = "NONE (no conditioning fed to modulation)"
        elif cond_keys:
            kind = f"MLP EMBEDDING (learned; out_dim={cond_out})"
        elif cond_out == idec.get("cond_dims", 1):
            kind = f"RAW SCALAR (no embedding; {cond_out} value(s) passed through)"
        else:
            kind = f"FOURIER features (out_dim={cond_out})"
        print(f"  modulation input width : {lat_w.shape[1]} = {lat_ch} latent + {cond_out} cond"
              + (" + 2 anchor" if idec.get("faf_as_input") else ""))
        print(f"  cond_encoding (arg)    : {idec.get('cond_encoding')}   (cond_mlp_out={idec.get('cond_mlp_out')} is UNUSED unless 'mlp')")
        print(f"  learnable cond params  : {cond_keys or 'NONE'}")
        print(f"  >>> WEEKS MODULATED AS : {kind}")
        if idec.get("time_as_input"):
            print("  NOTE: time_as_input=True -> time ALSO enters as an INPUT COORDINATE, not (only) modulation")

    # consistency assertions (warn, don't crash)
    print("\n=== CONSISTENCY CHECKS ===")
    ok = list(idec.get("latent_dim", [])) == [lat.shape[1], *lat.shape[2:]]
    print(f"  [{'PASS' if ok else 'FAIL'}] config latent_dim matches actual latents tensor")
    flag_says_visit = bool(dcfg.get("independent_visits"))
    data_says_visit = (n_lat == n_visits and n_eyes != n_visits)
    cons = (flag_says_visit == data_says_visit) or (n_eyes == n_visits)
    print(f"  [{'PASS' if cons else 'FAIL'}] independent_visits flag ({flag_says_visit}) agrees with the tensor counts")


if __name__ == "__main__":
    main()
