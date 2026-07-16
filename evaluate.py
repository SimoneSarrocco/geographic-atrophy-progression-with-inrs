import os
import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import AtlasBuilder
from build_atlas import AtlasBuilder
from data_loading.dataset import validate_splits

def parse_args():
    parser = argparse.ArgumentParser(description="GAP-INR Standalone Validation & Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a checkpoint .pth file, OR a run directory (with --select_by).")
    parser.add_argument("--select_by", type=str, default=None, choices=["dice", "loss"],
                        help="If --checkpoint is a run DIR, pick checkpoint_best.pth (dice) or "
                             "checkpoint_best_loss.pth (loss). Ignored when --checkpoint is a file.")
    
    # Overrides for validation strategy
    parser.add_argument("--holdout_strategy", type=str, choices=["last", "specific", "leave_one_out", "none"],
                        help="Override validation holdout strategy")
    parser.add_argument("--holdout_visit", type=int,
                        help="Override validation holdout visit (1-indexed chronological position)")
    parser.add_argument("--independent_visits", type=str, choices=["true", "false"],
                        help="Override whether to treat each visit independently")
    parser.add_argument("--support_k", type=int,
                        help="Clinical forecast on the TEST set: fit the latent on the first K "
                             "chronological visits per eye and score ALL later visits (e.g. "
                             "--support_k 1 = fit on baseline, predict visits 2..N). Activates test().")
    parser.add_argument("--pairwise", action="store_true",
                        help="Per-PAIR forecast on the TEST set, matching ImageFlowNet's pairwise "
                             "eval EXACTLY: for every visit pair (a<b) per eye, fit a fresh latent on "
                             "ONLY the older visit a and predict the newer visit b. Scores interp/extrap "
                             "(b is/ isn't the eye's last visit) + minor/major GA growth, into "
                             "leave_one_out_summary.csv. Activates test(); overrides --support_k.")
    parser.add_argument("--skip_val", action="store_true",
                        help="Skip the validation routine and only run the test evaluation "
                             "(useful with --support_k to get just the forecast numbers).")
    parser.add_argument("--test", choices=["on", "off"], default=None,
                        help="Force the final TEST-set evaluation on/off, overriding the checkpoint's "
                             "config. 'on' runs test() (LOO interp+extrap on the held-out test eyes) "
                             "inside __init__; 'off' skips it. Default: use the checkpoint's test.activate.")
    
    # Overrides for optimization parameters
    parser.add_argument("--epochs_val", type=int, help="Override number of validation optimization epochs")
    parser.add_argument("--lr_latent", type=float, help="Override validation learning rate for latents")
    parser.add_argument("--val_latent_init", type=str, choices=["random", "nearest_train"],
                        help="Override validation latent initialization strategy")
    
    # Cross-model comparison dump (consumed by models/comparison/make_comparison_figure.py).
    parser.add_argument("--dump_root", type=str, default=None,
                        help="If set, write cross-model comparison .npz dumps under this root "
                             "(one per held-out test eye, target = its last visit).")
    parser.add_argument("--dump_scenario", type=str, choices=["matched", "full", "interp"], default="full",
                        help="'full' = GAP-INR at full capability (all n-1 visits, LOO-last); "
                             "'matched' = baseline-only, pair with --support_k 1; "
                             "'interp' = per-visit interior dumps (pair with --holdout_strategy "
                             "leave_one_out) for the interpolation comparison + trajectory figure.")
    parser.add_argument("--dump_method", type=str, default=None,
                        help="Comparison method key (default: gap_inr for full, gap_inr_k1 for matched).")
    parser.add_argument("--dump_newvisits_root", type=str, default=None,
                        help="If set, write NEW-time-point GAP-INR FAF/mask dumps (interpolated "
                             "midpoints + extrapolated future visits, no GT) under this root for "
                             "make_trajectory.py --new-root.")
    parser.add_argument("--dump_newvisits_scenario", type=str, default="static",
                        help="Scenario subdir for the new-visit dumps (default static; match "
                             "make_trajectory.py --new-scenario).")
    parser.add_argument("--dump_static_root", type=str, default=None,
                        help="If set, write Part-2 STATIC-segmentation dumps (one per observed test "
                             "visit) under this root for comparison vs NISF/MetaSeg.")
    parser.add_argument("--dump_static_method", type=str, default=None,
                        help="Static method key: gap_inr_pervisit (independent_visits) or "
                             "gap_inr_perpatient (one latent/eye). Default inferred from the config.")

    # Resolution / sampling-grid override (INR is resolution-agnostic -> eval at any grid, no retrain)
    parser.add_argument("--config_data", type=str, default=None,
                        help="Evaluate at a different dataset SECTION from configs/config_data.yaml "
                             "(swaps the INR sampling grid/resolution without retraining), e.g. "
                             "faf_ga_twovar_wktemporal_256 (extrap, IFN-matched) or _620 (interp).")

    # Output path
    parser.add_argument("--output_dir", type=str, help="Override output directory for logging and figures")
    parser.add_argument("--test_latents", type=str, default=None,
                        help="Path to a frozen test-latents file (.pt) for REPRODUCIBLE evaluation. If it "
                             "exists, the per-round optimised latents are RELOADED and the (non-deterministic) "
                             "test-time optimisation is skipped -> identical numbers every run. If it does NOT "
                             "exist, the TTO runs once and SAVES the latents to this path. Default "
                             "<output_dir>/test_latents.pt; pass 'off' to disable.")
    
    # Device
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Override computation device")
    
    return parser.parse_args()

def main():
    args_cmd = parse_args()

    # Resolve a run DIR + --select_by to the matching best checkpoint (dice vs combined loss).
    # The .pth may sit directly in the given dir OR nested in the training run's timestamped
    # subdir (e.g. <run>/faf_ga_twovar_wktemporal_512_<ts>_loc/checkpoint_best.pth), so fall back
    # to a recursive search (shallowest match wins) when it is not at the top level.
    if os.path.isdir(args_cmd.checkpoint) and args_cmd.select_by:
        _fname = "checkpoint_best_loss.pth" if args_cmd.select_by == "loss" else "checkpoint_best.pth"
        _direct = os.path.join(args_cmd.checkpoint, _fname)
        if os.path.exists(_direct):
            args_cmd.checkpoint = _direct
        else:
            import glob as _glob
            _hits = sorted(_glob.glob(os.path.join(args_cmd.checkpoint, "**", _fname), recursive=True),
                           key=lambda p: (p.count(os.sep), p))
            if not _hits:
                raise FileNotFoundError(
                    f"--select_by {args_cmd.select_by}: no {_fname} found under {args_cmd.checkpoint}")
            args_cmd.checkpoint = _hits[0]
        print(f"[select_by={args_cmd.select_by}] -> {args_cmd.checkpoint}")

    if not os.path.exists(args_cmd.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args_cmd.checkpoint}")

    print(f"Loading checkpoint from: {args_cmd.checkpoint}")
    chkp = torch.load(args_cmd.checkpoint, map_location="cpu", weights_only=False)
    
    # Extract the configuration args from the checkpoint
    args = chkp["args"]

    # (a) Resolution override FIRST (before other dataset overrides): swap the whole dataset section
    # to a different config_data.yaml entry, exactly like run.py --config_data. Lets a 512-trained INR
    # be evaluated at 256 (extrapolation, matching the IFN 256 run) or 620 (interpolation) with no
    # retraining. Splits are identical across the _256/_512/_620 sections, so no leakage.
    if args_cmd.config_data is not None:
        _cd_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config_data.yaml")
        with open(_cd_yaml) as _f:
            _sections = yaml.safe_load(_f)
        if args_cmd.config_data not in _sections:
            raise SystemExit(f"--config_data {args_cmd.config_data!r} not in {_cd_yaml} "
                             f"(have: {sorted(_sections)})")
        args["dataset"] = _sections[args_cmd.config_data]
        args["config_data"] = args_cmd.config_data
        # Resolve subject_ids from its YAML path into a dict, exactly like run.py does at load time.
        # The raw config_data section stores subject_ids as a PATH string; validate_splits (and the
        # loader) expect the resolved {train/val/test: [...]} dict. Without this, sids stays a str
        # and validate_splits crashes at `sids.get(...)`.
        _sids = args["dataset"].get("subject_ids")
        if isinstance(_sids, str):
            with open(_sids) as _sf:
                args["dataset"]["subject_ids"] = yaml.safe_load(_sf)[args["dataset"]["dataset_name"]]["subject_ids"]
        print(f"Overriding config_data -> {args_cmd.config_data} (dataset section swapped; "
              f"INR evaluated at this sampling grid)")

    # Apply CLI overrides if provided
    if args_cmd.holdout_strategy is not None:
        args["validation"]["holdout_strategy"] = args_cmd.holdout_strategy
        print(f"Overriding holdout_strategy -> {args_cmd.holdout_strategy}")
        
    if args_cmd.holdout_visit is not None:
        args["validation"]["holdout_visit"] = args_cmd.holdout_visit
        print(f"Overriding holdout_visit -> {args_cmd.holdout_visit}")
        
    if args_cmd.independent_visits is not None:
        val_val = args_cmd.independent_visits.lower() == "true"
        args["dataset"]["independent_visits"] = val_val
        print(f"Overriding independent_visits -> {val_val}")

    if args_cmd.support_k is not None:
        args.setdefault("test", {})
        args["test"]["support_k"] = args_cmd.support_k
        args["test"]["activate"] = True
        print(f"Overriding test.support_k -> {args_cmd.support_k} (test() will run the "
              f"first-{args_cmd.support_k}-visit forecast on the test set)")

    if args_cmd.pairwise:
        args.setdefault("test", {})
        args["test"]["pairwise"] = True
        args["test"]["activate"] = True
        print("Overriding test.pairwise -> True (test() will run the per-pair forecast on the "
              "test set, matching ImageFlowNet's pairwise eval)")

    if args_cmd.test is not None:
        args.setdefault("test", {})
        args["test"]["activate"] = (args_cmd.test == "on")
        print(f"Overriding test.activate -> {args['test']['activate']}")
        
    if args_cmd.epochs_val is not None:
        args["epochs"]["val"] = args_cmd.epochs_val
        print(f"Overriding epochs.val -> {args_cmd.epochs_val}")
        
    if args_cmd.lr_latent is not None:
        args["optimizer"]["lr_latent"] = args_cmd.lr_latent
        print(f"Overriding optimizer.lr_latent -> {args_cmd.lr_latent}")
        
    if args_cmd.val_latent_init is not None:
        args["optimizer"]["val_latent_init"] = args_cmd.val_latent_init
        print(f"Overriding optimizer.val_latent_init -> {args_cmd.val_latent_init}")
        
    if args_cmd.dump_root is not None:
        _method = args_cmd.dump_method or ("gap_inr_k1" if args_cmd.dump_scenario == "matched" else "gap_inr")
        args["comparison_dump"] = {"enable": True, "root": args_cmd.dump_root,
                                   "scenario": args_cmd.dump_scenario, "method": _method,
                                   "split": "test"}
        print(f"Comparison dump ENABLED -> root={args_cmd.dump_root} scenario={args_cmd.dump_scenario} "
              f"method={_method} (held-out test eyes, target=last visit)")

    if args_cmd.dump_newvisits_root is not None:
        _nvmethod = args_cmd.dump_method or ("gap_inr_k1" if args_cmd.dump_scenario == "matched" else "gap_inr")
        args["comparison_dump_newvisits"] = {"enable": True, "root": args_cmd.dump_newvisits_root,
                                             "scenario": args_cmd.dump_newvisits_scenario,
                                             "method": _nvmethod, "split": "test"}
        print(f"New-visit dump ENABLED -> root={args_cmd.dump_newvisits_root} "
              f"scenario={args_cmd.dump_newvisits_scenario} method={_nvmethod} "
              f"(interpolated + extrapolated new visits, no GT)")

    if args_cmd.dump_static_root is not None:
        _indep = args.get("dataset", {}).get("independent_visits", False)
        _smethod = args_cmd.dump_static_method or ("gap_inr_pervisit" if _indep else "gap_inr_perpatient")
        args["comparison_dump_static"] = {"enable": True, "root": args_cmd.dump_static_root,
                                          "method": _smethod, "split": "test"}
        print(f"Static-seg dump ENABLED -> root={args_cmd.dump_static_root} method={_smethod} "
              f"(every observed test visit)")

    if args_cmd.device is not None:
        args["device"] = args_cmd.device
    else:
        args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {args['device']}")

    # Split-integrity + checkpoint-vs-split LEAKAGE guard. validate_splits raises if the
    # current config's train/val/test overlap. The latent-bank check catches the dangerous case
    # where the checkpoint was TRAINED on a different split than we now evaluate on: for a per-eye
    # latent model the bank size == #train eyes, so a mismatch means the test set we're about to
    # score may include eyes the checkpoint trained on (leakage).
    sets = validate_splits(args)
    lat = chkp.get("latents")
    indep = args.get("dataset", {}).get("independent_visits", False)
    if lat is not None and not indep and len(sets.get("train", [])) and lat.shape[0] != len(sets["train"]):
        print("\n" + "*" * 78)
        print(f"*** LEAKAGE WARNING: checkpoint has {lat.shape[0]} per-eye latents, but the current "
              f"config\n*** resolves {len(sets['train'])} TRAIN eyes. This checkpoint was trained on a "
              f"DIFFERENT split,\n*** so test-set metrics may be INVALID (test eyes seen in training). "
              f"Re-train on the\n*** current split before reporting test numbers.")
        print("*" * 78 + "\n")

    # Setup directories
    chkp_dir = os.path.dirname(args_cmd.checkpoint)
    chkp_epoch = chkp["epoch"]
    
    if args_cmd.output_dir is not None:
        args["output_dir"] = args_cmd.output_dir
    else:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = args["validation"]["holdout_strategy"]
        eval_dirname = f"evaluation_ep{chkp_epoch}_{strategy}_joint_{time_stamp}"
        args["output_dir"] = os.path.join(chkp_dir, eval_dirname)
        
    os.makedirs(args["output_dir"], exist_ok=True)
    print(f"Output directory for evaluation results: {args['output_dir']}")

    # Frozen test-latents for REPRODUCIBLE evaluation: default <output_dir>/test_latents.pt.
    # exists -> load & skip TTO (deterministic); missing -> run TTO once & save. 'off' disables.
    if args_cmd.test_latents == "off":
        args["test_latents_mode"] = None
    else:
        _lat_path = args_cmd.test_latents or os.path.join(args["output_dir"], "test_latents.pt")
        args["test_latents_path"] = _lat_path
        args["test_latents_mode"] = "load" if os.path.exists(_lat_path) else "save"
        print(f"[reproducibility] test latents: {args['test_latents_mode'].upper()} <- {_lat_path}")

    # Save the modified config file for reproducibility
    eval_config_path = os.path.join(args["output_dir"], "config_evaluation.yaml")
    with open(eval_config_path, "w") as f:
        # We need to temporarily remove tb_writer if it exists because SummaryWriter is not serializable as YAML
        tb_writer_tmp = args.pop("tb_writer", None)
        yaml.dump(args, f)
        if tb_writer_tmp is not None:
            args["tb_writer"] = tb_writer_tmp
    print(f"Saved evaluation configuration to {eval_config_path}")
    
    # Setup TensorBoard SummaryWriter
    tb_log_dir = os.path.join(args["output_dir"], "tb_logs")
    args["tb_writer"] = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log directory: {tb_log_dir}")
    
    # Configure AtlasBuilder to load model and not run training
    args["epochs"]["train"] = 0
    args["validate_every"] = 1
    args["load_model"] = {
        "path": chkp_dir,
        "epoch": chkp_epoch
    }
    
    # Ensure validation is activated for standalone script run
    args["validation"]["activate"] = True
    
    print("\nInitializing model and data loading...")
    # NB: if test.activate is set (e.g. via --support_k), test() runs inside AtlasBuilder.__init__.
    atlas_builder = AtlasBuilder(args)

    if args_cmd.skip_val:
        print("\n--skip_val set: skipping the validation routine.")
    else:
        print("\nRunning validation/evaluation routine...")
        atlas_builder.validate(epoch_train=chkp_epoch)

    # Paper-ready leave-one-out summary (held-out DICE/PSNR/SSIM/IoU mean±SE, interp vs
    # extrapolation, lesion-area MAE) from the metric JSONs just written.
    try:
        import subprocess, sys
        here = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, os.path.join(here, "summarize_eval.py"),
               "--eval_dir", args["output_dir"]]
        # Pass the frozen data CSV so summarize_eval also emits the minor/major GA-growth buckets
        # (ImageFlowNet framing). Growth uses the CANONICAL 620->512 mask grid (summarize_eval's
        # defaults), identical to eval_omega, so the buckets are method-independent -- regardless of
        # whether this GAP-INR run itself scores at 620 or 512.
        _csv = (args.get("dataset") or {}).get("tsv_file")
        if _csv:
            cmd += ["--data_csv", _csv]
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[summary] could not build leave-one-out summary: {e}")

    # Clean up TensorBoard
    if args["tb_writer"] is not None:
        args["tb_writer"].close()
        
    print("\nEvaluation successfully completed!")

if __name__ == "__main__":
    main()
