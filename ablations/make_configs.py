#!/usr/bin/env python
"""Generate the GAP-INR ablation config suite from a single base config.

WHY a generator (not 20 hand-written YAMLs): every ablation is the SAME 620-crop,
short-training reference config with exactly ONE knob changed. Generating them from
one base keeps them in lockstep (change the base -> regenerate -> all ablations move
together) and makes the "what is being varied" explicit in one table below. The
emitted files are real, standalone config_atlas YAMLs you can also run by hand:

    python run.py --config_atlas ablations/configs/<name>.yaml

BASE = ablations/_base_620.yaml (frozen ablation reference) with the short-run overrides applied:
    epochs.train = 25         # cosine T_max == epochs.train, so 25 epochs ANNEAL FULLY
    validate_every = 1        # evaluate every epoch (the "does it even improve?" question)
    train_eval_every = 1
    output_dir = ./ablations/runs/<name>

Each ablation below is BASE + a small override dict. Cells equal to BASE (e.g. latent
64x32x32, sr_weight 10, omega 30, CE+Dice, FiLM, per-eye, aug off, cosine) are NOT
re-emitted -- they ARE the `baseline` run, which every group compares against.

Run:  python ablations/make_configs.py        # writes ablations/configs/*.yaml
"""
import copy
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# Frozen ablation BASE (moved out of configs/ so the main run uses only config_atlas.yaml +
# config_data.yaml). This is a self-contained merged config (config_data: faf_ga_620) kept here
# solely as the ablation reference; edit the ABLATIONS spec below, not this file.
BASE_CONFIG = os.path.join(HERE, "_base_620.yaml")
OUT_DIR = os.path.join(HERE, "configs")

# Overrides applied to EVERY ablation. 50 training epochs (cosine T_max == epochs.train, so it still
# anneals fully), validate every 5 epochs (validation TTO is expensive, so 5 keeps it tractable while
# still giving a usable val curve / checkpoint-selection cadence), and HOLD OUT THE LAST visit at
# validation (holdout 'last' = EXTRAPOLATION-only). This is ~4x FASTER per epoch than leave-one-out
# (1 held-out fold/eye vs every visit position) -> used for fast ablation SELECTION. NB: it is
# extrapolation-only; the interp+extrap leave_one_out protocol (matching comparison/omega_eval_spec.py)
# is reserved for the FINAL eval of the locked winner (evaluate.py --holdout_strategy leave_one_out,
# no retraining needed). (A0-length ablations set their OWN epochs.train.)
BASE_OVERRIDES = {
    "epochs.train": 50,
    "epochs.val": 25,
    "validate_every": 5,
    "validation.train_eval_every": 1,
    "validation.holdout_strategy": "last",
}

# STAGED ablation: first compare the three TEMPORAL mechanisms (Stage 1), then bake the WINNER into
# the base so every OTHER ablation (latent / omega / sr / ...) is built on top of it (Stage 2).
# `--temporal <name>` selects which mechanism the base uses. Applied to EVERY config before its own
# per-ablation override, so e.g. `a5_omega100` inherits the winning temporal mechanism in Stage 2.
#   scalar    -- FiLM modulation, RAW SCALAR condition (the standard; matches configs/config_atlas.yaml)
#   mlp       -- FiLM modulation, MLP-embedded condition (learned smooth embedding, dim 16; == a8_condmlp)
#   timeinput -- time as an INPUT coordinate, Fourier-encoded (== a1_timeinput_freq6)
TEMPORAL_PRESETS = {
    "scalar":    {},
    "mlp":       {"inr_decoder.cond_encoding": "mlp", "inr_decoder.cond_mlp_out": 16,
                  "inr_decoder.cond_mlp_hidden": 16},
    "timeinput": {"config_data": "faf_ga_timeinput_512", "inr_decoder.cond_dims": 0,
                  "inr_decoder.time_as_input": True, "inr_decoder.in_dim": 2,
                  "inr_decoder.time_num_frequencies": 6},
}

# (group, name, one-line rationale, override dict {dotted.key: value})
# A cell that would equal BASE is omitted -- `baseline` covers it.
ABLATIONS = [
    # --- A0: reference baseline (the LR-annealing / training-length ablations were removed -- not needed) ---
    ("A0_train", "baseline",      "620 FiLM per-eye, 25ep cosine, eval every epoch (REFERENCE)", {}),

    # --- A1: temporal mechanism (HEADLINE) FiLM vs time-as-input ---
    # FAIR version: time enters as a coordinate with a proper Fourier encoding (6 bands), NOT a bare
    # scalar. A bare scalar (time_num_frequencies=0) is a known degenerate config -- its magnitude is
    # tiny vs the coordinates so the SIREN ignores it and predictions collapse to identical-per-visit.
    # With 6 bands TimeEncoding.out_dim = 1 + 2*6 = 13, which the decoder AUTO-APPENDS to the SIREN
    # input (siren_in_dim += time_encoding.out_dim), so in_dim STAYS 2 (coords) -- do NOT bump in_dim
    # (in_dim=3 would make the SIREN expect 4 inputs while only 2+13 arrive). This gives time-as-input
    # a genuine chance, so "FiLM vs time-as-input" is a fair comparison. cond_dims -> 0 here.
    ("A1_temporal", "a1_timeinput_freq6",
        "time as INPUT coordinate with Fourier encoding (6 bands) -- FAIR vs FiLM",
        {"config_data": "faf_ga_timeinput_512",
         "inr_decoder.cond_dims": 0, "inr_decoder.time_as_input": True, "inr_decoder.in_dim": 2,
         "inr_decoder.time_num_frequencies": 6}),

    # (A2 per-visit latent, A3 latent spatial resolution, A4 recon-aux sr_weight REMOVED --
    #  A3 latent dims are swept in round 2; A4 sr_weight is ablated in round 3; A2 per-visit dropped.)

    # --- A5: SIREN frequency (boundary sharpness) ---
    ("A5_omega", "a5_omega100", "omega_0 = 100 (higher frequency)",
        {"inr_decoder.omega_0": 100, "inr_decoder.omega_start": 100, "inr_decoder.omega_end": 100}),

    # --- A6: segmentation loss ---
    ("A6_segloss", "a6_ce", "CE only (seg_dice_weight 0)", {"optimizer.seg_dice_weight": 0.0}),

    # --- A7: augmentation ---
    ("A7_aug", "a7_aug_on", "pseudo_eye augmentation ON", {"data_augmentation.activate": True}),

    # --- A8: FiLM-condition EMBEDDING. baseline feeds the temporal var (weeks_from_baseline) to FiLM
    #     as a RAW SCALAR (cond_encoding 'none'). This variant learns a small MLP embedding of it
    #     instead (cond_encoding 'mlp', output/hidden dim 16). Tests whether a richer learned embedding
    #     of the conditioning variable helps the FiLM modulation. cond_dims STAYS 1 (FiLM kept).
    ("A8_condembed", "a8_condmlp",
        "FiLM condition MLP-embedded (dim 16) vs raw-scalar baseline",
        {"inr_decoder.cond_encoding": "mlp", "inr_decoder.cond_mlp_out": 16,
         "inr_decoder.cond_mlp_hidden": 16}),

    # --- A9: TWO temporal variables (AgeatVisit + weeks_from_baseline), all RAW scalars (NO embedding).
    #     The data sections enable BOTH conditions; cond_dims is auto-computed in build_atlas._init_inr
    #     (= #enabled conditions, minus the temporal_condition when time_as_input=true). cond_encoding
    #     'none' + cond_num_frequencies 0 = raw FiLM scalar; time_num_frequencies 0 = raw time-input
    #     scalar (NB: a bare-scalar time-input has weak gradient vs the coords -- this is the explicit
    #     "no embedding" comparison the user asked for). in_dim STAYS 2 (decoder auto-appends the
    #     time-encoding out_dim, =1 when frequencies=0).
    ("A9_twovar", "a9_film_age_weeks",
        "BOTH AgeatVisit + weeks_from_baseline as FiLM (raw scalars), NO time-as-input",
        {"config_data": "faf_ga_twovar_wktemporal_512",
         "inr_decoder.time_as_input": False, "inr_decoder.in_dim": 2,
         "inr_decoder.time_num_frequencies": 0,
         "inr_decoder.cond_encoding": "none", "inr_decoder.cond_num_frequencies": 0}),
    ("A9_twovar", "a9_timeinput_weeks_film_age",
        "weeks_from_baseline as TIME-INPUT (raw), AgeatVisit as FiLM (raw)",
        {"config_data": "faf_ga_twovar_wktemporal_512",
         "inr_decoder.time_as_input": True, "inr_decoder.in_dim": 2,
         "inr_decoder.time_num_frequencies": 0,
         "inr_decoder.cond_encoding": "none", "inr_decoder.cond_num_frequencies": 0}),
    ("A9_twovar", "a9_timeinput_age_film_weeks",
        "AgeatVisit as TIME-INPUT (raw), weeks_from_baseline as FiLM (raw)",
        {"config_data": "faf_ga_twovar_agetemporal_512",
         "inr_decoder.time_as_input": True, "inr_decoder.in_dim": 2,
         "inr_decoder.time_num_frequencies": 0,
         "inr_decoder.cond_encoding": "none", "inr_decoder.cond_num_frequencies": 0}),
    # Round-2 candidate: combine the two round-1 leaders. a9_film_age_weeks (BOTH vars FiLM, best
    # areaMAE) injected RAW; a8_condmlp (best PSNR/SSIM/loss) used an MLP embedding but only on the
    # single weeks scalar. This is the untried cell: BOTH AgeatVisit + weeks_from_baseline as FiLM,
    # BOTH through a learned MLP embedding. cond_dims auto-computes to 2 (time_as_input=false ->
    # both conditions kept), so MLPConditionEncoding is Linear(2->16->16): a JOINT smooth embedding
    # of (age, weeks) feeding FiLM. cond_num_frequencies 0 (irrelevant for 'mlp' kind).
    ("A9_twovar", "a9_film_age_weeks_mlp",
        "BOTH AgeatVisit + weeks_from_baseline as FiLM, BOTH MLP-embedded (dim 16), NO time-as-input",
        {"config_data": "faf_ga_twovar_wktemporal_512",
         "inr_decoder.time_as_input": False, "inr_decoder.in_dim": 2,
         "inr_decoder.time_num_frequencies": 0,
         "inr_decoder.cond_encoding": "mlp", "inr_decoder.cond_mlp_out": 16,
         "inr_decoder.cond_mlp_hidden": 16, "inr_decoder.cond_num_frequencies": 0}),
]


# ============================ ROUND 2 (a9-base: latent sweep + augmentation) ============================
# Round-1 winner = a9_film_age_weeks: BOTH AgeatVisit + weeks_from_baseline as RAW-SCALAR FiLM (no MLP
# embedding, no time-as-input), latent 64x32x32. The full-metric reeval (comparison_reeval_last.csv)
# showed raw-scalar FiLM beats the shared-MLP variant (a9_film_age_weeks_mlp) on DICE (0.884 vs 0.881),
# LPIPS (0.359 vs 0.399) and areaMAE (0.220 vs 0.289), so Round 2 is rebuilt on the RAW-SCALAR base.
# Round 2 BAKES that into the base (A9_BASE) and then, one knob at a time: (a) raw-scalar vs separate-MLP
# embedding, (b) pseudo-eye augmentation, (c) the latent SPATIAL sweep (channels fixed), (d) the latent
# CHANNEL sweep (spatial fixed). Omega moved to ROUND 4. Same short 50-epoch protocol as round 1.
# Generated with `make_configs.py --round2`; emits ONLY these r2_* configs (round-1 untouched) and
# MERGES their labels into manifest.yaml so compare_ablations.py groups them correctly.
A9_BASE = {
    "config_data": "faf_ga_twovar_wktemporal_512",   # enables BOTH AgeatVisit + weeks_from_baseline
    "inr_decoder.time_as_input": False,             # both vars go to FiLM (no time-as-coordinate)
    "inr_decoder.in_dim": 2,
    "inr_decoder.time_num_frequencies": 0,
    "inr_decoder.cond_encoding": "none",            # RAW-SCALAR FiLM (round-1 winner a9_film_age_weeks)
    "inr_decoder.cond_num_frequencies": 0,
    "inr_decoder.latent_dim": [64, 32, 32],         # round-1 winner latent (round-2 sweeps this)
}
ROUND2 = [
    ("R2_a9base", "r2_a9_base",
        "a9 raw-scalar FiLM (age+weeks), latent 64x32x32 -- ROUND-2 REFERENCE (== round-1 winner)", {}),

    # (r2_a9_separated_mlp conditioning-MLP variant and r2_a9_base_augmented augmentation variant REMOVED.)

    # --- latent SPATIAL resolution sweep (channels fixed at 64; base spatial = 32x32) ---
    ("R2_latent", "r2_a9_latent8",   "latent 64x8x8 (coarse)",   {"inr_decoder.latent_dim": [64, 8, 8]}),
    ("R2_latent", "r2_a9_latent16",  "latent 64x16x16",          {"inr_decoder.latent_dim": [64, 16, 16]}),
    ("R2_latent", "r2_a9_latent64",  "latent 64x64x64",          {"inr_decoder.latent_dim": [64, 64, 64]}),
    ("R2_latent", "r2_a9_latent128", "latent 64x128x128 (fine)", {"inr_decoder.latent_dim": [64, 128, 128]}),

    # --- latent CHANNEL sweep (spatial fixed at 32x32; base channels = 64) ---
    ("R2_chan", "r2_a9_chan16",  "latent 16x32x32 (16 channels)",   {"inr_decoder.latent_dim": [16, 32, 32]}),
    ("R2_chan", "r2_a9_chan32",  "latent 32x32x32 (32 channels)",   {"inr_decoder.latent_dim": [32, 32, 32]}),
    ("R2_chan", "r2_a9_chan128", "latent 128x32x32 (128 channels)", {"inr_decoder.latent_dim": [128, 32, 32]}),
    ("R2_chan", "r2_a9_chan256", "latent 256x32x32 (256 channels)", {"inr_decoder.latent_dim": [256, 32, 32]}),
    ("R2_chan", "r2_a9_chan512", "latent 512x32x32 (512 channels)", {"inr_decoder.latent_dim": [512, 32, 32]}),
]


# ======================= FULL LATENT GRID (channels x spatial, 6x6 = 36 cells) =======================
# The two round-2 sweeps only cover a CROSS through the base cell (channel column @ spatial 32, spatial
# row @ 64 channels) = 10 of the 36 (channels x spatial) combinations. This emits the OTHER 26 cells so
# the whole 6x6 grid is trained and can be shown as a channels-vs-spatial heatmap / scatter. The S=1 row
# is the GLOBAL-VECTOR ablation (grid vs single vector per eye) across every channel width -- moved here
# from round 3 so it is swept over channels rather than pinned at 256. Same A9_BASE (raw-scalar FiLM
# age+weeks) + short 50-epoch protocol as round 2; ONLY the latent_dim changes. Names r2_a9_grid_c{C}_s{S}
# MUST match ablation_metrics.grid_run_name(). Cells already trained by the round-2 sweeps
# (C==64 at S in {8,16,32,64,128}, or S==32) are skipped -> exactly 26 configs.
# Generated with `make_configs.py --gridsweep`: emits ONLY these grid cells, manifest merged.
GRID_CHANNELS = [16, 32, 64, 128, 256, 512]
GRID_SPATIAL = [1, 8, 16, 32, 64, 128]   # S=1 == GLOBAL VECTOR per eye (no spatial grid), swept over all channels
# Cells already trained by the two round-2 sweeps: the SPATIAL sweep is at C=64 over S in {8,16,32,64,128}
# (it never trained S=1), and the CHANNEL sweep is at S=32 over all C. Everything else -- including the whole
# S=1 column (the global-vector ablation across channel widths, moved here from round 3) -- is emitted here.
_TRAINED_SPATIAL_AT_C64 = {8, 16, 32, 64, 128}
GRIDSWEEP = [
    ("R_grid", f"r2_a9_grid_c{C}_s{S}",
     f"latent {C}x{S}x{S} (full channels x spatial grid cell)",
     {"inr_decoder.latent_dim": [C, S, S]})
    for C in GRID_CHANNELS for S in GRID_SPATIAL
    if not ((C == 64 and S in _TRAINED_SPATIAL_AT_C64) or S == 32)
]


# ======================= ROUND 3 (architectural ablations on the round-2 winner) =======================
# Headline ablations that demonstrate WHY GAP-INR works, one-knob-at-a-time on the round-2 WINNER.
# >>> EDIT R3_BASE['inr_decoder.latent_dim'] to the round-2 winning latent once you pick it; it defaults
#     to the round-1 winner latent 64x32x32 as a placeholder. <<<
# Generated with `make_configs.py --round3`: emits ONLY r3_* configs (rounds 1-2 untouched), manifest merged.
# R3_BASE inherits the Round-2 winner: raw-scalar FiLM (now baked into A9_BASE) + the winning latent.
# >>> The latent below is PROVISIONAL: 64x64x64 won the FIRST (MLP-based) round-2 sweep; UPDATE it to
#     whatever wins the RAW-SCALAR re-run of round 2 before generating round 3. <<<
R3_BASE = {
    **A9_BASE,                                  # raw-scalar FiLM (age+weeks) -- round-1/2 winner conditioning
    "inr_decoder.latent_dim": [256, 32, 32],    # round-2 winner r2_a9_chan256 (256x32x32)
}
ROUND3 = [
    ("R3_base", "r3_base",
        "raw-scalar FiLM (age+weeks) + latent 256x32x32 -- ROUND-3 REFERENCE (== r2_a9_chan256)", {}),

    # (1) GRID vs GLOBAL-VECTOR latent -- the core GAP-INR contribution + why NISF/Neonatal-INR fail.
    #     MOVED to the grid sweep (S=1 row): the global-vector ablation is now swept over ALL channel
    #     widths (make_configs.py --gridsweep, cells r2_a9_grid_c{C}_s1) instead of pinned at 256 here.
    #     This shows the collapse is intrinsic to the missing spatial grid, not a channel-capacity artifact.

    # (2) TEST-TIME OPTIMIZATION: seg-supervised TTO (default) vs RECON-ONLY (NISF 'fit pixels, get labels').
    ("R3_tto", "r3_recon_only_tto", "recon-only test-time latent fit (seg_loss_val OFF) -- NISF-style",
        {"optimizer.seg_loss_val": False}),

    # (3) MULTI-TASK: does the FAF-reconstruction auxiliary help GA segmentation?  (sr_weight 10 -> 1 -> 0)
    ("R3_recon_aux", "r3_sr0_segonly", "seg-only (sr_weight 0, no recon auxiliary)",
        {"optimizer.sr_weight": 0.0}),
    ("R3_recon_aux", "r3_sr1_lightrecon", "light recon auxiliary (sr_weight 1 vs baseline 10)",
        {"optimizer.sr_weight": 1.0}),

    # (4) OUTPUT HEAD: separate seg head (default) vs the ORIGINAL shared-output decoder (seg + recon share
    #     the same final output layer / features, out_dim [1,2]).
    ("R3_head", "r3_shared_output", "shared-output decoder (seg from the recon output features) vs separate seg head",
        {"inr_decoder.shared_output_layer": True}),

    # (5) TEMPORAL DOMAIN-PRIOR losses (Lachinov et al. TMI, adapted to the INR). batch_by_eye groups
    #     an eye's visits into one batch so the pooled soft-Dice becomes the TEMPORAL/stacked Dice
    #     (their Eq. 6); the monotonicity penalty softly enforces non-decreasing predicted GA over time
    #     (their phi_dot>=0). Both are GA-irreversibility priors -> should help area-MAE + monotonicity.
    ("R3_temporal", "r3_temporal_dice",
        "TEMPORAL (stacked) Dice over an eye's whole visit sequence (batch_by_eye; Lachinov Eq. 6)",
        {"optimizer.batch_by_eye": True}),
    ("R3_temporal", "r3_mono_penalty",
        "monotonicity penalty (non-decreasing GA over time) + batch_by_eye (soft phi_dot>=0)",
        {"optimizer.batch_by_eye": True, "optimizer.mono_penalty.activate": True}),
]


# ======================= ROUND 4 (SIREN omega schedule on the round-3 winner) =======================
# Final lever: the per-LAYER SIREN frequency schedule (models/omega_scheduler.py). The baseline is
# 'constant' omega 30 (every layer). Tests whether a uniformly higher frequency, or progressively
# sharpening frequencies with depth, recovers high-frequency FAF/GA-boundary detail. omega_0 is the
# input-layer frequency; omega_start->omega_end is interpolated across the hidden layers.
# >>> R4_BASE inherits the round-3 winner; edit R3_BASE's latent first, then regenerate (--round4). <<<
# Generated with `make_configs.py --round4`: emits ONLY r4_* configs (rounds 1-3 untouched), manifest merged.
R4_BASE = {
    **R3_BASE,                                  # round-3 winner (a9 + winning latent + winning arch knobs)
}
ROUND4 = [
    ("R4_omega", "r4_omega_base", "round-3 winner, omega CONSTANT 30 -- ROUND-4 REFERENCE", {}),
    ("R4_omega", "r4_omega_constant100",
        "omega CONSTANT 100 (uniformly higher frequency, every layer)",
        {"inr_decoder.schedule_type": "constant",
         "inr_decoder.omega_0": 100, "inr_decoder.omega_start": 100, "inr_decoder.omega_end": 100}),
    ("R4_omega", "r4_omega_linear_30_100",
        "omega LINEAR schedule 30->100 across depth (coarse early layers, sharp late layers)",
        {"inr_decoder.schedule_type": "linear",
         "inr_decoder.omega_0": 30, "inr_decoder.omega_start": 30, "inr_decoder.omega_end": 100}),
    ("R4_omega", "r4_omega_exp_30_100",
        "omega EXPONENTIAL schedule 30->100 across depth",
        {"inr_decoder.schedule_type": "exponential",
         "inr_decoder.omega_0": 30, "inr_decoder.omega_start": 30, "inr_decoder.omega_end": 100}),
]


def _set_dotted(d, dotted, value):
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"override key {dotted!r} not present in base config (typo?)")
    node[keys[-1]] = value


def build(temporal="scalar", round2=False, round3=False, round4=False, gridsweep=False):
    if temporal not in TEMPORAL_PRESETS:
        raise SystemExit(f"--temporal must be one of {list(TEMPORAL_PRESETS)}; got {temporal!r}")
    temporal_base = TEMPORAL_PRESETS[temporal]
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Each later round emits ONLY its own r*_ configs (earlier rounds untouched) and bakes a winning
    # base into every config before the per-ablation knob.
    if gridsweep:
        spec, base_extra, extra_label = GRIDSWEEP, A9_BASE, " +a9_base" + str(A9_BASE)
    elif round4:
        spec, base_extra, extra_label = ROUND4, R4_BASE, " +r4_base" + str(R4_BASE)
    elif round3:
        spec, base_extra, extra_label = ROUND3, R3_BASE, " +r3_base" + str(R3_BASE)
    elif round2:
        spec, base_extra, extra_label = ROUND2, A9_BASE, " +a9_base" + str(A9_BASE)
    else:
        spec, base_extra, extra_label = ABLATIONS, {}, ""
    merge = round2 or round3 or round4 or gridsweep

    manifest = []
    for group, name, rationale, overrides in spec:
        cfg = copy.deepcopy(base)
        for k, v in BASE_OVERRIDES.items():
            _set_dotted(cfg, k, v)
        # Stage-2 temporal mechanism baked into the base (no-op for 'scalar'); the per-ablation
        # override below still wins for the temporal-trio configs themselves.
        for k, v in temporal_base.items():
            _set_dotted(cfg, k, v)
        # Round-2: a8 conditioning baked into every config (before per-ablation latent/omega knob).
        for k, v in base_extra.items():
            _set_dotted(cfg, k, v)
        for k, v in overrides.items():
            _set_dotted(cfg, k, v)
        cfg["output_dir"] = f"./ablations/runs/{name}"

        out_path = os.path.join(OUT_DIR, f"{name}.yaml")
        base_note = (str(temporal_base) if temporal != "scalar" else "")
        base_note = (base_note + extra_label).strip()
        header = (f"# ABLATION {group} :: {name}\n"
                  f"# {rationale}\n"
                  f"# Temporal base: {temporal}{(' ' + base_note) if base_note else ''}\n"
                  f"# Override vs baseline: {overrides if overrides else '(none -- this IS the base)'}\n"
                  f"# Auto-generated by ablations/make_configs.py -- edit the spec there, not here.\n")
        with open(out_path, "w") as f:
            f.write(header)
            yaml.dump(cfg, f, sort_keys=True)
        manifest.append({"group": group, "name": name, "rationale": rationale,
                         "config": os.path.relpath(out_path, REPO_ROOT),
                         "config_data": cfg["config_data"], "temporal_base": temporal})
        print(f"  wrote {out_path}")

    # Round-2 MERGES into the existing manifest (preserve round-1 labels); round-1 overwrites it.
    manifest_path = os.path.join(HERE, "manifest.yaml")
    if merge and os.path.exists(manifest_path):
        existing = {m["name"]: m for m in (yaml.safe_load(open(manifest_path)) or [])}
        for entry in manifest:
            existing[entry["name"]] = entry
        manifest_out = list(existing.values())
    else:
        manifest_out = manifest
    with open(manifest_path, "w") as f:
        yaml.dump(manifest_out, f, sort_keys=False)
    tag = ("full latent grid (channels x spatial)" if gridsweep else
           "round-4 omega-schedule sweep" if round4 else
           "round-3 architectural ablations" if round3 else
           "round-2 a9-base sweep" if round2 else f"temporal base = '{temporal}'")
    print(f"\n{len(manifest)} configs written to {OUT_DIR}  ({tag})")
    if gridsweep:
        print("  Grid run (pick free GPUs):\n"
              "      python ablations/run_ablations.py --gpus 0,1,2,3 "
              + " ".join(n for _, n, _, _ in GRIDSWEEP))
    elif round2 or round3 or round4:
        _names = ROUND4 if round4 else ROUND3 if round3 else ROUND2
        _rnum = 4 if round4 else 3 if round3 else 2
        print(f"  Round-{_rnum} run (pick free GPUs):\n"
              "      python ablations/run_ablations.py --gpus 0,1,2,3 "
              + " ".join(n for _, n, _, _ in _names))
    elif temporal != "scalar":
        print("  NB: 'baseline' now encodes the winning temporal mechanism. Archive Stage-1 runs first:\n"
              "      mv ablations/runs ablations/runs_stage1_temporal   (then run the remaining ablations)")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate GAP-INR ablation configs.")
    ap.add_argument("--temporal", default="scalar", choices=list(TEMPORAL_PRESETS),
                    help="temporal mechanism baked into the base (Stage-2: set to the Stage-1 winner). "
                         "scalar=raw-scalar FiLM (default), mlp=MLP-embedded FiLM (dim 16), "
                         "timeinput=Fourier time-as-input.")
    ap.add_argument("--round2", action="store_true",
                    help="emit ONLY the round-2 a9-base sweep (separate-MLP, augmentation, latent spatial + "
                         "channel sweeps) on the round-1 winner a9_film_age_weeks_mlp; round-1 untouched, merged.")
    ap.add_argument("--round3", action="store_true",
                    help="emit ONLY the round-3 architectural ablations (grid-vs-vector latent, recon-only "
                         "TTO, sr_weight, shared-output head) on the round-2 winner. EDIT R3_BASE latent to "
                         "the winner first. rounds 1-2 untouched, manifest merged.")
    ap.add_argument("--round4", action="store_true",
                    help="emit ONLY the round-4 omega-schedule sweep (constant100, linear/exponential 30->100) "
                         "on the round-3 winner. EDIT R3_BASE latent first. rounds 1-3 untouched, merged.")
    ap.add_argument("--gridsweep", action="store_true",
                    help="emit ONLY the 20 remaining full-grid latent cells (channels x spatial, the 6x5 "
                         "grid minus the 10 already covered by the round-2 cross) on A9_BASE. Merged.")
    args = ap.parse_args()
    build(temporal=args.temporal, round2=args.round2, round3=args.round3, round4=args.round4,
          gridsweep=args.gridsweep)
