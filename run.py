import os
from datetime import datetime
import yaml
import argparse
import wandb as wd
from torch.utils.tensorboard import SummaryWriter
from build_atlas import AtlasBuilder
from data_loading.dataset import validate_splits
from download_lakefs_data import download_dataset
os.environ["WANDB__SERVICE_WAIT"] = "500"

# Repo root (this file's directory), so default config paths resolve regardless of the CWD.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_DATA = os.path.join(_REPO_ROOT, 'configs', 'config_data.yaml')


def initial_setup(cmd_args=None):
    atlas_path = (cmd_args or {}).get('config_atlas') or \
        os.path.join(_REPO_ROOT, 'configs', 'config_atlas.yaml')
    print(f"Loading atlas config from: {atlas_path}")
    with open(atlas_path, 'r') as stream:
        args_atlas = yaml.safe_load(stream)
    with open(_DEFAULT_CONFIG_DATA, 'r') as stream:
        # Precedence: explicit --config_data on the CLI wins; otherwise use the `config_data`
        # field INSIDE the chosen --config_atlas file (so per-config data sections like
        # faf_ga_768 / faf_ga_indep / faf_ga_timeinput are actually honored). Falls back to 'faf_ga'.
        config_data = (cmd_args or {}).get('config_data') or args_atlas.get('config_data', 'faf_ga')
        args_data = {'dataset': yaml.safe_load(stream)[config_data]}
    args = {**args_data, **args_atlas}
    with open(args['dataset']['subject_ids'], 'r') as stream:
        args['dataset']['subject_ids'] = yaml.safe_load(stream)[args['dataset']['dataset_name']]['subject_ids']
    if cmd_args is not None:
        args = override_args(args, cmd_args)

    # Guarantee train/val/test are disjoint + leakage-free BEFORE any training starts.
    validate_splits(args)

    print(f"\n--- Initiating Multi-threaded LakeFS Pre-fetch for {config_data} ---")
    try:
        download_dataset(config_data, config_path=_DEFAULT_CONFIG_DATA, num_workers=16)
    except Exception as e:
        print(f"Pre-fetch encountered an issue (will fallback to sequential): {e}")
    print("----------------------------------------------------------------\n")

    job_id = os.getenv("SLURM_JOB_ID", "loc")[-3:]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dsetup = args['config_data']

    run_name =f"{dsetup}_{time_stamp}_{job_id}"
    args['output_dir'] = f"{args['output_dir']}/{run_name}"
    os.makedirs(args['output_dir'], exist_ok=True)
    print(f"Output directory: {args['output_dir']}")

    # save config files
    with open(os.path.join(args['output_dir'], 'config_data.yaml'), 'w') as f:
        yaml.dump(args_data, f)
    with open(os.path.join(args['output_dir'], 'config_atlas.yaml'), 'w') as f:
        yaml.dump(args_atlas, f)
    print(f"Saved config files to {args['output_dir']}")

    has_seg = args['inr_decoder']['out_dim'][-1] > 0
    expected_sr_mods = len(args['dataset']['modalities']) - 1 if has_seg else len(args['dataset']['modalities'])
    if args['inr_decoder']['out_dim'][0] != expected_sr_mods:
        print(f"WARNING: The number of output dimensions ({args['inr_decoder']['out_dim'][0]}) " 
              f"might not match the number of intensity modalities ({expected_sr_mods}).")
    if args['atlas_gen']['conditions'] is not None: # check if all atlas conditions are set True in the dataset config
        for key in list(args['atlas_gen']['conditions'].keys()):
            if not args['dataset']['conditions'][key]:
                print(f"WARNING: The atlas condition {key} is not set True in the dataset config."
                    f"Turning off the atlas generation for {key}.")
                args['atlas_gen']['conditions'].pop(key)

    if args['logging']: # init weights and biases if logging is True
        wd.init(config=args, project=args['project_name'], 
                                entity=args['wandb_entity'], name=run_name)
    
    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(args['output_dir'], 'tb_logs')
    args['tb_writer'] = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log directory: {tb_log_dir}")
    
    return args


def override_args(config_args, cmd_args):
    for key, value in cmd_args.items():
        key1, key2 = key.split("__") if "__" in key else (key, None)
        if key2 is None:
            if value is not None:
                config_args[key] = value
        else:
            if value is not None:
                config_args[key1][key2] = value
    return config_args


def parse_cmd_args():
    parser = argparse.ArgumentParser(description="GAP-INR Atlas Builder")
    parser.add_argument("--config_data", type=str, default=None,
                        help="Override the dataset section in config_data.yaml. If omitted, the "
                             "config_data field inside the --config_atlas file is used.")
    parser.add_argument("--config_atlas", type=str, help="Path to a config_atlas YAML (default: configs/config_atlas.yaml)")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--inr_decoder__out_dim", type=int, nargs='+', help="Number of output dimensions [#modalities, #classes of segmentation]")
    parser.add_argument("--inr_decoder__tf_dim", type=int, help="Degrees of freedom for the transformation")
    parser.add_argument("--inr_decoder__cnn_kernel_size", type=int, help="Kernel size for the CNN for spatial modulation")
    parser.add_argument("--inr_decoder__latent_dim", type=int, nargs='+', help="Latent dimension [c,x,y,z]")
    parser.add_argument("--inr_decoder__hidden_size", type=int, help="Hidden size of the sr network")
    parser.add_argument("--inr_decoder__num_hidden_layers", type=int, help="Number of hidden layers of the sr network")
    parser.add_argument("--inr_decoder__modulated_layers", type=int, nargs='+', help="Modulated layers")
    parser.add_argument("--atlas_gen__cond_scale", type=float, help="Scale of the condition vector")
    parser.add_argument("--n_subjects__train", type=int, help="Number of subjects to use for training")
    parser.add_argument("--n_subjects__val", type=int, help="Number of subjects to use for validation")
    parser.add_argument("--overfit", action="store_true", help="Use the same subjects for training and validation")
    parser.add_argument("--overfit_subject_id", type=str, help="Specific subject ID to overfit on")
    parser.add_argument("--overfit_eye_laterality", type=str, help="Specific eye laterality (OD or OS) to overfit on")
    args = parser.parse_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    return cmd_args


def main():
    cmd_args = parse_cmd_args()
    args = initial_setup(cmd_args)
    print(args['inr_decoder'])
    atlas_builder = AtlasBuilder(args)
    
    # Close TensorBoard writer
    if 'tb_writer' in args and args['tb_writer'] is not None:
        args['tb_writer'].close()


if __name__ == "__main__":
    main()
