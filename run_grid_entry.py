import os
import sys
import yaml
import argparse
from datetime import datetime
import wandb as wd
from torch.utils.tensorboard import SummaryWriter

# GAP-INR imports
from build_atlas import AtlasBuilder
from download_lakefs_data import download_dataset
from run import override_args

os.environ["WANDB__SERVICE_WAIT"] = "500"


def parse_args():
    parser = argparse.ArgumentParser(description="GAP-INR Grid Search Run Entry")
    parser.add_argument("--config-atlas", type=str, required=True, help="Path to config_atlas.yaml")
    parser.add_argument("--config-data", type=str, required=True, help="Path to config_data.yaml")
    return parser.parse_args()


def initial_setup(config_atlas_path, config_data_path):
    # Load specific configurations passed to the script
    with open(config_atlas_path, 'r') as stream:
        args_atlas = yaml.safe_load(stream)
        
    config_name = args_atlas.get('config_data', 'faf_ga')
    
    with open(config_data_path, 'r') as stream:
        args_data = {'dataset': yaml.safe_load(stream)[config_name]}
        
    args = {**args_data, **args_atlas}
    
    # Resolve subject ids path (relative vs absolute)
    subject_ids_path = args['dataset']['subject_ids']
    if not os.path.isabs(subject_ids_path):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        subject_ids_path = os.path.join(project_dir, subject_ids_path.lstrip('./'))
        
    with open(subject_ids_path, 'r') as stream:
        args['dataset']['subject_ids'] = yaml.safe_load(stream)[args['dataset']['dataset_name']]['subject_ids']

    # Pre-fetch from LakeFS
    print(f"\n--- Initiating Multi-threaded LakeFS Pre-fetch for {config_name} ---")
    try:
        download_dataset(config_name, config_path=config_data_path, num_workers=16)
    except Exception as e:
        print(f"Pre-fetch encountered an issue (will fallback to sequential): {e}")
    print("----------------------------------------------------------------\n")

    # The output directory has already been custom constructed in the configuration
    # by the grid search script (e.g. output_root/label)
    os.makedirs(args['output_dir'], exist_ok=True)
    print(f"Output directory: {args['output_dir']}")

    # Save copy of current configs to output directory
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
              
    if args['atlas_gen']['conditions'] is not None:
        for key in list(args['atlas_gen']['conditions'].keys()):
            if not args['dataset']['conditions'][key]:
                print(f"WARNING: The atlas condition {key} is not set True in the dataset config. "
                      f"Turning off the atlas generation for {key}.")
                args['atlas_gen']['conditions'].pop(key)

    # Initialize Weights & Biases if logging is enabled
    run_name = os.path.basename(args['output_dir'])
    if args.get('logging', False):
        wd.init(config=args, project=args.get('project_name', 'GAP-INR_omega_grid'), 
                entity=args.get('wandb_entity'), name=run_name)
    
    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(args['output_dir'], 'tb_logs')
    args['tb_writer'] = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log directory: {tb_log_dir}")
    
    return args


def main():
    cmd_args = parse_args()
    args = initial_setup(cmd_args.config_atlas, cmd_args.config_data)
    print("INR Decoder parameters:")
    print(args['inr_decoder'])
    
    # Run the AtlasBuilder
    atlas_builder = AtlasBuilder(args)
    
    # Close TensorBoard writer
    if 'tb_writer' in args and args['tb_writer'] is not None:
        args['tb_writer'].close()


if __name__ == "__main__":
    main()
