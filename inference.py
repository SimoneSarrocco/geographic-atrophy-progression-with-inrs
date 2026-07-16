import os
import yaml
import argparse
import torch
from build_atlas import AtlasBuilder
from run import initial_setup, override_args

def parse_inference_args():
    parser = argparse.ArgumentParser(description="GAP-INR Standalone Inference")
    parser.add_argument("--config_data", type=str, default="faf_ga", help="Dataset configuration")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--n_subjects", type=int, default=10, help="Number of subjects to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output_name", type=str, default="inference_run", help="Name for the output directory")
    
    # Allow overriding any config parameter via --key__subkey value
    args, unknown = parser.parse_known_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    
    # Parse unknown args for deep overrides
    for i in range(0, len(unknown), 2):
        if unknown[i].startswith("--"):
            cmd_args[unknown[i][2:]] = unknown[i+1]
            
    return cmd_args

def main():
    cmd_args = parse_inference_args()
    
    # 1. Load base config
    with open('./configs/config_atlas.yaml', 'r') as stream:
        args_atlas = yaml.safe_load(stream)
    
    # 2. Setup args (reuse run.py logic)
    # We set epochs to 0 so AtlasBuilder doesn't start training
    args_atlas['epochs']['train'] = 0
    args_atlas['load_model']['path'] = cmd_args['checkpoint']
    args_atlas['n_subjects'][cmd_args['split']] = cmd_args['n_subjects']
    
    # Merge and override
    args = initial_setup(cmd_args)
    
    print(f"\n--- Starting GAP-INR Inference ---")
    print(f"Checkpoint: {cmd_args['checkpoint']}")
    print(f"Split: {cmd_args['split']}")
    print(f"Output: {args['output_dir']}")
    print(f"----------------------------------\n")

    # 3. Initialize AtlasBuilder (will load weights automatically)
    atlas_builder = AtlasBuilder(args)
    
    # 4. Run Validation (which performs latent optimization + metrics + prophecy)
    # validate() expects epoch_train, we pass 0
    print("\nRunning evaluation pipeline...")
    atlas_builder.validate(epoch_train=0)
    
    # 5. Generate Atlas if requested in config
    if args.get('atlas_gen', {}).get('activate', False):
        print("\nGenerating temporal atlas...")
        atlas_builder.generate_atlas(epoch=0)

    print(f"\nInference completed. Results saved to: {args['output_dir']}")
    
    if 'tb_writer' in args and args['tb_writer'] is not None:
        args['tb_writer'].close()

if __name__ == "__main__":
    main()
