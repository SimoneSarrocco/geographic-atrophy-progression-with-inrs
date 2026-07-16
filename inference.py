import os
import yaml
import argparse
import torch
from build_model import ModelBuilder
from run import initial_setup, override_args

def parse_inference_args():
    parser = argparse.ArgumentParser(
        description="GAP-INR Standalone Inference",
        epilog="The model config must describe the same architecture the checkpoint was trained "
               "with, otherwise the weights will not load. Pass --config_model <file> when the "
               "checkpoint was not trained with configs/config_model.yaml; "
               "verify_run_config.py --checkpoint <ckpt> prints what a checkpoint used.")
    parser.add_argument("--config_data", type=str, default=None,
                        help="Override the dataset section in config_data.yaml. If omitted, the "
                             "config_data field inside the --config_model file is used (as in run.py). "
                             "A section whose enabled `conditions` differ from the checkpoint's "
                             "changes cond_dims, and the weights will not load.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--n_subjects", type=int, default=10, help="Number of subjects to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Split that --n_subjects caps. This does NOT choose what runs: the "
                             "validation routine always evaluates val (and train, per "
                             "validation.train_eval_every), and the test split is evaluated when "
                             "test.activate is true in the config. Use evaluate.py to target a split.")
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

    # These are inference-script options, not config keys: keep them out of the config override so
    # they don't overwrite same-named config entries (n_subjects is a per-split dict in the config).
    checkpoint = cmd_args.pop('checkpoint')
    n_subjects = cmd_args.pop('n_subjects')
    output_name = cmd_args.pop('output_name')
    split = cmd_args.get('split', 'val')

    args = initial_setup(cmd_args)

    # The checkpoint carries the epoch it was written at; load_checkpoint() addresses it as
    # <dir>/checkpoint_epoch_<epoch>.pth.
    chkp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    chkp_dir = os.path.dirname(os.path.abspath(checkpoint))
    args['load_model'] = {'path': chkp_dir, 'epoch': chkp['epoch']}
    args['epochs']['train'] = 0          # load and evaluate, never train
    args['validate_every'] = 1
    args['validation']['activate'] = True
    args['n_subjects'][split] = n_subjects
    args['output_dir'] = os.path.join(chkp_dir, output_name)
    os.makedirs(args['output_dir'], exist_ok=True)

    print(f"\n--- Starting GAP-INR Inference ---")
    print(f"Checkpoint: {checkpoint} (epoch {chkp['epoch']})")
    print(f"Split: {split}")
    print(f"Output: {args['output_dir']}")
    print(f"----------------------------------\n")

    # Initialize ModelBuilder (loads the weights via load_model)
    model_builder = ModelBuilder(args)

    # Run validation: latent optimization + metrics + future-visit prediction
    print("\nRunning evaluation pipeline...")
    model_builder.validate(epoch_train=0)

    # Generate conditioned renders if requested in config
    if args.get('generate_cond_renders', False):
        print("\nGenerating conditioned renders...")
        model_builder.generate_renders(epoch=0)

    print(f"\nInference completed. Results saved to: {args['output_dir']}")
    
    if 'tb_writer' in args and args['tb_writer'] is not None:
        args['tb_writer'].close()

if __name__ == "__main__":
    main()
