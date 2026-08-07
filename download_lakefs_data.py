import os
import yaml
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data_loading.lakefs_config import make_loader, check_connection, LakeFSConfigError

# We need the path resolution logic
def extract_faf_ga_path(row_dict, mod_key):
    path = str(row_dict.get(mod_key, ""))
    if pd.isna(row_dict.get(mod_key)) or path.strip() == "" or path.strip().lower() == "nan":
        return None
        
    path = path.strip()
    
    # Check if we need to reconstruct FAF-GA paths
    try:
        pat = str(row_dict['Patient_ID']).strip()
        eye = str(row_dict['Eye']).strip()
        vis = str(row_dict['Visit_ID']).strip()
        if vis.isdigit():
            vis = f"V{int(vis):02d}"
        elif vis.startswith('V') and len(vis) == 2 and vis[1].isdigit():
            vis = f"V0{vis[1]}"
        filename = os.path.basename(path)
        
        if 'FAF' in filename or 'faf_path' in mod_key:
            mod_folder = 'Spectralis_faf'
        elif 'SLO' in filename or 'slo_path' in mod_key:
            mod_folder = 'Spectralis_slo'
        elif 'mask' in filename or 'ga_mask_path' in mod_key:
            mod_folder = 'Spectralis_faf'
        elif 'OCT' in filename or 'oct' in mod_key.lower():
            mod_folder = 'Spectralis_oct'
        else:
            mod_folder = 'Spectralis_slo'
            
        return f"data/{pat}/{eye}/{vis}/{mod_folder}/{filename}"
    except Exception:
        return path


def download_dataset(dataset_key, config_path='./configs/config_data.yaml',
                     lakefs_cfg_path=None, num_workers=8):
    """Pre-fetch every image the dataset section needs into the local cache.

    No-op with a printed note when lakeFS is not configured, since the images are then
    expected to be on local disk already (docs/DATA.md). Credentials, repository,
    branch and cache directory all come from data_loading/lakefs_config.py, so this
    and training always read the same place."""
    if lakefs_cfg_path:
        os.environ['GAPINR_LAKEFS_CFG'] = lakefs_cfg_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if dataset_key not in config:
        print(f"Dataset {dataset_key} not found in {config_path}")
        return

    ds_config = config[dataset_key]
    if 'lakefs' not in ds_config:
        print(f"Dataset {dataset_key} does not have a 'lakefs' configuration section.")
        return
        
    print(f"Preparing to download files for {dataset_key}...")

    try:
        lakefs_loader = make_loader(ds_config)
    except LakeFSConfigError as e:
        print(f"[lakeFS] {e}")
        return
    if lakefs_loader is None:
        return          # not configured: images are read from local disk

    tsv_file = ds_config.get('tsv_file')
    if not tsv_file or not os.path.exists(tsv_file):
        print(f"Metadata file not found: {tsv_file}")
        return
        
    print(f"Loading metadata from {tsv_file}...")
    if tsv_file.endswith('.tsv') or tsv_file.endswith('.csv'):
        sep = '\t' if tsv_file.endswith('.tsv') else ','
        df = pd.read_csv(tsv_file, sep=sep, low_memory=False)
    else:
        df = pd.read_excel(tsv_file)
        
    modalities = ds_config.get('modalities', [])
    print(f"Final setup: {len(df)} rows. Target modalities: {modalities}")
    
    # Collect all needed objects to download
    objects_to_download = set()
    
    is_faf_ga = (ds_config.get('dataset_name') == 'faf_ga')

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing paths"):
        row_dict = row.to_dict()
        for mod in modalities:
            if mod not in row_dict:
                continue
            path = str(row_dict[mod])
            if pd.isna(row_dict[mod]) or path.strip() == "" or path.strip().lower() == "nan":
                continue
                
            if is_faf_ga:
                if mod == 'ga_mask_path':
                    for m in ['mask01.png', 'mask02.png', 'mask03.png']:
                        pat = str(row_dict['Patient_ID']).strip()
                        eye = str(row_dict['Eye']).strip()
                        vis = str(row_dict['Visit_ID']).strip()
                        if vis.isdigit():
                            vis = f"V{int(vis):02d}"
                        elif vis.startswith('V') and len(vis) == 2 and vis[1].isdigit():
                            vis = f"V0{vis[1]}"
                        m_path = f"data/{pat}/{eye}/{vis}/Spectralis_faf/{pat}_{eye}_{vis}_{m}"
                        objects_to_download.add(m_path)
                else:
                    path = extract_faf_ga_path(row_dict, mod)
                    if path:
                        objects_to_download.add(path)
            else:
                if path:
                    objects_to_download.add(path)
                
    objects_list = list(objects_to_download)
    print(f"Found {len(objects_list)} unique object paths to verify/download.")
    
    # Check cache and prepare download list
    to_download = []
    
    for path in objects_list:
        local_name, obj_name = lakefs_loader.get_local_and_obj_names(path)
        if not os.path.exists(local_name):
            to_download.append((obj_name, local_name))
            
    if not to_download:
        print("All files are already downloaded and cached!")
        return
        
    print(f"Need to download {len(to_download)} files. Starting with {num_workers} parallel workers...")
    
    endpoint = cfg.get('lakefs_endpoint', cfg.get('s3_endpoint'))
    access_key = cfg.get('lakefs_access_key', cfg.get('access_key'))
    secret_key = cfg.get('lakefs_secret_key', cfg.get('secret_key'))
    ca_path = cfg.get('ca_path', cfg.get('ca_bundle'))
    
    def _download_task(args):
        import boto3
        obj_key, local_path = args
        try:
            # Boto3 clients are not thread-safe, so we create one per thread
            if endpoint and access_key and secret_key:
                verify_ssl = ca_path if ca_path else False
                s3_client = boto3.client(
                    's3',
                    endpoint_url=endpoint,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    verify=verify_ssl,
                )
            else:
                s3_client = boto3.client('s3')
                
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(
                Bucket=lakefs_config['repo'],
                Key=obj_key,
                Filename=local_path
            )
            return True, obj_key
        except Exception as e:
            return False, f"{obj_key}: {e}"

    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_download_task, item): item for item in to_download}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                if fail_count == 1:
                    print(f"\n[DEBUG] First failure reason: {msg}")
                
    print(f"Download complete: {success_count} succeeded, {fail_count} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-fetch a dataset's images from lakeFS into the local cache.",
        epilog="lakeFS is optional. With no credentials the training code reads images "
               "from local disk instead, and this script is not needed. See docs/DATA.md "
               "and configs/lakefs_cfg.example.yaml.")
    parser.add_argument('--dataset', type=str, default='faf_ga',
                        help="Dataset section in config_data.yaml (default: faf_ga)")
    parser.add_argument('--config', type=str, default='./configs/config_data.yaml',
                        help="Path to config_data.yaml")
    parser.add_argument('--lakefs-cfg', type=str, default=None,
                        help="Path to the credentials file. Default: configs/lakefs_cfg.yaml, "
                             "or $GAPINR_LAKEFS_CFG, or `config_path` in the dataset's "
                             "`lakefs:` block.")
    parser.add_argument('--workers', type=int, default=16, help="Parallel download workers")
    parser.add_argument('--check', action='store_true',
                        help="Print the resolved lakeFS settings and try one request, then "
                             "exit. Use this to verify credentials before a long run.")
    args = parser.parse_args()

    if args.check:
        if args.lakefs_cfg:
            os.environ['GAPINR_LAKEFS_CFG'] = args.lakefs_cfg
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        if args.dataset not in cfg:
            raise SystemExit(f"Dataset section '{args.dataset}' not found in {args.config}. "
                             f"Available: {', '.join(cfg)}")
        raise SystemExit(0 if check_connection(cfg[args.dataset]) else 1)

    download_dataset(args.dataset, args.config,
                     lakefs_cfg_path=args.lakefs_cfg, num_workers=args.workers)
