import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch

# Add current dir to path to import GAP-INR modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_loading.dataset import Data

def load_visit_data(dataset, row_dict):
    """
    Loads FAF image and all three grader masks for a specific visit row.
    """
    # 1. Load FAF
    faf_path = dataset.resolve_path(row_dict, 'faf_path', download=True)
    if not faf_path or not os.path.exists(faf_path):
        print(f"Warning: FAF path {faf_path} does not exist. Skipping visit.")
        return None, None, None
    
    img_faf = Image.open(faf_path).convert('L')
    img_faf_np = np.array(img_faf)
    target_size = img_faf.size # (width, height)
    
    # 2. Load Grader Masks (mask01, mask02, mask03)
    masks = []
    for m_suffix in ['mask01', 'mask02', 'mask03']:
        path = dataset._resolve_grader_mask_path(row_dict, m_suffix, download=True)
        if path and os.path.exists(path):
            m_img = Image.open(path).convert('L')
            if m_img.size != target_size:
                m_img = m_img.resize(target_size, Image.Resampling.NEAREST)
            masks.append(np.array(m_img) > 127)
        else:
            masks.append(None)
            
    # 3. Load or compute Majority Vote Mask
    valid_masks = [m for m in masks if m is not None]
    if len(valid_masks) > 0:
        masks_sum = np.sum(valid_masks, axis=0)
        threshold = (len(valid_masks) + 1) // 2
        m_maj = masks_sum >= threshold
    else:
        m_maj = np.zeros_like(img_faf_np, dtype=bool)
        
    return img_faf_np, masks, m_maj

def generate_progression_overlay(faf, mask_curr, mask_prev):
    """
    Creates an RGB overlay indicating lesion stable/growth/regression categories.
    Stable = Blue, Growth = Green, Regression = Red.
    """
    overlay = np.zeros((*faf.shape, 3), dtype=np.uint8)
    
    if mask_prev is None:
        # First visit: color the entire lesion mask in Blue (Baseline)
        overlay[mask_curr] = [51, 153, 255]
    else:
        stable = mask_curr & mask_prev
        growth = mask_curr & (~mask_prev)
        regression = mask_prev & (~mask_curr)
        
        overlay[stable] = [51, 153, 255]   # Blue
        overlay[growth] = [51, 204, 51]    # Green
        overlay[regression] = [255, 51, 51] # Red
        
    return overlay

def plot_patient_grader_variability(eye_id, visits_data, output_dir):
    """
    Generates a figure displaying FAF across visits (row 1) and the contour overlap
    of all three grader segmentations (row 2).
    Uses thin linewidths to show boundary differences clearly.
    """
    K = len(visits_data)
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8), squeeze=False)
    
    # Elegant overall title
    fig.suptitle(f"Patient-Eye: {eye_id} - Grader Segmentation Variability", fontsize=16, fontweight='bold')
    
    for i, v_data in enumerate(visits_data):
        visit_id = v_data['visit_id']
        weeks = v_data['weeks']
        faf = v_data['faf']
        masks = v_data['masks']
        
        # Format week header
        week_label = "Baseline" if weeks == 0 else f"Week {weeks:.1f}"
        
        # Row 1: FAF only
        ax1 = axes[0, i]
        ax1.imshow(faf, cmap='gray')
        ax1.set_title(f"Visit: {visit_id} ({week_label})", fontsize=12, fontweight='semibold')
        ax1.axis('off')
        
        # Row 2: FAF with overlaid contours of the three graders
        ax2 = axes[1, i]
        ax2.imshow(faf, cmap='gray')
        
        colors = ['#E53935', '#4CAF50', '#1E88E5'] # Red, Green, Blue
        labels = ['Grader 1', 'Grader 2', 'Grader 3']
        legend_handles = []
        
        for m_idx, mask in enumerate(masks):
            if mask is not None and mask.any():
                # Contour at level=0.5 defines boundary of binary mask.
                # Use thin linewidth=0.8 to clearly reveal subtle differences.
                ax2.contour(mask, levels=[0.5], colors=colors[m_idx], linewidths=0.8)
                line = plt.Line2D([0], [0], color=colors[m_idx], lw=1.5, label=labels[m_idx])
                legend_handles.append(line)
        
        ax2.axis('off')
        if i == 0 and len(legend_handles) > 0:
            ax2.legend(handles=legend_handles, loc='upper left', framealpha=0.6, fontsize=9)
            
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{eye_id}_grader_variability.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

def plot_patient_progression(eye_id, visits_data, mask_type, output_dir):
    """
    Generates a figure showing FAF across visits (row 1) and lesion growth/stable/shrinkage
    progression overlay on FAF (row 2).
    """
    # Check if there is at least one non-None mask for the specified grader across all visits
    g_idx = 0 if mask_type == 'grader1' else (1 if mask_type == 'grader2' else (2 if mask_type == 'grader3' else -1))
    if g_idx != -1:
        has_any_mask = any(v_data['masks'][g_idx] is not None for v_data in visits_data)
        if not has_any_mask:
            # Skip generating progression plot if this grader has no annotations for this patient-eye
            return
            
    K = len(visits_data)
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8), squeeze=False)
    
    if mask_type == 'grader1':
        title_suffix = "Grader 1"
    elif mask_type == 'grader2':
        title_suffix = "Grader 2"
    elif mask_type == 'grader3':
        title_suffix = "Grader 3"
    else:
        title_suffix = "Majority Vote"
        
    fig.suptitle(f"Patient-Eye: {eye_id} - Lesion Progression ({title_suffix})", fontsize=16, fontweight='bold')
    
    for i, v_data in enumerate(visits_data):
        visit_id = v_data['visit_id']
        weeks = v_data['weeks']
        faf = v_data['faf']
        
        # Get active mask for this strategy
        if mask_type == 'grader1':
            mask_curr = v_data['masks'][0] if v_data['masks'][0] is not None else np.zeros_like(faf, dtype=bool)
            mask_prev = (visits_data[i-1]['masks'][0] if visits_data[i-1]['masks'][0] is not None else np.zeros_like(faf, dtype=bool)) if i > 0 else None
        elif mask_type == 'grader2':
            mask_curr = v_data['masks'][1] if v_data['masks'][1] is not None else np.zeros_like(faf, dtype=bool)
            mask_prev = (visits_data[i-1]['masks'][1] if visits_data[i-1]['masks'][1] is not None else np.zeros_like(faf, dtype=bool)) if i > 0 else None
        elif mask_type == 'grader3':
            mask_curr = v_data['masks'][2] if v_data['masks'][2] is not None else np.zeros_like(faf, dtype=bool)
            mask_prev = (visits_data[i-1]['masks'][2] if visits_data[i-1]['masks'][2] is not None else np.zeros_like(faf, dtype=bool)) if i > 0 else None
        else: # majority
            mask_curr = v_data['majority']
            mask_prev = visits_data[i-1]['majority'] if i > 0 else None
        
        # Format week header
        week_label = "Baseline" if weeks == 0 else f"Week {weeks:.1f}"
        
        # Row 1: FAF only
        ax1 = axes[0, i]
        ax1.imshow(faf, cmap='gray')
        ax1.set_title(f"Visit: {visit_id} ({week_label})", fontsize=12, fontweight='semibold')
        ax1.axis('off')
        
        # Row 2: FAF with overlaid progression mask
        ax2 = axes[1, i]
        ax2.imshow(faf, cmap='gray')
        
        overlay = generate_progression_overlay(faf, mask_curr, mask_prev)
        alpha_mask = np.sum(overlay, axis=-1) > 0
        
        overlay_rgba = np.zeros((*faf.shape, 4), dtype=float)
        overlay_rgba[alpha_mask, :3] = overlay[alpha_mask] / 255.0
        overlay_rgba[alpha_mask, 3] = 0.55  # Alpha transparency level
        
        ax2.imshow(overlay_rgba)
        ax2.axis('off')
        
        # Legend showing category definitions
        if i == 0:
            legend_elements = [
                Patch(facecolor='#3399ff', edgecolor='none', alpha=0.6, label='Baseline Lesion')
            ]
            ax2.legend(handles=legend_elements, loc='upper left', framealpha=0.6, fontsize=9)
        elif i == 1:
            legend_elements = [
                Patch(facecolor='#3399ff', edgecolor='none', alpha=0.6, label='Stable'),
                Patch(facecolor='#33cc33', edgecolor='none', alpha=0.6, label='Growth'),
                Patch(facecolor='#ff3333', edgecolor='none', alpha=0.6, label='Regression')
            ]
            ax2.legend(handles=legend_elements, loc='upper left', framealpha=0.6, fontsize=9)
            
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{eye_id}_progression_{mask_type}.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication-ready grader variability and lesion progression figures.")
    parser.add_argument("--num_patients", type=int, default=10, help="Number of random patient-eyes to process. Use 0 or negative for all.")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Directory where figures will be saved.")
    args_cli = parser.parse_args()
    
    # Load configs
    with open('./configs/config_atlas.yaml', 'r') as f:
        args_atlas = yaml.safe_load(f)
    with open('./configs/config_data.yaml', 'r') as f:
        args_data = {'dataset': yaml.safe_load(f)['faf_ga']}
    args = {**args_data, **args_atlas}
    
    # Initialize full FAF-GA dataset using df_loaded to avoid splitting/filtering out visits
    print("Loading FAF-GA dataset metadata...")
    df_all = pd.read_csv(args['dataset']['tsv_file'])
    
    # Set subject ids to empty dict to prevent filtering inside dataset class
    args['dataset']['subject_ids'] = {'train': [], 'val': [], 'test': []}
    args['dataset']['world_bbox'] = [512, 512]
    args['dataset']['sampling_bbox'] = [512, 512]
    args['inr_decoder']['out_dim'] = [1, 1]
    
    # Initialize Data split
    dataset = Data(args, tsv_file=df_all, split='train', df_loaded=df_all)
    
    # Find all unique patient-eyes (Eye_ID column)
    id_col = args['dataset']['id_column']
    unique_eyes = dataset.df[id_col].unique()
    print(f"Found {len(unique_eyes)} unique patient-eyes in FAF-GA metadata.")
    
    if args_cli.num_patients > 0:
        # Sample a subset of eyes to save processing time
        selected_eyes = np.random.choice(unique_eyes, min(args_cli.num_patients, len(unique_eyes)), replace=False)
        print(f"Selected {len(selected_eyes)} patient-eyes for visualization.")
    else:
        selected_eyes = unique_eyes
        print(f"Processing ALL {len(selected_eyes)} patient-eyes.")
        
    # Create target figure directories
    var_dir = os.path.join(args_cli.output_dir, "grader_variability")
    majority_prog_dir = os.path.join(args_cli.output_dir, "majority_vote_progression")
    
    # Subdirectories for single graders
    grader1_dir = os.path.join(args_cli.output_dir, "single_grader_progression", "grader1")
    grader2_dir = os.path.join(args_cli.output_dir, "single_grader_progression", "grader2")
    grader3_dir = os.path.join(args_cli.output_dir, "single_grader_progression", "grader3")
    
    # Process each selected patient-eye
    for eye_idx, eye_id in enumerate(selected_eyes):
        eye_df = dataset.df[dataset.df[id_col] == eye_id]
        
        # Sort visits chronologically by weeks_from_baseline
        eye_df_sorted = eye_df.sort_values(by='weeks_from_baseline')
        
        visits_data = []
        for _, row in eye_df_sorted.iterrows():
            row_dict = row.to_dict()
            visit_id = row_dict.get('Visit_ID', 'N/A')
            weeks = row_dict.get('weeks_from_baseline', 0.0)
            
            # Load images & masks
            faf, masks, m_maj = load_visit_data(dataset, row_dict)
            
            if faf is not None:
                visits_data.append({
                    'visit_id': visit_id,
                    'weeks': weeks,
                    'faf': faf,
                    'masks': masks, # [mask01, mask02, mask03]
                    'majority': m_maj
                })
                
        if len(visits_data) == 0:
            print(f"[{eye_idx + 1}/{len(selected_eyes)}] Skipping {eye_id} - no valid visit data loaded.")
            continue
            
        print(f"[{eye_idx + 1}/{len(selected_eyes)}] Generating figures for {eye_id} with {len(visits_data)} visits...")
        
        # 1. Grader variability overlay contours (thin contours)
        plot_patient_grader_variability(eye_id, visits_data, var_dir)
        
        # 2. Individual grader progression overlays saved to subfolders
        plot_patient_progression(eye_id, visits_data, 'grader1', grader1_dir)
        plot_patient_progression(eye_id, visits_data, 'grader2', grader2_dir)
        plot_patient_progression(eye_id, visits_data, 'grader3', grader3_dir)
        
        # 3. Majority vote progression overlay
        plot_patient_progression(eye_id, visits_data, 'majority', majority_prog_dir)

    print(f"\nDone! Beautiful publication-ready figures saved to: {args_cli.output_dir}")

if __name__ == '__main__':
    main()
