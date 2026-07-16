import pandas as pd
import os
import glob
from tqdm import tqdm

def enrich_clinical_data(main_csv_path, slo_root_dir, output_path):
    print(f"Reading main CSV: {main_csv_path}")
    df = pd.read_csv(main_csv_path)
    
    enriched_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching Scans"):
        # Identify the scan directory ID from oct_path
        oct_filename = os.path.basename(row['oct_path'])
        scan_id = oct_filename.split('.')[0] # e.g. EYE_01_308
        
        scan_dir = os.path.join(slo_root_dir, scan_id)
        
        # Default values in case metadata is missing
        row['ScaleXSlo'] = 1.0
        row['ScaleYSlo'] = 1.0
        row['OCT_MinX_mm'] = 0.0
        row['OCT_MinY_mm'] = 0.0
        row['OCT_MaxX_mm'] = 0.0
        row['OCT_MaxY_mm'] = 0.0
        row['metadata_found'] = False
        
        if os.path.exists(scan_dir):
            # 1. Extract Scaling from scaninfo.csv
            info_path = os.path.join(scan_dir, f"{scan_id}_scaninfo.csv")
            if os.path.exists(info_path):
                info_df = pd.read_csv(info_path)
                if not info_df.empty:
                    row['ScaleXSlo'] = info_df['ScaleXSlo'].iloc[0]
                    row['ScaleYSlo'] = info_df['ScaleYSlo'].iloc[0]
                    row['metadata_found'] = True
            
            # 2. Extract Bounding Box from bscans.csv
            bscans_path = os.path.join(scan_dir, f"{scan_id}_bscans.csv")
            if os.path.exists(bscans_path):
                bs_df = pd.read_csv(bscans_path)
                if not bs_df.empty:
                    # Calculate physical extent of the scan in the SLO coordinate system
                    x_coords = pd.concat([bs_df['StartX'], bs_df['EndX']])
                    y_coords = pd.concat([bs_df['StartY'], bs_df['EndY']])
                    row['OCT_MinX_mm'] = x_coords.min()
                    row['OCT_MaxX_mm'] = x_coords.max()
                    row['OCT_MinY_mm'] = y_coords.min()
                    row['OCT_MaxY_mm'] = y_coords.max()
        
        enriched_rows.append(row)
        
    enriched_df = pd.DataFrame(enriched_rows)
    match_rate = enriched_df['metadata_found'].mean() * 100
    print(f"Enrichment Complete. Match Rate: {match_rate:.2f}%")
    
    enriched_df.to_csv(output_path, index=False)
    print(f"Enriched CSV saved to: {output_path}")

if __name__ == "__main__":
    MAIN_CSV = "./data/clinical_metadata_raw.csv"
    SLO_DIR = "./data/SLO"
    OUTPUT = "./data/clinical_metadata.csv"
    
    enrich_clinical_data(MAIN_CSV, SLO_DIR, OUTPUT)
