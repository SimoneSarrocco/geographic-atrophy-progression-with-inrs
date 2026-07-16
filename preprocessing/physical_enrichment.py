import csv
import os

def get_scan_metadata(scan_dir, scan_id):
    metadata = {
        'ScaleXSlo': '1.0',
        'ScaleYSlo': '1.0',
        'OCT_MinX': '0.0',
        'OCT_MaxX': '0.0',
        'OCT_MinY': '0.0',
        'OCT_MaxY': '0.0'
    }
    
    # 1. Read Scale from scaninfo.csv
    info_path = os.path.join(scan_dir, f"{scan_id}_scaninfo.csv")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata['ScaleXSlo'] = row.get('ScaleXSlo', '1.0')
                metadata['ScaleYSlo'] = row.get('ScaleYSlo', '1.0')
                break
                
    # 2. Read Bounding Box from bscans.csv
    bscans_path = os.path.join(scan_dir, f"{scan_id}_bscans.csv")
    if os.path.exists(bscans_path):
        with open(bscans_path, 'r') as f:
            reader = csv.DictReader(f)
            xs, ys = [], []
            for row in reader:
                try:
                    xs.extend([float(row['StartX']), float(row['EndX'])])
                    ys.extend([float(row['StartY']), float(row['EndY'])])
                except (KeyError, ValueError):
                    continue
            if xs and ys:
                metadata['OCT_MinX'] = str(min(xs))
                metadata['OCT_MaxX'] = str(max(xs))
                metadata['OCT_MinY'] = str(min(ys))
                metadata['OCT_MaxY'] = str(max(ys))
                
    return metadata

def main():
    main_csv = "./data/clinical_metadata_raw.csv"
    slo_root = "./data/SLO"
    output_csv = "./data/clinical_metadata.csv"
    
    if not os.path.exists(main_csv):
        print(f"Error: Could not find {main_csv}")
        return

    print(f"Processing {main_csv}...")
    
    with open(main_csv, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['ScaleXSlo', 'ScaleYSlo', 'OCT_MinX', 'OCT_MaxX', 'OCT_MinY', 'OCT_MaxY']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        count = 0
        matches = 0
        for row in reader:
            count += 1
            oct_path = row.get('oct_path', '')
            scan_id = os.path.basename(oct_path).split('.')[0]
            scan_dir = os.path.join(slo_root, scan_id)
            
            meta = get_scan_metadata(scan_dir, scan_id)
            row.update(meta)
            
            if float(meta['ScaleXSlo']) != 1.0:
                matches += 1
                
            writer.writerow(row)
            
    print(f"Finished. Processed {count} rows. Found metadata for {matches} scans.")
    print(f"Enriched CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()
