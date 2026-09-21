import os
import glob
import csv
import re
import argparse
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:  # SlicerSALT's headless Python has no Tk runtime.
    tk = None
    filedialog = None
import numpy as np
from feature_metadata import (load_labels, load_dataset_labels,
                              write_feature_manifest, validate_feature_contracts)


# =============================================================================
# Helper: Folder Picker
# =============================================================================
def prompt_folder(title):
    if tk is None or filedialog is None:
        raise RuntimeError("tkinter is unavailable; pass --spharm_dir explicitly")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title, initialdir=os.getcwd())
    root.destroy()
    return folder if folder else None


# =============================================================================
# Group Classification
# =============================================================================
def classify_subject(subject_name):
    is_left_side = subject_name.startswith("left_") or subject_name.startswith("lh_") or "_lh" in subject_name.lower()
    name_upper = subject_name.upper()
    
    # 1. Healthy Control / Normal (0)
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", 0
    
    # 2. Ipsilateral TLE (Diseased) (1)
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", 1
        
    # 3. Contralateral TLE (Healthy-side) (2)
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", 2
        
    # 4. General TLE (1)
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", 1
        
    return "Unknown", -1


# =============================================================================
# Parse SPHARM-PDM .coef format
# =============================================================================
def parse_coef(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Match all triplets {x, y, z}
    pattern = re.compile(r"\{([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+)\}")
    matches = pattern.findall(content)
    
    coeffs = []
    for m in matches:
        coeffs.append([float(x) for x in m])
    
    # Get num coeffs from first number e.g. { 169, ...
    num_match = re.search(r"\{\s*(\d+)", content)
    if num_match:
        num_coeffs = int(num_match.group(1))
        return coeffs[:num_coeffs]
    
    return coeffs


# =============================================================================
# Main Extraction
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spharm_dir", default=None,
                        help="Path to spharm_results folder. ถ้าไม่ระบุจะเด้ง dialog")
    parser.add_argument('--labels_csv', help='Explicit labels keyed by the output Subject name; columns Subject,BinaryClass[,Class,Group]')
    parser.add_argument('--allow-missing-provenance', action='store_true',
                        help='Explicit exploratory mode for historical/augmented coefficients without _processing.json; output is marked provenance_status=missing and must not be used as fixed-reference validation data')
    parser.add_argument('--coef_variant', choices=('ellalign', 'spharm', 'auto'), default='ellalign',
                        help='Coefficient variant to extract. Use the same variant for train/test; default is SPHARM_ellalign.')
    args = parser.parse_args()
    labels = load_labels(args.labels_csv)
    dataset_labels = load_dataset_labels(args.labels_csv)

    spharm_dir = args.spharm_dir
    if not spharm_dir:
        print("Opening folder picker...")
        spharm_dir = prompt_folder("Select 'spharm_results' folder (or output_xxx folder)")
    
    if not spharm_dir:
        print("Error: No folder selected. Exiting.")
        return

    # Handle parent directory selection
    if os.path.basename(spharm_dir.rstrip("\\/")).lower() != "spharm_results":
        candidate = os.path.join(spharm_dir, "spharm_results")
        if os.path.isdir(candidate):
            spharm_dir = candidate

    print(f"Target folder: {spharm_dir}")

    # 1. ค้นหาไฟล์ .coef.  The balanced set contains only ellalign
    # coefficients.  The old preference for *_SPHARM.coef when both files
    # existed made train and test use different coordinate frames.
    if args.coef_variant == 'ellalign':
        patterns = [("*_SPHARM_ellalign.coef", "ellalign")]
    elif args.coef_variant == 'spharm':
        patterns = [("*_SPHARM.coef", "spharm")]
    else:
        patterns = [("*_SPHARM_ellalign.coef", "ellalign"),
                    ("*_SPHARM.coef", "spharm"),
                    ("*.coef", "coef")]
    coef_files = []
    source = None
    for pattern, variant_name in patterns:
        coef_files = sorted(glob.glob(os.path.join(spharm_dir, pattern)))
        if coef_files:
            source = variant_name
            break

    if not coef_files:
        raise FileNotFoundError(f"No {args.coef_variant} SPHARM coefficient files found in {spharm_dir}")

    print(f"Found {len(coef_files)} coefficient files (source type: '{source}'). Processing...")

    # ตั้งค่าโฟลเดอร์สำหรับผลลัพธ์ ML
    ml_output_dir = os.path.join(os.path.dirname(spharm_dir), "ml_features")
    if not os.path.exists(ml_output_dir):
        os.makedirs(ml_output_dir)

    coef_csv_path = os.path.join(ml_output_dir, "spharm_coef_features.csv")

    # 2. อ่านไฟล์แรกเพื่อทราบจำนวนสัมประสิทธิ์
    first_coeffs = parse_coef(coef_files[0])
    num_coeffs = len(first_coeffs)
    # Validate the entire batch before opening (and potentially replacing) a CSV.
    # The normal path is strict. Historical and synthetic folders created by
    # augment_plsda_balanced.py may not carry sidecars, so allow an explicit,
    # clearly marked exploratory fallback for those inputs.
    try:
        feature_contract = validate_feature_contracts(coef_files)
    except ValueError as exc:
        if not args.allow_missing_provenance:
            raise
        feature_contract = {
            'version': 'spharm-feature-v1',
            'icp_mode': 'unknown',
            'icp_geometry_version': None,
            'icp_reference_sha256': None,
            'spharm_template': None,
            'spharm_template_sha256': None,
            'num_iterations': None,
            'subdiv_level': None,
            'spharm_degree': None,
            'grid_theta': None,
            'grid_phi': None,
            'provenance_status': 'missing',
            'provenance_warning': str(exc),
        }
        print(f"WARNING: {exc}")
        print("WARNING: Continuing in exploratory mode; this CSV is not fixed-reference validated data.")
    feature_contract = dict(feature_contract)
    feature_contract['coefficient_variant'] = source
    if not num_coeffs:
        raise ValueError('Empty coefficient file')
    for filepath in coef_files:
        values = np.asarray(parse_coef(filepath), dtype=float)
        if values.shape != (num_coeffs, 3) or not np.isfinite(values).all():
            raise ValueError(f'Invalid or inconsistent coefficients: {filepath}')
        subject = os.path.basename(filepath)
        for suffix in ('_SPHARM_ellalign.coef', '_SPHARM.coef'):
            subject = subject.replace(suffix, '')
        if labels is not None and subject not in labels:
            raise ValueError(f'Missing explicit label for {subject}')
    print(f"Number of SPHARM coefficients per subject: {num_coeffs} (total {num_coeffs * 3} values)")

    # 3. สร้างหัวตาราง (Header)
    header = ["Subject", "Group", "Class", "BinaryClass"]
    if dataset_labels is not None:
        header.append("Dataset")
    header += [f"Coef_{i+1}" for i in range(num_coeffs * 3)]

    # ตั้งชื่อไฟล์เอาท์พุตตามโฟลเดอร์ที่เลือก เพื่อไม่ให้เขียนทับกัน
    folder_basename = os.path.basename(spharm_dir.rstrip("\\/"))
    coef_csv_path = os.path.join(ml_output_dir, f"{folder_basename}_coef_features.csv")

    with open(coef_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filepath in coef_files:
            filename = os.path.basename(filepath)
            
            # ลบส่วนขยายออกเพื่อใช้เป็นชื่อ Subject
            subject_name = filename
            for suffix in ("_SPHARM_ellalign.coef", "_SPHARM.coef"):
                subject_name = subject_name.replace(suffix, "")

            # จัดกลุ่ม
            group_name, group_label = classify_subject(subject_name)
            # Preserve unknown labels. Training loaders must reject these rows.
            binary_class = -1 if group_label < 0 else int(group_label == 1)
            if labels is not None:
                if subject_name not in labels:
                    raise ValueError(f'Missing explicit label for {subject_name}')
                group_name, group_label, binary_class = labels[subject_name]
            if dataset_labels is not None and subject_name not in dataset_labels:
                raise ValueError(f'Missing explicit Dataset membership for {subject_name}')

            # โหลดสัมประสิทธิ์
            coeffs = parse_coef(filepath)
            if len(coeffs) != num_coeffs:
                raise ValueError(f'Coefficient file changed during extraction: {filename}')

            flat_coeffs = np.array(coeffs).ravel() # ขนาด (3 * num_coeffs,)

            # ประกอบข้อมูลแถว
            row = [subject_name, group_name, group_label, binary_class]
            if dataset_labels is not None:
                row.append(dataset_labels[subject_name])
            row.extend(["{:.8f}".format(val) for val in flat_coeffs])
            writer.writerow(row)

    write_feature_manifest(coef_csv_path, coef_files, args.labels_csv,
                           contract=feature_contract)

    print(f"Successfully saved SPHARM coefficient features for {len(coef_files)} subjects to: {coef_csv_path}")
    print("=" * 60)
    print("Done! You can use this CSV file for your ML models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
