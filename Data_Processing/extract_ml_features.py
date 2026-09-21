import os
import glob
import csv
import argparse
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:  # SlicerSALT's headless Python has no Tk runtime.
    tk = None
    filedialog = None
import vtk
from vtk.util import numpy_support
import numpy as np
from feature_metadata import (load_labels, load_dataset_labels,
                              validate_mesh_processing_contracts,
                              write_geometry_manifest)


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
# Main Extraction
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spharm_dir", default=None,
                        help="Path to spharm_results folder. ถ้าไม่ระบุจะเด้ง dialog")
    parser.add_argument('--mesh_variant', choices=('ellalign', 'realigned', 'procalign', 'auto'), default='ellalign',
                        help='Mesh variant to extract. Use the same variant for train/test; default is SPHARM_ellalign.')
    parser.add_argument('--labels_csv', help='Explicit labels keyed by the output Subject name; columns Subject,BinaryClass[,Class,Group,Dataset]')
    parser.add_argument('--require-fixed-reference', action='store_true',
                        help='Require matching ICP/SPHARM processing sidecars and fixed_reference mode for every mesh')
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

    # 1. ค้นหาไฟล์ aligned vtk
    if args.mesh_variant == 'ellalign':
        patterns = [("*_SPHARM_ellalign.vtk", "ellalign")]
    elif args.mesh_variant == 'realigned':
        patterns = [("*_SPHARM_realigned.vtk", "realigned")]
    elif args.mesh_variant == 'procalign':
        patterns = [("*_SPHARM_procalign.vtk", "procalign")]
    else:
        patterns = [("*_SPHARM_ellalign.vtk", "ellalign"),
                    ("*_SPHARM_realigned.vtk", "realigned"),
                    ("*_SPHARM_procalign.vtk", "procalign"),
                    ("*.vtk", "vtk")]
    vtk_files = []
    source = None
    for pattern, variant_name in patterns:
        candidates = sorted(glob.glob(os.path.join(spharm_dir, pattern)))
        if pattern == '*.vtk':
            candidates = [f for f in candidates if not f.endswith('_grid.vtk')]
        if candidates:
            vtk_files = candidates
            source = variant_name
            break

    if not vtk_files:
        raise FileNotFoundError(f"No {args.mesh_variant} SPHARM mesh files found in {spharm_dir}")

    print(f"Found {len(vtk_files)} aligned meshes (source type: '{source}'). Processing...")
    processing_contract = None
    try:
        processing_contract = validate_mesh_processing_contracts(vtk_files)
    except ValueError as exc:
        if args.require_fixed_reference:
            raise
        print(f"WARNING: {exc}")
        print("WARNING: XYZ output will be marked without verified ICP/SPHARM provenance.")
    if args.require_fixed_reference:
        if source != 'ellalign':
            raise ValueError('--require-fixed-reference currently requires the ellalign mesh variant')
        if not processing_contract or processing_contract.get('icp_mode') != 'fixed_reference':
            raise ValueError('Fixed-reference extraction requires icp_mode=fixed_reference in every mesh sidecar')
        if not processing_contract.get('icp_reference_sha256') or not processing_contract.get('spharm_template_sha256'):
            raise ValueError('Fixed-reference extraction requires ICP and SPHARM template hashes')

    # ตั้งค่าโฟลเดอร์สำหรับผลลัพธ์ ML
    ml_output_dir = os.path.join(os.path.dirname(spharm_dir), "ml_features")
    if not os.path.exists(ml_output_dir):
        os.makedirs(ml_output_dir)

    coords_csv_path = os.path.join(ml_output_dir, "spharm_xyz_coords.csv")
    edges_csv_path = os.path.join(ml_output_dir, "mesh_edges.csv")

    # 2. อ่านไฟล์แรกเพื่อสร้างหัวคอลัมน์และดึง Graph Topology (Edges)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_files[0])
    reader.Update()
    poly = reader.GetOutput()

    num_points = poly.GetNumberOfPoints()
    print(f"Number of points per mesh: {num_points} vertices")

    # ดึงขอบเชื่อมต่อ (Edges) จากสามเหลี่ยมในโครงสร้าง Mesh
    print("Extracting mesh topology (undirected edges)...")
    cells = poly.GetPolys()
    id_list = vtk.vtkIdList()
    cells.InitTraversal()
    
    unique_edges = set()
    while cells.GetNextCell(id_list):
        n_pts = id_list.GetNumberOfIds()
        for i in range(n_pts):
            p1 = id_list.GetId(i)
            p2 = id_list.GetId((i + 1) % n_pts)
            # บันทึกเป็นคู่อันดับที่ไม่ซ้ำกัน
            edge = tuple(sorted((p1, p2)))
            unique_edges.add(edge)

    # เขียนขอบลงไฟล์ mesh_edges.csv
    with open(edges_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target"])
        for edge in sorted(unique_edges):
            writer.writerow(edge)
    print(f"Saved {len(unique_edges)} unique edges to: {edges_csv_path}")

    # 3. วนลูปอ่านพิกัดดิบของทุกไฟล์ และบันทึก
    print("Extracting coordinates and classifications...")
    
    # สร้างหัวตาราง (Header)
    header = ["Subject", "Group_Name", "Group_Label", "BinaryClass"]
    if dataset_labels is not None:
        header.append("Dataset")
    for idx in range(num_points):
        header.extend([f"x_{idx}", f"y_{idx}", f"z_{idx}"])

    rows_written = 0
    with open(coords_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filepath in vtk_files:
            filename = os.path.basename(filepath)
            # ลบส่วนขยายออกเพื่อใช้เป็นชื่อ Subject
            subject_name = filename
            for suffix in ("_SPHARM_realigned.vtk", "_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk"):
                subject_name = subject_name.replace(suffix, "")

            # จัดกลุ่ม
            group_name, group_label = classify_subject(subject_name)
            binary_class = -1 if group_label < 0 else int(group_label == 1)
            if labels is not None:
                if subject_name not in labels:
                    raise ValueError(f'Missing explicit label for {subject_name}')
                group_name, group_label, binary_class = labels[subject_name]
            if dataset_labels is not None and subject_name not in dataset_labels:
                raise ValueError(f'Missing explicit Dataset membership for {subject_name}')

            # โหลดพิกัด
            mesh_reader = vtk.vtkPolyDataReader()
            mesh_reader.SetFileName(filepath)
            mesh_reader.Update()
            mesh_poly = mesh_reader.GetOutput()
            
            if mesh_poly.GetNumberOfPoints() != num_points:
                raise ValueError(f"Mesh {filename} has {mesh_poly.GetNumberOfPoints()} points; expected {num_points}")

            pts = numpy_support.vtk_to_numpy(mesh_poly.GetPoints().GetData())
            flat_pts = pts.flatten() # ขนาด (3N,)

            # ประกอบข้อมูลแถว
            row = [subject_name, group_name, group_label, binary_class]
            if dataset_labels is not None:
                row.append(dataset_labels[subject_name])
            row.extend(["{:.8f}".format(val) for val in flat_pts])
            writer.writerow(row)
            rows_written += 1

    if rows_written != len(vtk_files):
        raise RuntimeError(f"XYZ row count mismatch: wrote {rows_written}, expected {len(vtk_files)}")
    write_geometry_manifest(coords_csv_path, vtk_files, source, num_points,
                            label_path=args.labels_csv,
                            processing_contract=processing_contract)
    print(f"Successfully saved features for {rows_written} subjects to: {coords_csv_path}")
    print("=" * 60)
    print("Done! You can use these two CSV files directly in your GNN model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
