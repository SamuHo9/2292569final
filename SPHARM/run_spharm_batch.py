import os
import glob
import slicer
import vtk
import time
import sys
import argparse
import subprocess
import json
import hashlib
from datetime import datetime

# The SPHARM-PDM CLP binaries are native Windows processes.  A malformed mesh
# can trigger an access violation; suppress the OS crash dialog so a batch run
# receives the failure and can continue with the next subject.
if sys.platform == "win32":
    try:
        import ctypes
        _SEM_FAILCRITICALERRORS = 0x0001
        _SEM_NOGPFAULTERRORBOX = 0x0002
        _SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX | _SEM_NOOPENFILEERRORBOX
        )
    except Exception:
        pass

def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        for arg in sys.argv:
            if arg.endswith(".py") and os.path.isfile(arg):
                return os.path.dirname(os.path.abspath(arg))
        return os.getcwd()

SCRIPT_DIR = get_script_dir()

def sprint(msg, log_file):
    print(msg)
    sys.stdout.flush()
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def file_sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def cleanup_subject_files(output_base_dir, basename):
    patterns = [
        f"{basename}_pp.*",
        f"{basename}_para.vtk", f"{basename}_paraMix.txt",
        f"{basename}_paraPhi.txt", f"{basename}_paraPhiHalf.txt",
        f"{basename}_paraTheta.txt",
        f"{basename}_surf.vtk",
        f"{basename}_SPHARM.vtk", f"{basename}_SPHARM.coef",
        f"{basename}_SPHARM_ellalign.vtk", f"{basename}_SPHARM_ellalign.coef",
        f"{basename}_SPHARM_procalign.vtk", f"{basename}_SPHARM_procalign.coef",
        f"{basename}_SPHARM_grid.vtk",
        f"{basename}_processing.json",
        f"{basename}_SPHARMMedialAxis.vtk",
        f"{basename}_MedialAxisScalars.csv",
    ]
    removed = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(output_base_dir, pat)):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
    return removed


def write_processing_contract(final_coef, input_path, output_base_dir,
                              config, reference_template):
    """Write provenance consumed by feature extraction and Desktop inference."""
    status = {}
    status_path = os.path.join(os.path.dirname(output_base_dir), 'icp_status.json')
    if os.path.isfile(status_path):
        try:
            with open(status_path, 'r', encoding='utf-8') as stream:
                status = json.load(stream)
        except (OSError, ValueError):
            status = {}
    sidecar = final_coef.replace('_SPHARM.coef', '_processing.json')
    payload = {
        'feature_contract': {
            'version': 'spharm-feature-v1',
            'icp_mode': status.get('mode', 'unknown'),
            'icp_geometry_version': status.get('geometry_version'),
            'icp_reference_sha256': status.get('reference_sha256'),
            'spharm_template': reference_template,
            'spharm_template_sha256': file_sha256(reference_template),
            'num_iterations': int(config['num_iter']),
            'subdiv_level': int(config['subdiv']),
            'spharm_degree': int(config['degree']),
            'grid_theta': 4.5,
            'grid_phi': 4.5,
        },
        'input_name': os.path.basename(input_path),
        'input_sha256': file_sha256(input_path),
    }
    with open(sidecar, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)

def run_cli_checked(module, params, step_name, log_file, allow_completed_with_errors=False):
    cli_node = slicer.cli.run(module, None, params, wait_for_completion=True)
    status = cli_node.GetStatusString()
    err = (cli_node.GetErrorText() or "").strip()
    out = (cli_node.GetOutputText() or "").strip()
    if err:
        sprint(f"    [{step_name}] stderr: {err[:600]}", log_file)
    if status == "Completed with errors" and allow_completed_with_errors:
        sprint(f"    !!! CLI '{step_name}' reported warnings; validating output nodes", log_file)
        return True
    if status != "Completed":
        sprint(f"    !!! CLI '{step_name}' status = {status}", log_file)
        if out:
            sprint(f"    [{step_name}] stdout: {out[:600]}", log_file)
        return False
    return True

def node_has_points(node, min_points=10):
    if node is None:
        return False
    pd = node.GetPolyData()
    if pd is None:
        return False
    return pd.GetNumberOfPoints() >= min_points

def improve_label_quality(input_node, log_file):
    try:
        from scipy import ndimage
        import numpy as _np
    except ImportError:
        sprint("    (skipping hole-fill: scipy not available in this Python)", log_file)
        return

    arr = slicer.util.arrayFromVolume(input_node)
    if arr is None or arr.size == 0:
        return
    binary = (arr > 0)
    before = int(binary.sum())
    if before == 0:
        return

    filled = ndimage.binary_fill_holes(binary)

    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(filled, structure=struct, iterations=1)

    labeled, n_comp = ndimage.label(closed, structure=struct)
    if n_comp > 1:
        sizes = ndimage.sum(closed, labeled, range(1, n_comp + 1))
        largest = int(_np.argmax(sizes)) + 1
        cleaned = (labeled == largest)
        removed_comps = n_comp - 1
    else:
        cleaned = closed
        removed_comps = 0

    after = int(cleaned.sum())
    new_arr = cleaned.astype(_np.uint8)
    slicer.util.updateVolumeFromArray(input_node, new_arr)

    delta = after - before
    if delta != 0 or removed_comps > 0:
        sprint(f"    Quality fix: {before} -> {after} voxels (delta={delta:+d})  "
               f"components removed: {removed_comps}", log_file)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch SPHARM-PDM processing for SlicerSALT.")
    parser.add_argument("--input_dir", type=str, help="Directory containing .nii.gz labels")
    parser.add_argument("--output_dir", type=str, help="Directory to save all results")
    parser.add_argument("--fast", action="store_true", help="Fast/test mode: ~4-5x faster but lower mesh resolution")
    parser.add_argument("--regenerate_spharm_only", action="store_true", help="Skip SegPostProcess and GenParaMesh")
    parser.add_argument("--reference_template", type=str, default=None,
                        help="Path to reference _SPHARM.vtk used as regTemplate/flipTemplate.")
    parser.add_argument("--num_iterations", type=int, default=None, help="Iterations for GenParaMesh")
    parser.add_argument("--subdiv_level", type=int, default=None, help="Subdivision level for ParaToSPHARMMesh")
    parser.add_argument("--spharm_degree", type=int, default=None, help="SPHARM degree for ParaToSPHARMMesh")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of parallel shards")
    parser.add_argument("--shard_index", type=int, default=0, help="Shard index (0-based) for this worker")
    args, _ = parser.parse_known_args()
    return args

def init_logging(output_base_dir, input_dir, output_root, mode_tag, num_iter, subdiv, degree, shard_index=0, num_shards=1):
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    if num_shards > 1:
        log_filename = f"spharm_debug_log_shard{shard_index}of{num_shards}.txt"
    else:
        log_filename = "spharm_debug_log.txt"
    log_file = os.path.join(log_dir, log_filename)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"--- SPHARM Batch Mode Start: {datetime.now()} ---\n")
        if num_shards > 1:
            f.write(f"Shard: {shard_index + 1} of {num_shards}\n")
        f.write(f"Mode: {mode_tag} (iter={num_iter}, subdiv={subdiv}, degree={degree})\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Output: {output_root}\n")
        f.write(f"SCRIPT_DIR: {SCRIPT_DIR}\n\n")
    sprint(f"SCRIPT_DIR resolved to: {SCRIPT_DIR}", log_file)
    return log_file

def find_label_files(input_dir, log_file):
    extensions = ["*.nii.gz", "*.nii", "*.hdr"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    file_list = sorted(list(set(file_list)))
    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        sprint(f"Detected {len(label_files)} label-specific files (filtering out original images).", log_file)
        file_list = label_files

    sprint(f"Verified {len(file_list)} files for processing.", log_file)
    return file_list

def resolve_reference_template(args, file_list, output_base_dir, log_file):
    reference_template = args.reference_template

    if not reference_template and file_list:
        # Check if side can be deduced from input files or path
        side = None
        for f in file_list:
            fname = os.path.basename(f).lower()
            if fname.startswith("lh_") or "left" in fname:
                side = "left"
                break
            elif fname.startswith("rh_") or "right" in fname:
                side = "right"
                break
        if not side:
            base_lower = output_base_dir.lower()
            if "left" in base_lower or "lh" in base_lower:
                side = "left"
            elif "right" in base_lower or "rh" in base_lower:
                side = "right"

        if side:
            repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
            cand = os.path.join(repo_root, "Templates", "SPHARM", f"template_spharm_{side}.vtk").replace("\\", "/")
            cand_coef = cand.replace(".vtk", ".coef")
            if os.path.isfile(cand) and os.path.isfile(cand_coef):
                reference_template = cand
                sprint(f"\nAuto-detected official template for [{side.upper()}]: {os.path.basename(reference_template)}", log_file)

    if reference_template:
        reference_template = os.path.abspath(reference_template).replace("\\", "/")
        ref_coef = reference_template.replace(".vtk", ".coef")
        if not os.path.isfile(ref_coef):
            sprint(f"\n[ERROR] reference_template '{os.path.basename(reference_template)}' missing matching .coef file ('{os.path.basename(ref_coef)}').", log_file)
            reference_template = None
        else:
            sprint(f"Using reference template: {reference_template}", log_file)
            sprint(f"Using reference coef:     {ref_coef}", log_file)
    else:
        sprint(f"\n[INFO] No reference template specified or found. Proceeding without template registration.", log_file)

    return reference_template

def process_single_subject(file_path, index, total_files, output_base_dir, args, config, reference_template, log_file):
    basename = os.path.basename(file_path)
    for ext in [".nii.gz", ".nii", ".mgz", ".nrrd", ".hdr"]:
        if basename.lower().endswith(ext):
            basename = basename[:-len(ext)]
            break

    spharm_base = os.path.join(output_base_dir, basename).replace("\\", "/")
    final_vtk = f"{spharm_base}_SPHARM.vtk"
    final_coef = f"{spharm_base}_SPHARM.coef"
    grid_vtk = f"{spharm_base}_SPHARM_grid.vtk"

    # Check if subject is already completely processed
    if (os.path.isfile(grid_vtk) and os.path.getsize(grid_vtk) > 100
            and os.path.isfile(final_vtk) and os.path.getsize(final_vtk) > 100
            and os.path.isfile(final_coef) and os.path.getsize(final_coef) > 100):
        sprint(f"\n>>> [{index+1}/{total_files}] ALREADY COMPLETED (skipping): {basename}", log_file)
        if not os.path.isfile(final_coef.replace('_SPHARM.coef', '_processing.json')):
            sprint("    WARNING: preprocessing provenance sidecar is missing; "
                   "feature extraction will refuse this artifact", log_file)
        return True

    sprint(f"\n>>> [{index+1}/{total_files}] STARTING: {basename}", log_file)

    pp_mask_path = os.path.join(output_base_dir, f"{basename}_pp.nrrd").replace("\\", "/")
    para_mesh_path = os.path.join(output_base_dir, f"{basename}_para.vtk").replace("\\", "/")
    surf_mesh_path = os.path.join(output_base_dir, f"{basename}_surf.vtk").replace("\\", "/")

    regen_mode = (args.regenerate_spharm_only
                  and os.path.exists(para_mesh_path)
                  and os.path.exists(surf_mesh_path))

    # In regeneration mode keep existing artifacts until the replacement has
    # been verified.  Deleting them before ParaToSPHARMMesh runs can turn a
    # recoverable CLI failure into permanent data loss.
    if not regen_mode:
        n_removed = cleanup_subject_files(output_base_dir, basename)
        if n_removed > 0:
            sprint(f"  - Cleaned {n_removed} stale files from previous run", log_file)

    input_node = None
    pp_node = None
    para_node = None
    surf_node = None

    try:
        if regen_mode:
            sprint(f"  - REGEN MODE: reusing _para.vtk + _surf.vtk", log_file)
            para_node = slicer.util.loadModel(para_mesh_path)
            surf_node = slicer.util.loadModel(surf_mesh_path)
            if not node_has_points(para_node) or not node_has_points(surf_node):
                sprint(f"  - !!! Failed to load _para/_surf — skipping", log_file)
                return False
        else:
            sprint(f"  - Loading volume...", log_file)
            input_node = slicer.util.loadLabelVolume(file_path)
            if not input_node:
                sprint(f"  - FAILED to load: {file_path}", log_file)
                return False

            sprint(f"  - STEP 0: Improving label quality (fill holes + close + largest comp)...", log_file)
            improve_label_quality(input_node, log_file)

            sprint(f"  - STEP 1: SegPostProcess (Cleaning labels)...", log_file)
            pp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{basename}_pp")
            pp_params = {'fileName': input_node.GetID(), 'outfileName': pp_node.GetID(), 'label': 1}
            if not run_cli_checked(slicer.modules.segpostprocessclp, pp_params, "SegPostProcess", log_file):
                return False
            slicer.util.saveNode(pp_node, pp_mask_path)

            sprint(f"  - STEP 2: GenParaMesh (Creating spherical mesh)...", log_file)
            para_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{basename}_para")
            surf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{basename}_surf")
            para_params = {
                'infile': pp_node.GetID(),
                'outParaName': para_node.GetID(),
                'outSurfName': surf_node.GetID(),
                'numIterations': config['num_iter'],
                'label': 1
            }
            if not run_cli_checked(slicer.modules.genparameshclp, para_params, "GenParaMesh", log_file,
                                   allow_completed_with_errors=True):
                return False
            if not node_has_points(para_node) or not node_has_points(surf_node):
                sprint(f"  - !!! GenParaMesh returned EMPTY mesh — skipping", log_file)
                return False
            slicer.util.saveNode(para_node, para_mesh_path)
            slicer.util.saveNode(surf_node, surf_mesh_path)

        sprint(f"  - STEP 3: ParaToSPHARMMesh (Computing PDM)...", log_file)
        spharm_params = {
            'inParaFile': para_node.GetID(),
            'inSurfFile': surf_node.GetID(),
            'outbase': spharm_base,
            'subdivLevel': config['subdiv'],
            'spharmDegree': config['degree']
        }

        own_spharm_path = f"{spharm_base}_SPHARM.vtk"
        own_ellalign_path = f"{spharm_base}_SPHARM_ellalign.vtk"

        if reference_template:
            spharm_params['regTemplateFile'] = reference_template
            spharm_params['regTemplateFileOn'] = True
            spharm_params['regTemplate'] = reference_template
            ref_coef = reference_template.replace(".vtk", ".coef")
            if os.path.isfile(ref_coef):
                spharm_params['flipTemplateFile'] = ref_coef
                spharm_params['flipTemplateFileOn'] = True
                spharm_params['flipTemplate'] = ref_coef
                spharm_params['flipTemplateOn'] = True
                sprint(f"    Using official template alignment ({os.path.basename(reference_template)}) -> will produce _SPHARM_procalign.vtk", log_file)
            else:
                spharm_params['flipTemplateFileOn'] = False
                spharm_params['flipTemplateOn'] = False
                sprint(f"    Using template alignment (no flipTemplate) -> will produce _SPHARM_procalign.vtk", log_file)

        # ParaToSPHARMMesh in SlicerSALT frequently exits with the wrapper
        # status "Completed with errors" after it has already written a valid
        # VTK/coef pair.  Accept that status only here, then enforce strict
        # file/geometry validation below; all other CLI failures remain fatal.
        if not run_cli_checked(slicer.modules.paratospharmmeshclp, spharm_params,
                               "ParaToSPHARMMesh", log_file,
                               allow_completed_with_errors=True):
            return False

        if not (os.path.exists(own_spharm_path) and os.path.exists(own_spharm_path.replace(".vtk", ".coef"))):
            sprint(f"  - ERROR: SPHARM outputs not found for {basename}", log_file)
            if not regen_mode:
                cleanup_subject_files(output_base_dir, basename)
            return False

        try:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(own_spharm_path)
            reader.Update()
            poly = reader.GetOutput()
            if poly is None or poly.GetNumberOfPoints() < 10 or poly.GetNumberOfCells() == 0:
                sprint(f"  - ERROR: Invalid/Corrupted SPHARM output (0 cells / NaN) for {basename}", log_file)
                if not regen_mode:
                    cleanup_subject_files(output_base_dir, basename)
                return False
        except Exception as exc:
            sprint(f"  - ERROR: Could not read SPHARM VTK for {basename}: {exc}", log_file)
            if not regen_mode:
                cleanup_subject_files(output_base_dir, basename)
            return False

        final_vtk = f"{spharm_base}_SPHARM.vtk"
        final_coef = f"{spharm_base}_SPHARM.coef"
        grid_vtk = f"{spharm_base}_SPHARM_grid.vtk"

        if os.path.exists(final_vtk) and os.path.exists(final_coef):
            theta_step, phi_step = "4.5", "4.5"
            sprint(f"  - SUCCESS: {basename} completed. Resampling to {theta_step}x{phi_step} degree grid...", log_file)

            resample_script = os.path.join(SCRIPT_DIR, "resample_spharm_grid.py")
            if not os.path.isfile(resample_script):
                sprint(f"  - ERROR: resample_spharm_grid.py not found at {resample_script}", log_file)
                return False
            else:
                cmd = [sys.executable, resample_script, final_coef, grid_vtk, theta_step, phi_step]
                sprint(f"  - Running: {' '.join(cmd)}", log_file)
                resample_ok = False
                try:
                    kwargs = {}
                    if os.name == 'nt':
                        kwargs['creationflags'] = 0x08000000
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
                    resample_ok = True
                    if result.stdout:
                        sprint(f"  - Resample stdout: {result.stdout.strip()}", log_file)
                except subprocess.CalledProcessError as sub_err:
                    sprint(f"  - ERROR: resample_spharm_grid.py failed (exit {sub_err.returncode})", log_file)
                    sprint(f"  - stderr: {sub_err.stderr.strip()}", log_file)

                if resample_ok and os.path.isfile(grid_vtk) and os.path.getsize(grid_vtk) > 100:
                    sprint(f"  - SUCCESS: {basename} Grid VTK created.", log_file)
                    write_processing_contract(final_coef, file_path, output_base_dir,
                                              config, reference_template)
                else:
                    sprint(f"  - ERROR: Grid resampling failed for {basename}.", log_file)
                    return False
        else:
            sprint(f"  - ERROR: Result VTK not generated for {basename}.", log_file)
            return False

    except Exception as e:
        import traceback
        sprint(f"  - CRITICAL ERROR for {basename}: {str(e)}", log_file)
        sprint(traceback.format_exc(), log_file)
        return False

    finally:
        for node in [input_node, pp_node, para_node, surf_node]:
            if node:
                try: slicer.mrmlScene.RemoveNode(node)
                except Exception: pass
        sprint(f"  - Cleanup done.", log_file)

    return True

def run_batch_spharm():
    import json
    args = parse_args()

    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError('Invalid shard configuration')

    if args.num_iterations is not None or args.subdiv_level is not None or args.spharm_degree is not None:
        NUM_ITER = args.num_iterations if args.num_iterations is not None else (200 if args.fast else 1000)
        SUBDIV   = args.subdiv_level if args.subdiv_level is not None else (5 if args.fast else 10)
        DEGREE   = args.spharm_degree if args.spharm_degree is not None else (6 if args.fast else 12)
        MODE_TAG = "CUSTOM"
    elif args.fast:
        NUM_ITER, SUBDIV, DEGREE, MODE_TAG = 200, 5, 6, "FAST"
    else:
        NUM_ITER, SUBDIV, DEGREE, MODE_TAG = 1000, 10, 12, "PRODUCTION"

    output_root = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(SCRIPT_DIR, "output")
    input_dir = os.path.abspath(args.input_dir) if args.input_dir else os.path.join(output_root, "aligned_nifti")
    output_base_dir = os.path.join(output_root, "spharm_results")
    os.makedirs(output_base_dir, exist_ok=True)

    log_file = init_logging(output_base_dir, input_dir, output_root, MODE_TAG, NUM_ITER, SUBDIV, DEGREE,
                            shard_index=args.shard_index, num_shards=args.num_shards)
    all_files = find_label_files(input_dir, log_file)

    if not all_files:
        raise RuntimeError('No input labels found')

    if args.num_shards > 1:
        file_list = [f for idx, f in enumerate(all_files) if idx % args.num_shards == args.shard_index]
        sprint(f"Shard {args.shard_index + 1}/{args.num_shards}: assigned {len(file_list)} of {len(all_files)} files.", log_file)
    else:
        file_list = all_files

    reference_template = resolve_reference_template(args, file_list, output_base_dir, log_file)

    config = {'num_iter': NUM_ITER, 'subdiv': SUBDIV, 'degree': DEGREE}

    outcomes = []
    for i, file_path in enumerate(file_list):
        try:
            ok = process_single_subject(file_path, i, len(file_list), output_base_dir,
                                        args, config, reference_template, log_file)
        except Exception as exc:
            sprint(f'  - CRITICAL ERROR for {file_path}: {exc}', log_file)
            ok = False
        outcomes.append({'input': file_path, 'success': bool(ok)})

    failed = sum(not row['success'] for row in outcomes)
    reference_sha256 = None
    if reference_template and os.path.isfile(reference_template):
        try:
            import hashlib as _hashlib
            _digest = _hashlib.sha256()
            with open(reference_template, 'rb') as _ref_stream:
                for _chunk in iter(lambda: _ref_stream.read(1024 * 1024), b''):
                    _digest.update(_chunk)
            reference_sha256 = _digest.hexdigest()
        except OSError:
            reference_sha256 = None
    status_file = os.path.join(output_root, f'spharm_status_shard{args.shard_index}.json')
    with open(status_file, 'w', encoding='utf-8') as stream:
        json.dump({
            'total': len(outcomes),
            'failed': failed,
            'subjects': outcomes,
            'input_dir': os.path.abspath(input_dir),
            'output_dir': os.path.abspath(output_root),
            'mode': MODE_TAG,
            'config': config,
            'reference_template': reference_template,
            'reference_sha256': reference_sha256,
            'required_outputs': [
                '_SPHARM.coef', '_SPHARM.vtk',
                '_SPHARM_grid.vtk', '_SPHARM_ellalign.coef'
            ],
        }, stream, indent=2)
    if failed:
        raise RuntimeError(f'SPHARM incomplete: {failed}/{len(outcomes)} failed; see {status_file}')

    sprint("\n" + "="*60, log_file)
    sprint("!!! ALL BATCH PROCESSING COMPLETED !!!", log_file)
    sprint("="*60, log_file)

if __name__ == "__main__":
    exit_code = 0
    try:
        run_batch_spharm()
    except Exception as _e:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            slicer.util.exit(exit_code)
        except Exception:
            pass
        try:
            import time, threading
            def force_kill():
                time.sleep(1.5)
                os._exit(exit_code)
            t = threading.Thread(target=force_kill, daemon=True)
            t.start()
        except Exception:
            os._exit(exit_code)
