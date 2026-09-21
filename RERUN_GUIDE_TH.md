# คู่มือรันใหม่: ICP reference เดิม, train/test และ downstream model

คู่มือนี้มี ICP สองแบบที่ห้ามใช้แทนกัน: reference จากรอบ groupwise เดิมสำหรับทำซ้ำ coordinate convention และ reference ที่ fit จาก train เท่านั้นสำหรับประเมินโมเดลอย่างอิสระ จากนั้นจึงอธิบายเส้นทาง train/test ต่อจนถึง Optuna/model evaluation

ผล ICP/SPHARM และ launchers รุ่นเก่าถูกล้างแล้วเพื่อไม่ให้สับสน รายการที่ลบ/เก็บไว้ดูได้ใน `CLEANUP_MANIFEST_20260921.json`; reference bundle และไฟล์ split ที่เส้นทาง train-only ต้องใช้ยังอยู่ครบ

## เลือก ICP workflow ให้ตรงงาน

| เป้าหมาย | Reference | ใช้ train/test เดิม 381 คนได้ไหม | ใช้รายงาน held-out test อิสระได้ไหม |
|---|---|---:|---:|
| ใช้ coordinate convention/reference เดิมทำซ้ำ หรือทดลองทีละไฟล์ | `ICP/references/legacy_groupwise_all_381_v1/{left,right}/mean_shape.ply` | ได้ | **ไม่ได้** เพราะ reference สร้างจากทั้ง 381 คน รวมผู้ที่อาจอยู่ใน test |
| ประเมินโมเดลใหม่ | fit reference ใหม่จาก train masks เท่านั้น ผ่าน `ICP/rerun_fixed_reference.py` | ได้ โดยแบ่ง train/test ก่อน ICP | ได้ในส่วน ICP reference; ยังต้องตรวจแหล่งที่มาของ SPHARM template และทำ ICP/SPHARM fold-local หากต้องการ CV ที่อิสระเต็มรูปแบบ |

ผลเทียบ IDs ใน `reference_manifest.json` ยืนยันว่า subject ปัจจุบันทุกคนอยู่ใน legacy groupwise reference แล้ว: ซ้าย 373/373 และขวา 377/377 ดังนั้นห้ามใช้ reference นี้สร้าง test features แล้วเรียกผลนั้นว่า independent held-out evaluation แม้ว่าการรันใหม่จะให้ผลซ้ำได้ก็ตาม

การ rerun แบบใหม่จะ freeze mean shape สุดท้ายจากรอบเดิม แล้วใช้ pairwise ICP กับ template นั้น ผลจึงไม่รับประกันว่าจะได้ transform/voxel output เหมือน groupwise run เดิมทุกไบต์ แม้ใช้ input เดิม; สิ่งที่ตรึงคือ template, hash, scale และ grid และการรันซ้ำด้วย code/parameters เดิมควรให้ผลสอดคล้องกัน

## A. ใช้ reference จาก ICP รอบเดิม: batch และ single

ตรวจสอบ `ICP/references/legacy_groupwise_all_381_v1/reference_manifest.json` ก่อนเริ่ม โดย bundle นี้คัดลอก mean shape จากผลเก่าและตรวจแล้วว่าแต่ละข้างมี 381 transforms/outputs, grid `128×128×128`, spacing `0.015625` และ common physical scale ที่กู้คืนจาก `T_matrices.npy`:

| ข้าง | Scale | SHA-256 ของ mean shape |
|---|---:|---|
| Left | `0.030540622985912882` | `e5db3de875fed2b774ef297526ba8f600742695e35415b1ebbca9d8f0063a94d` |
| Right | `0.03180055903090514` | `3f8c7896bca197a48955824eb37e90c91a3c48a6fc26b2ba893af7f216deb461` |

ตรวจ input ที่จะใช้ใน workspace ปัจจุบันแล้ว พบโฟลเดอร์ mask ที่เตรียมไว้ที่ `C:\Users\IHCK\Desktop\17-9-2569\merged_ds005602_ds004469` โดยมีไฟล์ `.nii.gz` ข้างละ 381 ไฟล์:

- ซ้าย: `C:\Users\IHCK\Desktop\17-9-2569\merged_ds005602_ds004469\left_hippocampus`
- ขวา: `C:\Users\IHCK\Desktop\17-9-2569\merged_ds005602_ds004469\right_hippocampus`

โฟลเดอร์เหล่านี้เป็น hippocampus masks ที่สกัดแล้ว ไม่ใช่ MRI T1 ต้นฉบับ คำสั่งด้านล่างใช้ legacy reference ที่สร้างจากชุดเดิม 381 ราย จึงเหมาะสำหรับ rerun เพื่อเทียบ coordinate convention/ตรวจขั้นตอนเท่านั้น ห้ามนำผลไปอ้างเป็น held-out test evaluation อิสระ

### คำสั่ง Git Bash: batch ซ้ายและขวา

เปิด Git Bash แล้วรันคำสั่งนี้จากโฟลเดอร์ใดก็ได้ ตัว launcher หา root โปรเจกต์และ input เอง ตรวจทั้งสองข้างก่อนเริ่ม แล้วรันซ้าย/ขวาตามลำดับ:

```bash
bash ~/Desktop/17-9-2569/Hippocampal-Shape-Analysis-for-Epilepsy-Detection/ICP/run_icp_legacy_reference_batch.sh
```

หากต้องการกำหนด run ID เอง ให้ใส่ต่อท้ายคำสั่ง เช่น `.../run_icp_legacy_reference_batch.sh 20260921_153000`.

launcher เรียก PythonSlicer โดยตรง ไม่เปลี่ยน PATH ในเครื่อง และใช้ run ID เดียวกันเก็บผลสองข้างไว้ด้วยกัน ค่า input เริ่มต้นที่ตรวจพบคือ:

การเรียก launcher ใหม่จะสร้าง run ID ใหม่อัตโนมัติ หากจะรันต่อใน run เดิม ให้ส่ง ID เดิมและตรวจว่าโฟลเดอร์ output ของข้างนั้นยังไม่มีอยู่ หากต้องเปลี่ยน input ให้เรียก `ICP/run_icp_with_reference.py` โดยตรงพร้อม `--input-dir` หรือ `--input-file`.

```text
C:\Users\IHCK\Desktop\17-9-2569\merged_ds005602_ds004469\left_hippocampus
C:\Users\IHCK\Desktop\17-9-2569\merged_ds005602_ds004469\right_hippocampus
```

ถ้าไม่ระบุ `--output-dir` สคริปต์สร้างโฟลเดอร์ใหม่ใต้ `reruns/icp_legacy_reference_<เวลา>\left` หรือ `...\right` ทุกครั้ง พร้อม `run_summary.json`; log ของ Slicer อยู่ใน `icp_debug_log.txt` ของ job นั้น

### คำสั่ง Git Bash: รัน mask เดียว

ตัวอย่างนี้ตรวจและรัน mask ซ้ายหนึ่งไฟล์ที่มีอยู่จริง:

```bash
cd ~/Desktop/17-9-2569/Hippocampal-Shape-Analysis-for-Epilepsy-Detection || exit 1
set -euo pipefail
python() { "/c/Program Files/SlicerSALT 6.0.0/bin/PythonSlicer.exe" "$@"; }
python ICP/run_icp_with_reference.py --side left \
  --input-file "../merged_ds005602_ds004469/left_hippocampus/left_Healthy_sub-00373_hippocampus_lh.nii.gz" \
  --check-only
python ICP/run_icp_with_reference.py --side left \
  --input-file "../merged_ds005602_ds004469/left_hippocampus/left_Healthy_sub-00373_hippocampus_lh.nii.gz"
```

สำหรับ mask เดี่ยวขวา ให้ใช้ `--side right` และไฟล์จาก `right_hippocampus` เท่านั้น

สำหรับ mask เดี่ยว ให้ใช้คำสั่ง Git Bash ด้านบนโดยเพิ่ม `--input-file "PATH_TO_MASK.nii.gz"` และระบุ `--side left` หรือ `--side right` ให้ตรงกับไฟล์ ส่วนการประเมิน train/test ให้ใช้เส้นทาง B ซึ่งเรียก Python runner จาก Git Bash.

ค่า fixed alignment ที่ runner บันทึก: grid/spacing จาก reference contract, pairwise ICP สูงสุด 100 รอบ, tolerance `0.0001`, 200 landmarks, interpolation `nn`, transform direction `auto`. `max_iterations=20` และ groupwise tolerance `0.00005` ถูกบันทึกเพื่อความสม่ำเสมอของ run manifest แต่ไม่ได้ใช้ fit mean ใน fixed-reference job. ค่าเก่าบางส่วนกู้คืนจาก log/status ไม่ได้ จึงอย่าอ้างว่าทุก hyperparameter ตรงกับการรันเดิม; mean shape, scale และ output grid มีที่มาและตรวจ hash แล้ว

## B. เส้นทางประเมินโมเดลด้วย reference จาก train เท่านั้น

ส่วนที่เหลือของคู่มือนี้เริ่มจาก mask ที่ FastSurfer สกัดแล้ว แบ่งผู้ป่วย train/test ก่อน ICP และทำ downstream จนถึง Optuna/model evaluation โดยรันซ้าย/ขวาแยกกัน แต่ใช้ protocol เดียวกัน ห้ามใช้ reference ในส่วน A กับเส้นทางนี้

## หลักการของรอบนี้

- ใช้ patient split ที่มีอยู่ใน `SPHARM/split_data/current_split_manifest.json` และป้ายกำกับจาก `ALL_Left_file_status.csv` กับ `ALL_Right_file_status.csv` ไม่สุ่มแบ่งชุดใหม่
- มี 381 ผู้เข้าร่วมใน split manifest; แถวที่มีป้ายกำกับใช้งานได้มีซ้าย 373 รายและขวา 377 ราย ตัวช่วยจะบันทึกและตัด mask ที่ไม่มีสถานะตรวจสอบแล้วออก ไม่เดาป้ายกำกับจากชื่อไฟล์
- สร้าง ICP reference ซ้ายจาก **train masks ซ้ายเท่านั้น** และ reference ขวาจาก **train masks ขวาเท่านั้น** แล้วจัดแนว train/test ของแต่ละข้างเข้าหา reference เดิมของข้างนั้น
- ซ้ายและขวาใช้ค่า ICP เหมือนกัน แต่มี reference คนละอัน เพราะเป็นรูปร่างคนละ hemisphere
- ใช้ SPHARM template คงที่ของ hemisphere ที่ตรงกัน (`Templates/SPHARM/template_spharm_left.vtk` หรือ `...right.vtk`) และสร้าง feature ด้วย `ellalign` เท่านั้น
- ผลทั้งหมดเขียนลง `$RUN_ROOT` ใหม่ สคริปต์จะหยุดถ้าโฟลเดอร์ผลลัพธ์มีไฟล์อยู่แล้ว จึงไม่ปนกับผลเก่าหรือเขียนทับข้อมูลเดิม

## 1. เตรียม Git Bash และตำแหน่งไฟล์

เปิด Git Bash แล้วรันชุดนี้ในหน้าต่างเดียวกับขั้นตอนถัดไป คำสั่งจะเข้า root โปรเจกต์เอง และตั้ง input เป็นโฟลเดอร์ mask ที่ตรวจพบในเครื่องปัจจุบัน:

```bash
cd ~/Desktop/17-9-2569/Hippocampal-Shape-Analysis-for-Epilepsy-Detection || exit 1
set -euo pipefail
python() { "/c/Program Files/SlicerSALT 6.0.0/bin/PythonSlicer.exe" "$@"; }

PROJECT_ROOT="$PWD"
PYTHON_SLICER="/c/Program Files/SlicerSALT 6.0.0/bin/PythonSlicer.exe"
SLICER_EXE="/c/Program Files/SlicerSALT 6.0.0/SlicerSALT.exe"
PREPROCESS_ROOT="/c/Users/IHCK/Desktop/17-9-2569/merged_ds005602_ds004469"
LEFT_MASK_DIR="$PREPROCESS_ROOT/left_hippocampus"
RIGHT_MASK_DIR="$PREPROCESS_ROOT/right_hippocampus"
RUN_ROOT="$PROJECT_ROOT/ICP/fixed_reference_rerun_$(date +%Y%m%d_%H%M%S)"

test -f "$PYTHON_SLICER"
test -f "$SLICER_EXE"
test -d "$LEFT_MASK_DIR"
test -d "$RIGHT_MASK_DIR"
printf 'Input masks: %s and %s\nOutput: %s\n' "$LEFT_MASK_DIR" "$RIGHT_MASK_DIR" "$RUN_ROOT"
```

ถ้าใช้ masks จากโฟลเดอร์อื่น ให้แก้ `PREPROCESS_ROOT` ก่อนรัน ส่วนการสกัด MRI/FastSurfer ต้องใช้ Python environment ของ FastSurfer แยกต่างหาก; คำสั่ง `python()` ในที่นี้ชี้ไปยัง PythonSlicer สำหรับ ICP/SPHARM/features เท่านั้น

## 2. แยก mask ตาม split แล้วรัน ICP

รัน preflight ก่อน คำสั่งจะตรวจ input, split manifest, status tables และ executable โดยยังไม่คัดลอก mask หรือเริ่ม ICP จากนั้นจึงรันจริง:

```bash
python ICP/rerun_fixed_reference.py \
  --left-mask-dir "$LEFT_MASK_DIR" \
  --right-mask-dir "$RIGHT_MASK_DIR" \
  --output-root "$RUN_ROOT" \
  --slicer-exe "$SLICER_EXE" \
  --check-only

python ICP/rerun_fixed_reference.py \
  --left-mask-dir "$LEFT_MASK_DIR" \
  --right-mask-dir "$RIGHT_MASK_DIR" \
  --output-root "$RUN_ROOT" \
  --slicer-exe "$SLICER_EXE"
```

สคริปต์จะตรวจว่าทุก ID จับคู่ได้กับ manifest/status, ไม่มี ID ซ้ำหรือไฟล์ขาด แล้วคัดลอก mask เป็น `masks\{left,right}\{train,test}` พร้อม CSV ป้ายกำกับ จากนั้นทำงาน ICP 6 งาน: fit reference 2 งาน (หนึ่งงานต่อข้าง) และ align แบบ fixed-reference 4 งาน (train/test ต่อข้าง)

ค่าที่กำหนดเหมือนกันทั้งสองข้าง: grid 128 voxels, spacing ที่ได้จาก reference contract, groupwise fit สูงสุด 20 รอบ, tolerance `0.00005`, pairwise ICP 100 รอบ, tolerance `0.0001`, 200 landmarks และ nearest-neighbor interpolation สำหรับ label mask

หลังจบ ให้ตรวจไฟล์ `fixed_reference_icp_rerun_manifest.json` และ `icp_status.json` ทั้งสี่ชุด `train_fixed`/`test_fixed` ว่ามี `success: true`, `mode: fixed_reference` และ `reference_sha256` ตรงกันภายในข้างเดียวกัน:

```text
$RUN_ROOT/icp/left/reference_fit/mean_shape.ply[.json]
$RUN_ROOT/icp/left/train_fixed/aligned_nifti/
$RUN_ROOT/icp/left/test_fixed/aligned_nifti/
$RUN_ROOT/icp/right/reference_fit/mean_shape.ply[.json]
$RUN_ROOT/icp/right/train_fixed/aligned_nifti/
$RUN_ROOT/icp/right/test_fixed/aligned_nifti/
```

ไฟล์ `labels/{side}/{split}_labels.csv` ถูกสร้างจาก status table และมี key `Subject` ที่ตรงกับชื่อผล SPHARM หลัง ICP เติม `_aligned` แล้ว จึงไม่ต้องเขียน label CSV เอง

## 3. รัน SPHARM ทั้งสี่ชุด

ใช้ Git Bash loop นี้จากหน้าต่างเดิม คำสั่งตั้ง `num_workers=5` และใช้ production template/ค่า production เดิมของ SPHARM:

```bash
for SIDE in left right; do
  for SPLIT in train test; do
    JOB="$RUN_ROOT/icp/$SIDE/${SPLIT}_fixed"
    INPUT_DIR="$JOB/aligned_nifti"
    TEMPLATE="$PROJECT_ROOT/Templates/SPHARM/template_spharm_${SIDE}.vtk"

    "$PYTHON_SLICER" "$PROJECT_ROOT/SPHARM/run_spharm_parallel.py" \
      --slicer_exe "$SLICER_EXE" \
      --num_workers 5 \
      --input_dir "$INPUT_DIR" \
      --output_dir "$JOB" \
      --reference_template "$TEMPLATE"

    "$PYTHON_SLICER" "$PROJECT_ROOT/SPHARM/verify_spharm_outputs.py" \
      --input_dir "$INPUT_DIR" \
      --output_dir "$JOB" \
      --report "$JOB/verify_spharm_outputs.json" \
      --deep
  done
done
```

ผล mesh/coefficient อยู่ในแต่ละ `spharm_results`; log และ log รวมของ worker อยู่ใน `logs` ภายในโฟลเดอร์งานนั้น เช่น `icp\left\train_fixed\logs\spharm_debug_log_merged.txt` สคริปต์จะแจ้งหยุดทันทีถ้า worker ล้มเหลวหรือ verifier พบ artifact ขาด/mesh ใช้ไม่ได้

## 4. สกัด coefficient และ XYZ features พร้อมป้ายกำกับ

```bash
COEF_EXTRACTOR="$PROJECT_ROOT/Data_Processing/extract_ml_features_coef.py"
XYZ_EXTRACTOR="$PROJECT_ROOT/Data_Processing/extract_ml_features.py"

for SIDE in left right; do
  for SPLIT in train test; do
    JOB="$RUN_ROOT/icp/$SIDE/${SPLIT}_fixed"
    SPHARM_DIR="$JOB/spharm_results"
    LABELS_CSV="$RUN_ROOT/labels/${SIDE}/${SPLIT}_labels.csv"

    "$PYTHON_SLICER" "$COEF_EXTRACTOR" \
      --spharm_dir "$SPHARM_DIR" \
      --labels_csv "$LABELS_CSV" \
      --coef_variant ellalign

    "$PYTHON_SLICER" "$XYZ_EXTRACTOR" \
      --spharm_dir "$SPHARM_DIR" \
      --mesh_variant ellalign \
      --labels_csv "$LABELS_CSV" \
      --require-fixed-reference
  done
done
```

ไฟล์ที่ต้องได้ในแต่ละ `ml_features` คือ `spharm_results_coef_features.csv` พร้อม `.json` และ `spharm_xyz_coords.csv` พร้อม `.json` ตัวสกัดจะหยุดหากชื่อ subject ไม่ตรง label, coefficient/mesh ไม่มี sidecar, มี geometry คนละแบบ หรือไม่พบ fixed ICP reference

## 5. สร้าง model input ชุดใหม่และตรวจ contract

```bash
"$PYTHON_SLICER" "$PROJECT_ROOT/Model/build_fixed_reference_inputs.py" \
  --run-root "$RUN_ROOT" \
  --output-root "$RUN_ROOT/model_inputs"
```

ตัวตรวจนี้จะสร้าง `All_coef`, `All_coef_Ds004469`, `All_coef_Ds005602` และ XYZ สำหรับ PointNet โดยไม่มี PLS-DA หรือ synthetic rows ในไฟล์ input จากนั้นตรวจ schema 507 coefficient/3,006 XYZ columns, label/cohort ตรงกัน, train/test ไม่มี patient ซ้ำ และ train/test ใช้ ICP/SPHARM reference hash เดียวกัน

จำนวนรายการที่ควรได้จาก manifest/status ปัจจุบัน (Healthy/TLE):

| ชุดข้อมูล | Left train | Left test | Right train | Right test |
|---|---:|---:|---:|---:|
| All | 191 / 107 | 48 / 27 | 220 / 82 | 55 / 20 |
| Ds004469 | 37 / 17 | 10 / 2 | 42 / 11 | 11 / 1 |
| Ds005602 | 154 / 90 | 38 / 25 | 178 / 71 | 44 / 19 |

ใน `Ds004469` test ด้านขวามี TLE เพียง 1 ราย และซ้ายมี 2 ราย จึงทำให้ sensitivity/accuracy ต่อชุดย่อยนี้ไม่เสถียรและไม่ควรตีความเป็นหลักฐานทางคลินิก

## 6. รัน Optuna, grouped 10-fold และทดสอบ held-out test

```bash
export PYTHONPATH="$(cygpath -m "$PROJECT_ROOT/Model/_training_cuda_site");$(cygpath -m "$PROJECT_ROOT/Model/_training_site")"
export MPLCONFIGDIR="$(cygpath -w "$PROJECT_ROOT/Model/_mplconfig")"

"$PYTHON_SLICER" "$PROJECT_ROOT/Model/optuna_leakage_free_all.py" \
  --data_root "$RUN_ROOT/model_inputs" \
  --protocol all \
  --cohort all \
  --side all \
  --model all \
  --folds 10 \
  --n_trials 10 \
  --tune_epochs 20 \
  --tune_patience 5 \
  --final_epochs 40 \
  --seed 42 \
  --eval_seeds 42,123,2026 \
  --device auto \
  --children_per_pair 8 \
  --noise_scale 0.02 \
  --pls_components 8 \
  --output_root "$RUN_ROOT/model_results"
```

ผลหลักอยู่ที่ `$RUN_ROOT/model_results`: `OPTUNA_TUNING_SUMMARY.csv`, `OPTUNA_FINAL_SUMMARY.csv`, `OPTUNA_WINNERS_BY_COHORT_SIDE.csv`, `RUN_MANIFEST.json`, Optuna trials และโฟลเดอร์ final แยก protocol/cohort/side/model/seed การเลือกผู้ชนะทำจาก OOF balanced accuracy; held-out test ไม่ถูกใช้ปรับค่า Optuna และประเมินแยกสำหรับ seed `42,123,2026`

## ขอบเขตการตีความผล

การแยกก่อน ICP ทำให้ held-out test ไม่ได้ร่วมสร้าง ICP reference และแต่ละ test mask ถูกจัดแนวเข้าหา reference ที่ fit จาก train ของข้างเดียวกันเท่านั้น อย่างไรก็ดี reference ปัจจุบัน fit จาก outer-train ทั้งชุดก่อนทำ grouped 10-fold ดังนั้น validation fold ใน OOF เคยมีรูปร่างร่วมกำหนด reference นี้อยู่ การประเมิน OOF จึงยังไม่ใช่ fold-local ครบตั้งแต่ ICP/SPHARM; หากต้องการอ้างว่า CV ปลอดการรั่วตั้งแต่ geometry ต้องสร้าง ICP reference และ SPHARM/features แยกใหม่ภายในแต่ละ fold ด้วย

SPHARM template ที่เก็บใน `Templates/SPHARM` ถูกใช้แบบคงที่และมีการบันทึก hash แต่ repository ยังไม่มีบันทึกแหล่งที่มาหรือรายชื่อผู้เข้าร่วมที่ใช้สร้าง template จึงยังยืนยันไม่ได้ว่า template นี้เป็นอิสระจากผู้เข้าร่วมใน held-out test หรือไม่ ก่อนรายงานว่า pipeline ทั้งชุดเป็นการทดสอบอิสระ ต้องตรวจสอบที่มาของ template ด้วย

ผลนี้เป็นการทดลองวิจัย ไม่ใช่เครื่องมือวินิจฉัยโรคที่ผ่านการยืนยันทางคลินิก การเปลี่ยน ICP reference/geometry ทำให้ต้องสร้าง features และฝึก weights ใหม่ ห้ามนำ weights เก่ามาปะกับ feature ชุดนี้
