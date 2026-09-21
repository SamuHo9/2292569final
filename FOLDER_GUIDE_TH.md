# คู่มือโฟลเดอร์ของระบบ

เอกสารนี้อธิบายว่าแต่ละโฟลเดอร์ในโปรเจกต์ใช้ทำอะไร อยู่ในขั้นตอนไหน และควรใช้เป็นผลลัพธ์หลัก
หรือเป็นข้อมูลอ้างอิงเท่านั้น โดยยึด pipeline ปัจจุบันใน `RERUN_GUIDE_TH.md`

อัปเดต 21 กันยายน 2026: ผล ICP/SPHARM รุ่นเก่าที่สร้างซ้ำได้ถูกลบหลังตรวจและเก็บ ICP reference bundle แล้ว โฟลเดอร์ผลรันใหม่ให้กำหนดแยกตาม run ห้ามเขียนทับหรือเอามาปนกับผลเก่าใน `Model/` รายการ cleanup อยู่ใน `CLEANUP_MANIFEST_20260921.json`

## แผนผังการไหลของข้อมูล

```text
FastSurfer / MRI
    -> ICP
    -> ICP fixed-reference bundle + isolated run output
    -> SPHARM
    -> SPHARM/split_data
    -> Data_Processing
    -> Model/All_coef* และ Model/Output_Dataset
    -> Model/optuna_runs_10fold_all
    -> DesktopApp inference
```

## 1. โฟลเดอร์ระดับโปรเจกต์

| โฟลเดอร์ | หน้าที่ | สถานะ |
|---|---|---|
| `FastSurfer` | source และโมดูล segmentation สำหรับแปลง MRI เป็น brain/hippocampus masks | ใช้ใน preprocessing |
| `ICP` | สร้าง mesh, align แต่ละ subject กับ fixed reference และเก็บสถานะ ICP/SPHARM ที่เกี่ยวข้อง | ใช้ใน pipeline |
| `Templates` | SPHARM templates แยกซ้าย/ขวา | เก็บไว้และตรวจ source ก่อนอ้าง held-out test |
| `SPHARM` | รัน SPHARM-PDM, realignment, resampling, ตรวจ output และสร้าง split | ใช้ใน pipeline |
| `Data_Processing` | แปลง SPHARM เป็น coefficient/XYZ features, สร้าง manifest และ dataset | ใช้ใน pipeline |
| `Model` | data contract, model runners, training runtimes, datasets และผลประเมิน | ใช้ใน pipeline |
| `DesktopApp` | application สำหรับ import, แสดง mesh และ inference | ใช้สำหรับใช้งานปลายทาง |
| `Visualize` | ตัวดู mesh, ROC, bootstrap, PLS-DA และ plot สำหรับวิเคราะห์/นำเสนอ | เครื่องมือเสริม |
| `Model_Results_Excel` | workbook, CSV และ script สำหรับจัดรูปผลวิจัย | รายงาน/วิเคราะห์เสริม |
| `Model_Evaluation_and_Benchmarks` | ผล benchmark รุ่นก่อน เช่น bootstrap summary | ผลอ้างอิง ไม่ใช่ผล Optuna หลัก |
| `tests` | regression tests ของ contract, pipeline status, reference และ predictor schema | ใช้ตรวจโค้ด |
| `build` | ไฟล์ build/package ของ application หรือ dependency | ไม่ใช่ input ของ model |

## 2. `ICP`

### `ICP\references\legacy_groupwise_all_381_v1`

reference bundle ที่สร้างจาก mean shape ของ ICP groupwise เดิม ซ้าย/ขวาแยกกัน มี `mean_shape.ply`, sidecar `.json` และ `reference_manifest.json` ระบุ SHA-256, scale และ grid (128³, spacing 0.015625) ต้นทางคือรูปร่างทั้ง 381 คนต่อข้าง และตรวจยืนยันว่า ALL ปัจจุบันซ้าย 373/373 กับขวา 377/377 อยู่ใน reference นี้แล้ว จึงใช้ทำซ้ำ coordinate convention เดิม แต่ห้ามอ้าง independent held-out evaluation กับคนชุดนี้

### Runner และไฟล์โค้ด

- `ICP.py`: core; ต้องระบุโหมด `--reference_template`, `--fit_reference` หรือ `--exploratory_groupwise` ให้ชัด และรับ input เป็น directory หรือไฟล์เดียว
- `run_icp_with_reference.py`: Python launcher สำหรับรัน legacy fixed reference ทีละข้าง/ทีละไฟล์ ตรวจ input/reference ก่อนเปิด Slicer และตรวจ output หลัง Slicer จบ
- `run_icp_legacy_reference_batch.sh`: Git Bash runner สำหรับ preflight และรัน batch ซ้าย/ขวาใน run เดียวกัน
- `reference_contract.py`: ตรวจ hash/version/physical scale/grid ของ ICP reference
- `rerun_fixed_reference.py`, `prepare_mask_split.py`: ตัวช่วย Python สำหรับ Git Bash แบ่ง mask ตาม manifest แล้ว fit reference จาก train เท่านั้น
- `create_legacy_reference_from_previous_run.py`: สร้าง/ตรวจ bundle จาก legacy outputs; รอบปัจจุบันสร้าง bundle ไว้แล้ว
- `plot_icp_convergence.py`: plot groupwise history ถ้ามี

ผลรันใหม่ไม่เขียนลง `ICP\output_left_hippocampus` หรือ `ICP\output_right_hippocampus`; ให้ใช้ output directory ใหม่ที่กำหนดในคำสั่ง runner เสมอ

## 3. `Templates`

### `Templates\SPHARM`

- `template_spharm_left.vtk` และ `.coef`: SPHARM reference ของ hippocampus ซ้าย
- `template_spharm_right.vtk` และ `.coef`: SPHARM reference ของ hippocampus ขวา

ไฟล์เหล่านี้เป็นคนละชั้นกับ ICP reference ใน `ICP\references` ใช้ template ให้ตรง hemisphere และบันทึก hash ใน processing metadata ที่มา/กลุ่มคนที่ใช้สร้าง SPHARM templates ยังต้องตรวจสอบก่อนรายงานว่า held-out test เป็นอิสระครบทุกขั้น

## 4. `SPHARM`

### ไฟล์คำสั่งหลัก

- `run_spharm_parallel.py`: orchestrator แบ่ง SPHARM workers; เรียกผ่าน SlicerSALT Python runtime
- `run_spharm_batch.py`: worker ประมวลผลภายใน SlicerSALT
- `resample_spharm_grid.py`, `realign_spharm.py`: resample และปรับแนว mesh
- `check_spharm_environment.py`, `verify_spharm_outputs.py`, `verify_bilateral_outputs.py`: ตรวจ runtime/ความครบถ้วน/ความสอดคล้องของสองข้าง

launcher รุ่นเก่าที่ hard-code ไปยัง `ICP\output_*` ถูกลบแล้ว ให้ใช้คำสั่งใน `RERUN_GUIDE_TH.md` ที่ส่ง input/output ชัดเจน

### `SPHARM\split_data`

หลัง cleanup เก็บเฉพาะไฟล์ที่ขั้นตอนแบ่ง mask ต้องใช้อ้างอิง:

- `ALL_Left_file_status.csv`, `ALL_Right_file_status.csv`: label/cohort รายข้าง
- `current_split_manifest.json`: patient-level train/test assignment

โฟลเดอร์ผล SPHARM ที่ copy ไว้ใน `ALL_Left`, `ALL_Right`, `Ds004469_*`, `Ds005602_*` และรายงาน completeness รุ่นก่อนถูกลบเพื่อลดความสับสน เมื่อ rerun ให้สร้าง SPHARM outputs ภายใต้ `$RunRoot` ใหม่และ verify ก่อนสร้าง features

## 5. `Data_Processing`

- `extract_ml_features.py`: สร้าง feature dataset จาก mesh/XYZ สำหรับงาน ML
- `extract_ml_features_coef.py`: สร้าง SPHARM coefficient features
- `export_ml_dataset.py`: export ตาราง ML พร้อม metadata
- `build_all_coef_datasets.py`: สร้าง `All_coef`, `All_coef_Ds004469` และ `All_coef_Ds005602`
- `feature_metadata.py`: เขียน/อ่าน feature manifest และ label metadata
- `augment_plsda_balanced.py`: สร้าง balanced PLS-DA dataset รุ่น exploratory/legacy
- `augment_plsda_interpolation.py`: interpolation augmentation รุ่นเก่า/เสริม

สองไฟล์ augmentation ถูกเก็บไว้เพื่ออ้างอิงและสร้าง dataset exploratory แต่ไม่ควรรันก่อนแบ่ง fold
สำหรับ final leakage-free Optuna เพราะ augmentation หลักต้องเกิดภายใน training fold

## 6. `Model` — dataset และ input

### `Model\All_coef`

cohort coefficient raw ชุดรวมปัจจุบัน แยก `left` และ `right` ใช้เป็น input หลักของ `coef_raw`,
`coef_plsda` และ direct PLS-DA โดยมี manifest กำกับว่าไม่มี PLS-DA global และไม่มี synthetic test row

### `Model\All_coef_Ds004469`

cohort coefficient ของข้อมูล Ds004469 ชุดปัจจุบัน แยกซ้าย/ขวา ใช้ใน Optuna เป็น cohort หนึ่ง
ไม่ใช่โฟลเดอร์ผลเก่าของ `Model\Ds004469`

### `Model\All_coef_Ds005602`

cohort coefficient ของข้อมูล Ds005602 ชุดปัจจุบัน แยกซ้าย/ขวา ใช้ใน Optuna เป็น cohort หนึ่ง

### `Model\Output_Dataset` / `Model\output_dataset`

โฟลเดอร์ feature export ที่มี coefficient, XYZ, mesh edges, train/test และ balanced files
ใน Windows อาจเห็นการสะกดตัวพิมพ์ต่างกัน แต่ต้องตรวจ path จริงก่อนใช้งาน; code หลักอ้าง `Output_Dataset`

- `*_train_*`: input train ของ runner ที่อ่านจาก Output_Dataset
- `*_test_*`: untouched test ที่ใช้ประเมิน
- `*_balanced_*`: balanced/exploratory input สำหรับ protocol รุ่นเก่าหรือการตรวจเทียบ
- `.json`: manifest ของ feature contract และ provenance

### `Model\All_Augment_tain`, `Model\Ds004469`, `Model\Ds005602`

เป็น dataset/ผลจาก training pipeline รุ่นก่อน มี CSV, NPZ, plot และผล model เก่าอยู่ จึงเก็บข้อมูลไว้
เพื่ออ้างอิง แต่ training entry point รุ่นเก่าถูกลบแล้ว และโฟลเดอร์เหล่านี้ไม่ใช่ source หลักของ final Optuna

### `Model\Legacy_PLS_Models_READONLY`

พื้นที่อ้างอิง source ของ PLS model รุ่นเก่า ปัจจุบันไม่มี active entry point แล้ว ใช้ดูประวัติเท่านั้น

### `Model\Legacy_PointNet_READONLY`

พื้นที่อ้างอิง PointNet รุ่นเก่า ปัจจุบันไม่มี active entry point แล้ว ไม่ใช้สร้างผล final ใหม่

## 7. `Model` — code และ runtime

### code ที่ใช้จริง

- `data_contract.py`: schema, provenance, patient grouping, overlap/hash checks
- `leakage_free_coef_training.py`: coefficient raw runner และ model implementations
- `leakage_free_plsda_training.py`: fold-local PLS-DA augmentation
- `leakage_free_pointnet_training.py`: raw XYZ PointNet
- `leakage_free_pointnet_plsda_training.py`: PointNet PLS-DA
- `leakage_free_plsda_classifier.py`: direct PLS-DA classifier
- `optuna_leakage_free_all.py`: orchestrate tuning, winner selection และ final runs
- `optuna_leakage_free_all.py`: entry point สำหรับ Optuna; เรียกผ่าน PythonSlicer พร้อมตั้ง training `PYTHONPATH` ตาม RERUN guide
- `verify_leakage_free_pipeline.py`: preflight dataset และ model interface
- `audit_*.py`: ตรวจ summary/output ของ runner รุ่น leakage-free

### `Model\_training_site`

Python package runtime ที่ใช้กับงาน model เช่น pandas, sklearn, matplotlib, Optuna และ package ที่เกี่ยวข้อง
ใช้เมื่อ SlicerSALT Python หลักไม่มี dependency ของ training ครบ

### `Model\_training_cuda_site`

runtime ของ PyTorch CUDA (`torch 2.8.0+cu128`) สำหรับ neural model และ PointNet ที่รันบน GPU
ห้ามลบ เพราะ runner และผลที่ได้อาศัย runtime นี้

### `Model\_mplconfig`

พื้นที่ config/cache ของ matplotlib เพื่อไม่ให้ matplotlib เขียนไฟล์ลงโฟลเดอร์ระบบหรือ user profile
ไม่ใช่ dataset และสร้างใหม่ได้ แต่เก็บไว้ช่วยให้ rerun เหมือนเดิม

## 8. `Model` — ผลการ train/evaluation รุ่น leakage-free

โฟลเดอร์ต่อไปนี้เป็นผลจาก runner ก่อนรอบ Optuna เต็ม ใช้เปรียบเทียบและตรวจ contract:

- `leakage_free_runs`: coefficient raw/no-PLS baseline
- `leakage_free_plsda_runs`: coefficient fold-local PLS-DA
- `pointnet_runs`: raw XYZ PointNet baseline
- `pointnet_plsda_runs`: PointNet fold-local PLS-DA
- `plsda_classifier_runs_10fold`: direct PLS-DA seed 42
- `plsda_classifier_runs_10fold_seed123`: direct PLS-DA seed 123
- `plsda_classifier_runs_10fold_seed2026`: direct PLS-DA seed 2026

ในแต่ละโฟลเดอร์มักมี `summary_*.csv`, `FINAL_SUMMARY.csv`, `FINAL_AUDIT.json`, model artifact,
metrics, manifest และ test predictions ใช้ตรวจย้อนหลังได้ แต่ผลรวมสำหรับรายงานล่าสุดควรอ้างจาก Optuna

## 9. `Model\optuna_runs_10fold_all` — ผลหลักปัจจุบัน

นี่คือโฟลเดอร์ผลลัพธ์หลักของการเปรียบเทียบปัจจุบัน

### `studies`

เก็บผล Optuna แยกตาม protocol/cohort/side/model รวม 90 studies และ trial records ของแต่ละ study

### `final`

เก็บ final single model ที่เลือกจาก OOF แล้ว แยก protocol/cohort/side/model/seed
แต่ละ run มี `run_manifest.json`, metrics, model artifact และ `test_predictions.csv`

### CSV ระดับราก

- `OPTUNA_TUNING_SUMMARY.csv`: ผล tuning และ OOF
- `OPTUNA_FINAL_SUMMARY.csv`: ผล test ของทุก final seed
- `OPTUNA_WINNERS_BY_COHORT_SIDE.csv`: ผู้ชนะภายใน protocol
- `OPTUNA_GLOBAL_WINNERS_BY_COHORT_SIDE.csv`: ผู้ชนะสุดท้าย 6 cohort-side
- `OPTUNA_WINNER_TEST_REPORT.csv`: mean/SD จาก seed 42, 123, 2026
- `OPTUNA_AUDIT.csv`: ตรวจ trial, fold และ failure
- `RUN_MANIFEST.json`: กติกาของ run ใหญ่ เช่น folds, seeds และ test policy

เมื่อเขียนผลวิจัยให้ใช้ `OPTUNA_GLOBAL_WINNERS_BY_COHORT_SIDE.csv` และอ้างรายละเอียดจาก
`OPTUNA_WINNER_TEST_REPORT.csv` ไม่ใช้ไฟล์ smoke หรือ summary รุ่นเก่า

## 10. `DesktopApp`

- `DesktopApp\main.py`: entry point ของ application
- `DesktopApp\LeftUI`: workflow ฝั่งซ้าย เช่น import, FastSurfer, ICP, SPHARM และผลลัพธ์
- `DesktopApp\RightUI`: viewer และ workflow ฝั่งขวา
- `DesktopApp\models\predictor.py`: เตรียม feature และเรียก model inference
- `DesktopApp\app_history.json`: ประวัติ/สถานะการใช้งาน app

DesktopApp ไม่ได้ใช้แทนการ train; ใช้ model artifact และ feature contract ที่สร้างจาก pipeline
เพื่อทำนายข้อมูลใหม่และแสดง mesh/result ให้ผู้ใช้

## 11. `Visualize`

### `Visualize\3D_Mesh_Viewers`

ดู aligned mesh, mean shape, SPHARM/ICP comparison, PLS-DA surfaces และ Grad-CAM output

### `Visualize\Data_Plots`

สร้าง ROC, violin, bootstrap และ plot จาก CSV/summary เพื่อวิเคราะห์และนำเสนอผล

โฟลเดอร์นี้เป็นเครื่องมือแสดงผล ไม่ได้เป็น input ให้ model และไม่ควรนำ plot กลับเข้า training

## 12. `Model_Results_Excel`

โฟลเดอร์นี้ใช้จัดผลสำหรับรายงานวิจัย:

- `01_Excel_Workbooks`: workbook ที่สร้างแล้ว
- `02_Bootstrap_and_Statistical_Tests`: bootstrap และ statistical outputs
- `03_Performance_Summary_CSVs`: ตาราง performance
- `04_Scripts`: script สร้างรายงาน/ตาราง/plot
- `05_Bootstrap_Violin_Plots`: violin ของ bootstrap
- `06_ROC_Curves`: ROC curves
- `07_Detailed_Model_Plots`: plot แยก model
- `08_PLSDA_Class_Violin_Plots`: plot class ของ PLS-DA
- `09_PLSDA_Score_Plots`: PLS-DA score plots
- `10_GradCAM_and_Distance_Mapping`: ผลวิเคราะห์เชิงรูปร่าง/attention รุ่นก่อน

เป็น output/reporting layer ไม่ใช่ source dataset หลัก และไม่ควรใช้ไฟล์ในนี้แทน
`OPTUNA_FINAL_SUMMARY.csv` เมื่อทำตัวเลข final ใหม่

## 13. `Model_Evaluation_and_Benchmarks`

`results_csv\bootstrap_1000_summary.csv` เป็น benchmark/report รุ่นก่อน ใช้ตรวจเทียบประวัติได้
แต่ผล final ที่แก้ leakage แล้วให้ยึด `Model\optuna_runs_10fold_all`

## 14. `tests`

- `test_pipeline_regressions.py`: ตรวจ data contract, label, patient split, reference, pipeline status,
  predictor schema และ evaluation utilities
- `smoke_legacy_svm.py`: smoke test รุ่นเก่าที่เก็บไว้เป็น reference ไม่ใช่ training command ปัจจุบัน

คำสั่งรัน:

```bash
cd ~/Desktop/17-9-2569/Hippocampal-Shape-Analysis-for-Epilepsy-Detection || exit 1
set -euo pipefail
PROJECT_ROOT="$(pwd)"
MODEL_ROOT="$PROJECT_ROOT/Model"
PYTHON_SLICER="/c/Program Files/SlicerSALT 6.0.0/bin/PythonSlicer.exe"
export PYTHONPATH="$(cygpath -m "$PROJECT_ROOT");$(cygpath -m "$MODEL_ROOT");$(cygpath -m "$MODEL_ROOT/_training_site")"
"$PYTHON_SLICER" -W ignore::DeprecationWarning -m unittest discover -s "$PROJECT_ROOT/tests" -v
```

ผลล่าสุด 21 กันยายน 2026: 32 tests ผ่าน (`OK`), syntax 7,945 Python files ผ่าน

## 15. ควรใช้โฟลเดอร์ไหนในแต่ละงาน

| งาน | โฟลเดอร์หลัก |
|---|---|
| ตรวจ MRI/segmentation | `FastSurfer`, `run_pipeline.py`, `ICP` |
| ตรวจ/รัน ICP เดิมซ้ำ | `ICP\references\legacy_groupwise_all_381_v1`, `ICP\run_icp_legacy_reference_batch.sh`, output directory ใหม่ |
| ทำ ICP สำหรับ train/test | `ICP\rerun_fixed_reference.py`, `ICP\prepare_mask_split.py`, split manifests/status ที่เก็บใน `SPHARM\split_data` |
| รัน SPHARM | `SPHARM\run_spharm_parallel.py`, `Templates\SPHARM`, `aligned_nifti` ใน output ของ run ใหม่ |
| ตรวจ split | `SPHARM\split_data\current_split_manifest.json` และ ALL side status CSV สองไฟล์ |
| สร้าง coefficient dataset | `Data_Processing`, `Model\All_coef*` |
| สร้าง raw XYZ input | `Model\Output_Dataset` |
| รัน model เปรียบเทียบ | `Model\optuna_leakage_free_all.py` ผ่าน PythonSlicer ตามคำสั่ง Git Bash ใน RERUN guide |
| ตรวจ tuning/final | `Model\optuna_runs_10fold_all` |
| ทำ inference | `DesktopApp`, model artifact และ feature contract |
| ทำกราฟ/รายงาน | `Visualize`, `Model_Results_Excel` |
| ตรวจ code | `tests` และ `Model\verify_leakage_free_pipeline.py` |

## 16. นโยบายการเก็บไฟล์

ควรเก็บ:

- `Templates\SPHARM` และ `ICP\references\legacy_groupwise_all_381_v1`
- raw MRI/masks และ `SPHARM\split_data\current_split_manifest.json` พร้อม `ALL_Left_file_status.csv`, `ALL_Right_file_status.csv`
- `_training_site`, `_training_cuda_site`
- `Model\optuna_runs_10fold_all`
- source code ที่ระบุใน `RERUN_GUIDE_TH.md`
- report และ audit ที่ใช้ในวิจัย

ไม่ควรนำมาเป็นผล final โดยอัตโนมัติ:

- `Model\All_Augment_tain`, `Model\Ds004469`, `Model\Ds005602`
- `Model\Legacy_*_READONLY`
- ผล ICP/SPHARM ที่อยู่ใน run output เก่าและไม่มี reference/provenance ที่ต้องใช้
- `Model_Evaluation_and_Benchmarks\results_csv`
- smoke output หรือ summary ที่ไม่ได้อยู่ใน `optuna_runs_10fold_all`

ผล model และ training logs ใน `Model/` ยังเก็บไว้เพื่อเทียบย้อนหลัง ไม่ได้ลบในการ cleanup รอบนี้ เพราะยังไม่มีผล rerun รุ่นใหม่มาแทน ส่วนรายชื่อ launchers ที่ยกเลิกและรายการไฟล์/ผลเก่าที่ลบอยู่ใน `CHANGELOG_DETAILED_TH.md`
