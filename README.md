# Hippocampal Shape Analysis for Epilepsy Research

โครงการนี้ประมวลผลภาพ MRI แบบ T1-weighted เพื่อสร้างรูปร่างฮิปโปแคมปัสซ้าย/ขวา สกัด SPHARM coefficients และ XYZ features แล้วฝึกโมเดลแยกกลุ่ม Healthy/TLE ระบบนี้เป็นเครื่องมือวิจัย ยังไม่ได้รับการยืนยันให้ใช้วินิจฉัยผู้ป่วยรายบุคคล

## ภาพรวมของกระบวนการ

```text
MRI T1 → preprocessing/segmentation ด้วย FastSurfer → hippocampus masks
→ ICP alignment ด้วย reference คงที่ → SPHARM-PDM → ตรวจ mesh/labels/provenance
→ coefficient และ XYZ features → train/validation/test ตามผู้ป่วย
→ fold-local augmentation/PLS-DA → grouped CV/Optuna → held-out evaluation
→ Desktop inference
```

ขั้นตอนและคำสั่ง rerun ทั้งระบบอยู่ใน [RERUN_GUIDE_TH.md](RERUN_GUIDE_TH.md) ส่วนหน้าที่ของแต่ละโฟลเดอร์อยู่ใน [FOLDER_GUIDE_TH.md](FOLDER_GUIDE_TH.md)

## ICP reference ที่สร้างจากรอบเดิม

ไฟล์ที่เก็บไว้ใน `ICP/references/legacy_groupwise_all_381_v1/` คือ mean shape ซ้าย/ขวาที่คัดลอกจากผล groupwise ICP เดิม จำนวน 381 รายต่อข้าง พร้อม metadata ที่ตรวจสอบ hash, scale และ grid แล้ว:

| ข้าง | Template SHA-256 | Physical-to-normalized scale | Grid |
|---|---|---:|---|
| Left | `e5db3de875fed2b774ef297526ba8f600742695e35415b1ebbca9d8f0063a94d` | 0.030540622985912882 | 128³, spacing 0.015625 |
| Right | `3f8c7896bca197a48955824eb37e90c91a3c48a6fc26b2ba893af7f216deb461` | 0.03180055903090514 | 128³, spacing 0.015625 |

Reference เหล่านี้สร้างจากผู้เข้าร่วมทั้งชุดเดิม ผลตรวจ IDs ยืนยันว่ารายชื่อปัจจุบันทั้ง **373/373 ซ้าย** และ **377/377 ขวา** อยู่ใน ICP reference เก่าแล้ว จึงใช้สำหรับ **ทำซ้ำ coordinate convention เดิม** เท่านั้น และห้ามใช้สร้าง test features เพื่ออ้าง held-out test แบบอิสระ ให้ใช้ขั้นตอน fit reference จาก training masks เท่านั้นใน RERUN guide

การรันใหม่จะใช้ mean shape เก่าเป็น target คงที่และทำ pairwise ICP จึงไม่ได้รับประกันว่า transform/output จะตรงกับ groupwise run เก่าทุกไบต์ แต่ hash, scale, grid และ reference frame จะคงที่

## รัน ICP ผ่าน Terminal

ใช้ Git Bash เรียก batch runner จากโฟลเดอร์ใดก็ได้; runner เปิด Slicer โดยตรง รอจนจบ แล้วตรวจจำนวนผลและ reference provenance:

```bash
bash ~/Desktop/17-9-2569/Hippocampal-Shape-Analysis-for-Epilepsy-Detection/ICP/run_icp_legacy_reference_batch.sh
```

ใน workspace ปัจจุบัน input ซ้าย/ขวาถูกเลือกอัตโนมัติจาก `../merged_ds005602_ds004469/left_hippocampus` และ `../merged_ds005602_ds004469/right_hippocampus` (381 masks ต่อข้าง); ใช้ `--input-dir` เพื่อระบุโฟลเดอร์อื่น หรือ `--input-file` เพื่อรันไฟล์เดียว สคริปต์ตั้ง output ใหม่ใต้ `reruns/` ทุกครั้ง ตรวจ `icp_status.json`, aligned-file count และ reference hash/provenance แล้วเขียน `run_summary.json`. log อยู่ใน `icp_debug_log.txt` ของ job

### ICP สำหรับการประเมิน train/test อย่างอิสระ

การประเมินงานวิจัยให้แบ่งผู้ป่วยก่อน ICP และ fit left/right references จาก **training masks เท่านั้น** จากนั้น align train และ test กับ reference ที่ตรึงไว้ ใช้ขั้นตอนในส่วน train/test ของ [RERUN_GUIDE_TH.md](RERUN_GUIDE_TH.md) อย่านำ legacy reference ด้านบนไปแทน

## ส่วนประกอบหลัก

- `run_pipeline.py`, `extract_hippocampus.py`: preprocessing, segmentation orchestration และแยก hippocampus masks
- `ICP/ICP.py`, `ICP/reference_contract.py`: mesh creation, fixed-reference alignment และสถานะ provenance
- `ICP/run_icp_with_reference.py`: Python launcher สำหรับ batch ทีละข้างหรือ mask เดี่ยว
- `ICP/run_icp_legacy_reference_batch.sh`: Git Bash runner สำหรับ preflight และ batch สองข้าง
- `ICP/rerun_fixed_reference.py`: Git Bash/Python workflow สำหรับ train-only reference และ align train/test
- `ICP/create_legacy_reference_from_previous_run.py`: ตรวจและสร้างสำเนา reference จากผล groupwise เดิม (รันแล้ว; output อยู่ใน `ICP/references/`)
- `SPHARM/run_spharm_parallel.py`, `SPHARM/run_spharm_batch.py`: SPHARM-PDM ผ่าน SlicerSALT
- `Data_Processing/`: สกัด features/labels/metadata
- `Model/`: loaders, fold-local training, tuning และ evaluation
- `DesktopApp/`: แอปสำหรับดูผลและ inference

ข้อมูลเดิมใน `Model/` เก็บไว้เพื่อเทียบย้อนหลัง ผลเหล่านั้นไม่ถูกลบพร้อมไฟล์ ICP/SPHARM รุ่นเก่า และไม่ควรถือว่าเป็นผลจากการ rerun รอบใหม่
