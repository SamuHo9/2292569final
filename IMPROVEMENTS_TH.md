# รายละเอียดการปรับปรุงระบบและผลตรวจสอบ

วันที่: 18 กันยายน 2026

## อัปเดตการเตรียม rerun — 21 กันยายน 2026

- กู้ ICP reference ซ้าย/ขวาจากผล groupwise เดิม โดยตรวจ `icp_status.json`, 381 subject names, `T_matrices.npy` และ geometry ของ aligned NIfTI ครบ 381 ไฟล์ต่อข้าง เก็บพร้อม hash/scale/grid ที่ `ICP/references/legacy_groupwise_all_381_v1/`
- เพิ่ม `ICP/run_icp_with_reference.py` ให้รันทีละข้างหรือทีละไฟล์จาก Python, ตรวจ input/reference ก่อนเริ่ม, รอ Slicer จบ และตรวจจำนวน output/hash/status ก่อนรายงานสำเร็จ
- แก้ PowerShell wrappers ให้รอ Slicer process tree ก่อนตรวจ status; แก้กรณี Git Bash/PowerShell รายงานว่าไม่มี `icp_status.json` ทั้งที่ Slicer ยังประมวลผลต่ออยู่
- `ICP/ICP.py` บังคับระบุโหมด alignment ให้ชัด และรองรับ single-file input; status ใหม่บันทึก reference version/provenance และ flag ว่า legacy reference ไม่เป็นอิสระจาก test ของชุดเดิม
- ลบ output ICP/SPHARM, launcher รุ่นเก่าที่ใช้ path ตายตัว และ post-SPHARM split helpers ที่ไม่อยู่ใน rerun ใหม่ หลังสร้าง bundle ที่ตรวจสอบแล้ว ลบ 49,609 ไฟล์/3.816 GiB; รายการอยู่ใน `CLEANUP_MANIFEST_20260921.json`; อัปเดตคู่มือทั้งหมดที่เกี่ยวข้อง
- ผล model, training logs, raw MRI/masks และ split/status manifests ที่ต้องใช้ rerun ยังคงอยู่; ไม่ลบหรือเขียนทับผล model เก่าก่อนมี rerun ใหม่

legacy reference นี้ใช้ทำซ้ำ coordinate frame เดิมเท่านั้น เพราะมาจากผู้เข้าร่วมทั้ง 381 ราย ห้ามใช้สร้าง test inputs แล้วอ้าง held-out evaluation อิสระ ให้ใช้ train-only flow ใน [RERUN_GUIDE_TH.md](RERUN_GUIDE_TH.md)

รอบที่ผู้ใช้รันเมื่อ 21 กันยายน 2026 ซ้ายประมวลผลสำเร็จครบ 381 volumes แม้ PowerShell runner เดิมแจ้ง error ก่อน Slicer จบ; status และ aligned outputs ถูกตรวจแล้ว ส่วนขวายังต้องรันด้วย Python launcher ใหม่

ตรวจ subject IDs กับ ALL status ชุดปัจจุบันแล้ว: left **373/373** และ right **377/377** อยู่ใน legacy ICP reference ทั้งหมด ผล overlap และ hash ของ status tables อยู่ใน `ICP/references/legacy_groupwise_all_381_v1/reference_manifest.json`

ปรับโค้ดเพื่อให้การประมวลผล การแบ่งข้อมูล และการวัดผลตรวจสอบได้มากขึ้น พร้อมทดสอบส่วนข้อมูลและ SVM จริงแล้ว งานนี้ยังไม่ใช่การรับรองความถูกต้องของการวินิจฉัย และยังไม่ได้รัน MRI → Desktop ครบทั้งระบบบนภาพใหม่หลังแก้ไข

## ระบบทำอะไร และ preprocessing อยู่ที่ไหน

ระบบศึกษาความสัมพันธ์ระหว่างรูปร่างฮิปโปแคมปัสกับ label กลุ่มโรคลมชักชนิด temporal lobe epilepsy (TLE) โดยแยกประมวลผลซ้าย/ขวา:

```text
MRI
 → image preprocessing/conform ภาพตาม FastSurfer
 → FastSurfer segmentation
 → extraction: แยก mask hippocampus ซ้าย/ขวา
 → mask preprocessing: closing/fill holes ตามโหมด
 → เตรียม mesh และ ICP: จัดตำแหน่ง/แนว/สเกลเทียบ reference
 → label preprocessing ก่อน SPHARM และตรวจคุณภาพ
 → SPHARM: แทนรูปร่างด้วย spherical harmonic coefficients/จุดสามมิติ
 → features + ตรวจ label/schema/ประวัติ preprocessing
 → แบ่งผู้ป่วย train/validation/test
 → fit scaler/PLS และ augmentation เฉพาะ train fold ใน baseline ใหม่
 → training / model selection / evaluation
 → บรรจุโมเดลพร้อม preprocessing ที่ตรงกัน
 → Desktop inference และบันทึกผล
```

Preprocessing จึงมีทั้งระดับภาพ, mask, geometry และ feature การ fit ขั้นตอนที่เรียนรู้จากข้อมูลต้องไม่ใช้ validation/test ของการประเมินนั้น groupwise ICP แบบเดิมและ legacy fixed reference ที่สร้างจากผลนั้นใช้ทำซ้ำ/สำรวจเท่านั้น ไม่ใช่เส้นทางยืนยันการทำนาย held-out ผู้ป่วยชุดเดิม

## รายการแก้ไข

### 1. MRI/extraction: ไม่รายงานสำเร็จเมื่อข้อมูลไม่ครบ

ไฟล์: `run_pipeline.py`

- เพิ่ม `numpy` ที่ขาด ซึ่งเดิมทำให้เส้นทางบันทึก paired MRI ล้มเหลว
- สร้าง NIfTI จาก array กับ affine โดยไม่ส่ง header ของ MGZ ที่ไม่เข้ากัน
- ส่งสถานะล้มเหลวกลับเมื่อ paired MRI ที่จำเป็นบันทึกไม่ได้ และจบคำสั่งด้วย exit code ที่สะท้อนผลจริง
- การข้ามงานที่ทำแล้วตรวจว่ามี paired MRI ด้วยในโหมดที่ต้องใช้
- โหมดข้าม FastSurfer ไม่บังคับตรวจ checkpoint ของขั้นตอนที่ไม่ได้เรียก
- แยก temporary directory ต่อรอบและปฏิเสธชื่อ subject ซ้ำจากคนละ input เพื่อป้องกันการทับกัน

### 2. ICP: เพิ่ม reference คงที่สำหรับข้อมูลใหม่

ไฟล์: `ICP/ICP.py`, `ICP/reference_contract.py`, `DesktopApp/LeftUI/icp_panel.py`

- รองรับ `--reference_template` จริง เดิมคำสั่งจาก GUI ไม่ได้ทำให้ alignment ใช้ reference คงที่ตามที่คาด
- เพิ่ม `--fit_reference` เพื่อสร้าง reference จาก training masks และบันทึก physical scale, grid, hash ของ template และแหล่งข้อมูลลงไฟล์ `.json` คู่กัน
- fixed-reference mode ใช้สเกลและ template ที่บันทึกไว้ ไม่คำนวณใหม่จากผู้ป่วยอื่นใน batch
- ตรวจ template ถูกเปลี่ยน, metadata หาย, grid ไม่เหมาะสม และรูปร่างหลุดกรอบ
- เพิ่ม margin ตอนกำหนดกรอบ และเปลี่ยนการ export ที่ผิดพลาด/ไม่ผ่านเกณฑ์ QC ให้หยุดพร้อมสถานะไม่สำเร็จ
- แก้ชื่อพารามิเตอร์ GUI ให้ตรงกับ CLI; กำหนด reference แยกข้างผ่าน `HIPPO_ICP_REFERENCE_LEFT` และ `HIPPO_ICP_REFERENCE_RIGHT`

**ผลต่อการใช้งาน:** mean shape เดิมไม่มี metadata จึงสร้าง legacy reference bundle โดยตรวจ scale ร่วมจาก 381 transform matrices และ grid จาก aligned NIfTI ทุกไฟล์ พร้อม hash ต้นทาง ไม่ได้เติม scale โดยเดา อย่างไรก็ตาม reference ดังกล่าว fit จากผู้เข้าร่วมทั้งชุดเดิม จึงห้ามใช้ประเมิน held-out test ของ cohort นั้น หากสร้าง evaluation ใหม่ให้ fit reference จาก training masks เท่านั้น แล้ว align train/test ด้วย bundle นั้น การเปลี่ยน geometry ต้องสร้าง features และฝึกโมเดลใหม่ ห้ามนำ weights เก่ามาปะกับ feature ใหม่

### 3. SPHARM: ตรวจผลจริงและที่มาของผลลัพธ์

ไฟล์: `SPHARM/run_spharm_batch.py`, `SPHARM/realign_spharm.py`, `DesktopApp/LeftUI/spharm_panel.py`

- ตรวจสถานะ CLI แบบแยกตามโมดูล: ขั้นเตรียมข้อมูลต้องเป็น `Completed` ส่วน GenParaMesh/ParaToSPHARMMesh ของ SlicerSALT อาจรายงาน `Completed with errors` หลังเขียนผลสำเร็จ จึงยอมรับได้เฉพาะเมื่อ mesh มีจุดจริง, VTK มี geometry, coefficient และ grid ผ่านการตรวจครบ
- การประมวลผลแต่ละ subject ส่งคืนผลสำเร็จ/ล้มเหลวจริง และ batch หยุดด้วยข้อผิดพลาดเมื่อมีงานล้มเหลว
- บันทึก `spharm_status_shardN.json` และ `_processing.json` เพื่อผูกผลกับ input, template, config และ geometry
- status รุ่นปัจจุบันบันทึกโหมด, `iter/subdiv/degree`, input/output directory, required suffixes และ SHA-256 ของ template เพื่อแยกผลคนละมาตรฐานออกจากกันได้
- แก้จุดที่ `run_batch_spharm()` เดิมละเลยค่า `False` จาก `process_single_subject()` ทำให้ mesh ว่างยังถูกพิมพ์ว่า completed; ตอนนี้ตรวจ `Completed` เท่านั้น ตรวจ ParaToSPHARMMesh/grid และคืน exit code ไม่สำเร็จเมื่อมี subject ตก
- เพิ่ม `SPHARM/verify_spharm_outputs.py` สำหรับตรวจ input ทุกตัวกับ `.coef`, `.vtk`, `_grid.vtk` และ `_ellalign.coef` จาก Terminal
- launcher `SPHARM/run_spharm.bat` และ wrappers ซ้าย/ขวาเป็นวิธีจากรอบก่อน ปัจจุบันลบแล้วเพื่อลดความสับสน; ใช้ `SPHARM/run_spharm_parallel.py` ในคำสั่งที่ระบุ input/output/reference ชัดเจนตาม `RERUN_GUIDE_TH.md` โดยยังใช้ค่าผลิต SPHARM เดิมที่บันทึกไว้
- เพิ่ม Windows error mode เพื่อไม่ให้ native access violation ของ `ParaToSPHARMMeshCLP.exe` เปิด dialog ค้างทั้ง batch และกำหนด wrapper มาตรฐานเป็น 5 workers เพื่อให้ซ้าย/ขวาใช้ความขนานเท่ากัน โดย subject ที่มี topology ผิดปกติยังถูกบันทึกเป็น failed รายตัว
- ข้ามผลเดิมได้เมื่อ fingerprint ตรงและผลที่จำเป็นมีครบ ลดการใช้ไฟล์ค้างจากการตั้งค่ารอบก่อน
- ตรวจ shard, input ว่าง, reference ที่หาย และ dependency ของ preprocessing
- แก้ GUI ให้ส่ง `--template` และไม่ส่ง flags ที่ realignment ไม่รองรับ พร้อมตรวจ exit code

### 4. Feature/label: ไม่แปลงข้อมูลไม่ทราบกลุ่มเป็น Healthy

ไฟล์: `Data_Processing/extract_ml_features_coef.py`, `Data_Processing/feature_metadata.py`, สคริปต์ augmentation สองไฟล์

- Unknown ใช้ `BinaryClass=-1` และตัวโหลดฝึกจะปฏิเสธ แทนการตกเป็น class 0
- เพิ่ม `--labels_csv` โดยต้องมี `Subject,BinaryClass`; เพิ่ม `Class,Group` ได้ ตรวจค่าซ้ำ/ความสอดคล้องของ label
- เมื่อใช้ label CSV ต้องมี label ครบตาม Subject ที่ extractor สร้าง ไม่เดา label ที่ขาดจากชื่อไฟล์
- ตรวจทั้ง batch ก่อนเปิด CSV: coefficients ต้องมีจำนวนตรงกัน, ไม่ว่าง, ไม่มี NaN/Inf และใช้ preprocessing contract เดียวกัน
- เขียน sidecar ของ feature CSV พร้อม hash และแหล่ง label
- บังคับให้ coefficient ทุกตัวมี `<subject>_processing.json` ที่มี `feature_contract`; ถ้า provenance หายหรือไม่ใช่ object จะหยุดทันที ไม่ปล่อยให้ไฟล์เก่าถูกตีความว่าใช้ fixed reference
- augmentation แบบ interpolation ไม่ผสม parent ข้าม binary class; ปฏิเสธ Unknown

**นิยามที่ยังคงเดิม:** class 0 อาจรวม contralateral ตามชุดข้อมูลเดิม จึงไม่ควรแปลว่าทุกตัวอย่างใน class 0 เป็นคนสุขภาพดี การเปลี่ยนนิยามโรคต้องยืนยันกับข้อมูลกำกับและออกแบบการทดลองใหม่

### 5. การแบ่งข้อมูล: ทุกสถาปัตยกรรมใช้ผู้ป่วยชุดเดียวกัน

ไฟล์หลัก: `Model/data_contract.py`; เชื่อมสคริปต์ฝึกเดิม **74 entry points**

- โหลดข้อมูลต้นฉบับจาก split ที่กำหนดร่วมกัน แทนแต่ละโมเดลเลือก CSV ต่างกัน
- ตรวจตัวตนผู้ป่วย/label ของ coefficient และ XYZ ให้ตรงกัน จึงเปรียบเทียบ SVM กับ PointNet ได้บน test cohort เดียวกัน
- ตรวจชื่อซ้ำ, train/test overlap, feature ซ้ำข้าม split, schema, finite values และ synthetic rows
- สคริปต์ฝึกเดิมใช้ original subjects เท่านั้น ไม่ใช้ข้อมูลที่ augment ทั้งชุดมาก่อน CV
- บันทึก subject IDs, protocol และ input identity/hash ลงผลรอบใหม่
- ทุกครั้งสร้างโฟลเดอร์ใหม่ใน `Model/validated_runs/` ชื่อโฟลเดอร์นี้หมายถึง workflow ที่เพิ่มการตรวจ ไม่ใช่ clinical validation

จำนวน train/test ที่ตรวจจริงสำหรับแต่ละข้าง:

| Cohort | ซ้าย | ขวา |
|---|---:|---:|
| Ds005602 | 223 / 56 | 228 / 57 |
| Ds004469 | 41 / 11 | 43 / 11 |
| All_Augment_tain และ combind_methode | 264 / 67 | 271 / 68 |
| Ds005602Train_Ds004469test | 279 / 52 | 285 / 54 |
| Ds004469Train_Ds005602test | 52 / 279 | 54 / 285 |

### 6. เพิ่ม baseline แบบ nested CV และ augmentation ภายใน fold

ไฟล์: `Model/validated_model.py`, `Model/train_validated_baseline.py`

- outer folds วัดผลกับผู้ป่วยที่ไม่ได้ใช้เลือก hyperparameters ของ fold นั้น
- inner folds เลือกจำนวน PLS components และค่า C
- scaler, PLS และ synthetic interpolation fit จาก training fold เท่านั้น; validation/test เป็นข้อมูลต้นฉบับ
- บันทึก out-of-fold predictions, test predictions, โมเดล และรายละเอียด fold/ผู้ป่วย/ค่าที่เลือก
- ปรับเพดาน PLS และ batch size ของสคริปต์เดิมให้รองรับชุดต้นฉบับที่เล็กลง; PointNet แบบ combined ใช้ loader และ stratification ร่วมกัน

**ขอบเขต:** ยังไม่ได้แปลงทั้ง 74 สคริปต์เป็น nested CV ค่า CV ของสคริปต์เดิมที่ใช้เลือกโมเดลยังเป็น selection CV ไม่ใช่ค่าประมาณอิสระแบบ nested CV

### 7. Evaluation: AUC และคำทำนายต้องตรงกับโมเดลจริง

ไฟล์: `Model/evaluation.py`, `Model/run_bootstrap_on_all_trained_models.py`, `Model/compare_runs.py`

- ใช้ `roc_auc_score` แทน double argsort ที่จัดการคะแนนเสมอผิด; ทดสอบกรณีทุกคะแนนเท่ากันได้ AUC 0.5
- bootstrap ใช้ saved `y_pred`; ไม่สร้างคำตอบใหม่จาก threshold 0.5 ซึ่งอาจต่างจาก `SVC.predict`
- ปฏิเสธ label/probability ผิดรูปแบบ และนับ bootstrap รอบที่ข้ามเพราะเหลือ class เดียว
- ใช้ RNG เฉพาะส่วนเพื่อทำซ้ำได้โดยไม่เปลี่ยน random state ของระบบอื่น
- `compare_runs.py` ตรวจ subject IDs, labels และ protocol ก่อนเปรียบเทียบ แม้ลำดับแถวต่างกัน; ห้ามเขียนทับไฟล์ output ที่มีแล้ว
- สรุป legacy bootstrap ใช้ชื่อใหม่พร้อมระบุว่า provenance เก่ายังไม่ผ่านการยืนยัน

### 8. รายงานเดิม: หยุดการสร้างตารางที่อ้างผลเก่าอย่างไม่ถูกต้อง

ไฟล์: `Model_Results_Excel/04_Scripts/generate_model_excel.py`, `generate_newest_bootstrap_results.py`

- ปิด entry point สองตัวนี้ด้วยข้อความอธิบายและชี้ไป `Model/compare_runs.py` เพราะอ่านผลเก่าที่ไม่ยืนยัน test cohort และอาจทับรายงานเดิม
- เอา t-test บน bootstrap replicates ออก ไม่แทนด้วย p-value ที่แต่งขึ้น
- ผลเปรียบเทียบใหม่เป็น CSV จาก predictions ที่ระบุตัวผู้ป่วยได้ ยังไม่ได้สร้าง workbook รูปแบบเดิมทดแทน
- ไฟล์ Excel/Word/ภาพทางสถิติที่มีอยู่ก่อนยังเป็นหลักฐานการทดลองเก่า การเปิดไฟล์ได้ไม่ทำให้ข้อสรุปเดิมได้รับการยืนยันใหม่

### 9. Desktop inference: ตรวจลำดับฟีเจอร์และรุ่น preprocessing

ไฟล์: `DesktopApp/models/predictor.py`, `DesktopApp/LeftUI/result_panel.py`

- รักษาชื่อคอลัมน์ DataFrame/Series แล้วเรียงตาม scaler/feature schema ที่ฝึกไว้
- ปฏิเสธคอลัมน์ซ้ำ/ขาด/เกิน, รูปร่าง input ผิด, input ว่าง และ NaN/Inf
- coefficient ที่มี preprocessing contract ใหม่ต้องตรงกับ `model_manifest.json` ของโมเดล และต้องเป็น fixed-reference mode
- ไม่รายงานสำเร็จเมื่อทำนายทั้งหมดล้มเหลวหรือบันทึกผลไม่ได้

**ยังไม่ได้เปลี่ยน weights ของ Desktop:** baseline SVM ใหม่ไม่ใช่ checkpoint ResNet ทดแทนโดยอัตโนมัติ เส้นทาง legacy ที่ไม่มี contract ยังไม่ได้รับการยืนยันทาง geometry การใช้งานรุ่นใหม่ต้องฝึก/บรรจุโมเดลที่ตรงกันก่อน

### 10. คำสั่งรันและ dependencies

ไฟล์: `run_everything.bat`, `setup.bat`, `Model/train_all_6_datasets.py`, `Model/run_dataset.ps1`, launchers ของ dataset และ requirements

- หยุดเมื่อขั้นตอนก่อนหน้าส่ง exit code ล้มเหลว แทนการไปต่อแล้วแจ้งสำเร็จ
- ฝึกครบรายการ 6 datasets ตามชื่อ runner, จัดเส้นทาง combined PointNet ให้ถูก และสร้าง log/summary แยกแต่ละรอบ
- เพิ่ม dependency ที่ขาด พร้อม `requirements-training.txt` และ `requirements-desktop.txt`
- README ระบุ Python 3.10+ ตามความต้องการ FastSurfer และเชื่อมรายงานนี้

## ผลทดสอบที่ทำจริง

- ใช้ PythonSlicer 3.9.10 ที่มีอยู่ในเครื่องรัน regression tests และ sklearn; ไม่ได้ติดตั้ง dependencies เพิ่ม
- รายละเอียดผลปัจจุบัน: `regression_results.txt`, `regression_summary.json`
- ผลตรวจล่าสุด 21 กันยายน 2026: **32 tests ผ่านทั้งหมด**, syntax ของ Python **7,945 ไฟล์ไม่พบข้อผิดพลาด** (`verify_changes.py`; การตรวจนี้ไม่ใช่ end-to-end MRI/SPHARM/GUI/inference run)
- ทดสอบ schema/predictor, unknown labels, reference metadata/hash, failure propagation, bootstrap, augmentation ภายใน fit, cohort ทั้ง 12 คู่ dataset/side และสคริปต์ฝึก 74 ตัว
- ส่วนที่ต้องใช้ Slicer/VTK/MRI/Qt/torch บางรายการใช้ AST extraction และ mocks เพื่อทดสอบตรรกะ จึงไม่ใช่ end-to-end test
- ตรวจ syntax ของ source Python และ parser ของ PowerShell launchers
- ฝึก nested SVM จริงและบันทึกผลใน `Model/validated_runs/Ds004469/left/nested_svm/20260918_172830_yrqfu93d/`
- ทดสอบส่วนโมเดลจาก SVM entry point เดิมจริง โดยแทนเฉพาะ plotting ด้วย stub เพราะไม่มี matplotlib; ผลใน `Model/validated_runs/Ds004469/left/SVM/20260918_173547_train_svm_pls_l1wj1hch/`
- การเรียก legacy SVM ทั้งไฟล์โดยตรงยังติด matplotlib ที่ขาด มีหลักฐานใน `legacy_svm_smoke.txt`; ผลทดสอบแยกส่วนอยู่ใน `legacy_svm_model_smoke.txt`
- เปรียบเทียบ predictions สองรอบสำเร็จหลังตรวจ identities: `Model/validated_runs/smoke_comparison.csv`

หลักฐานก่อน cleanup เคยพบว่า ICP เดิมเป็น `exploratory_groupwise` 381 รายต่อข้าง ส่วน SPHARM ผ่าน artifact check ซ้าย 373/381 และขวา 377/381; ขาด 12 ราย และ input เดิมจับคู่กันได้ 381/381 การตรวจในขณะนั้นยังไม่มี provenance sidecars จึงต้อง rerun ก่อนสกัด features ผล ICP/SPHARM และรายงาน `bilateral_spharm_verification.json` รุ่นเก่าถูกลบใน cleanup วันที่ 21 กันยายน 2026; ตัวเลขนี้เป็นประวัติ ไม่ใช่ output ที่ใช้กับ rerun ปัจจุบัน

ผล nested baseline: Ds004469 ซ้าย, train 41, test 11, outer CV 3 folds, augmentation 1 ลูกต่อ training subject:

| Metric | Nested out-of-fold | Test |
|---|---:|---:|
| Accuracy | 73.17% | 63.64% |
| Sensitivity ของ class 1 | 16.67% | 0.00% |
| Specificity ของ class 0 | 96.55% | 87.50% |
| Macro F1 | 0.5512 | 0.3889 |
| ROC AUC | 0.5431 | 0.4583 |

ใน test มี class 1 เพียง 3 ราย และรอบนี้ทำนายผิดทั้ง 3 ราย ส่วน class 0 ถูก 7/8 ราย **ผลนี้ยังไม่สนับสนุนการใช้งานวินิจฉัย** และไม่ควรเหมารวมว่าเป็นผลของทุกสถาปัตยกรรม ผลรอบนี้ใช้ historical features ซึ่งยังพิสูจน์ไม่ได้ว่า geometry เดิมเป็นอิสระจาก test ดังนั้นเป็นการตรวจ workflow ฝึก/ประเมินที่แก้ ไม่ใช่ผลยืนยันระบบภาพรุ่นใหม่

## วิธีรันซ้ำและย้ายไป preprocessing รุ่นใหม่

คู่มือคำสั่งปัจจุบันอยู่ใน [RERUN_GUIDE_TH.md](RERUN_GUIDE_TH.md): ส่วน A ใช้ legacy ICP reference ที่สร้างไว้สำหรับ batch/single repeatability; ส่วน B แบ่งผู้ป่วยก่อน ICP และ fit reference จาก train masks เท่านั้นสำหรับ evaluation ใหม่

อย่าใช้คำสั่ง verifier แบบเก่าที่ชี้ไปยัง `ICP/output_left_hippocampus` หรือ `ICP/output_right_hippocampus` เพราะผลเก่าถูกลบแล้ว ให้ตรวจ output directory ที่ runner สร้างใน `$RunRoot` เช่น:

```powershell
& 'C:\Program Files\SlicerSALT 6.0.0\bin\PythonSlicer.exe' SPHARM\verify_spharm_outputs.py `
  --input_dir "$RunRoot\icp\left\train_fixed\aligned_nifti" `
  --output_dir "$RunRoot\icp\left\train_fixed" `
  --deep
```

การประเมินใหม่ยังต้องตรวจ source ของ SPHARM template และจำไว้ว่าการสร้าง ICP reference จาก outer-train ก่อน 10-fold ทำให้ OOF ยังไม่ fold-local ทาง geometry หากจะอ้าง CV อิสระเต็ม pipeline ต้องสร้าง ICP/SPHARM/features ใหม่ในแต่ละ fold

## ข้อจำกัดและการเก็บงานเดิม

- ยังไม่ได้ทดสอบ FastSurfer inference, MRI export และ Desktop GUI ครบวงจรจริงหลังแก้; ส่วน ICP/SPHARM CLI รันกับข้อมูลจริงแล้ว แต่ยังมี SPHARM 12 รายที่ต้องแก้ segmentation/topology หรือทำ manual QC ก่อนนำไปสร้าง feature
- environment ที่ใช้ทดสอบไม่มี torch, nibabel, PyQt6, matplotlib และ VTK import ติด DLL จึงยังไม่ได้ฝึก neural models ทั้งหมดหรือแทน weights ในแอป
- ไม่มีการยืนยันด้าน sensitivity/calibration/external validation สำหรับโมเดลใช้งานจริงจากงานแก้โค้ดนี้
- เก็บสำเนา source/config ก่อนแก้ใน `maintenance_backup_20260918/`; ไม่ได้ reset การแก้ที่มีมาก่อน และผลทดลองใหม่อยู่ใน directory ใหม่
- สคริปต์ `maintenance_*.py` เป็นบันทึกการย้ายโค้ดครั้งนี้ ไม่ใช่คำสั่งใช้งานประจำและไม่ควรรันซ้ำ
- รายงานตรวจครั้งแรกอยู่ใน `AUDIT_REPORT_TH.md`; `audit_results.json` เป็นผลก่อนแก้ ส่วน `regression_summary.json` จะถูกเขียนใหม่เมื่อรัน `verify_changes.py`
- `ICP/output_left_hippocampus`, `ICP/output_right_hippocampus`, copied SPHARM result folders, root `ICP/icp_debug_log.txt` และ launcher ที่ชี้ไปยัง output เก่าถูกลบหลังสร้าง/ตรวจ reference bundle แล้ว
- Historical outputs ของโมเดลและ training logs ใน `Model/` ยังคงเก็บไว้เพื่อเปรียบเทียบ; แยกจากผล rerun ใหม่ที่กำหนด output root ใหม่
