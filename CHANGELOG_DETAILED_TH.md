# รายงานการปรับปรุงระบบทั้งหมด

เอกสารนี้สรุปการตรวจสอบ แก้ไข เพิ่มไฟล์ ปรับกระบวนการ ลบไฟล์เก่า และตรวจสอบผลทั้งหมดที่ทำกับโปรเจกต์
`Hippocampal-Shape-Analysis-for-Epilepsy-Detection` ใน session นี้ โดยเรียงตามปัญหาและเหตุผลของการแก้
เพื่อให้ใช้เป็นเอกสารอ้างอิงสำหรับการทำวิจัยและการรันซ้ำได้

วันที่จัดทำ: 19 กันยายน 2026  
โฟลเดอร์โปรเจกต์: `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection`

> ขั้นตอนที่ใช้งานปัจจุบันและตัวอย่างคำสั่งอยู่ใน [RERUN_GUIDE_TH.md](RERUN_GUIDE_TH.md); Sections 0–18 ด้านล่างเป็นบันทึกการตรวจ/แก้รอบก่อน ส่วน Section 19 เป็น cleanup และ ICP reference update ล่าสุดวันที่ 21 กันยายน 2026

## 0. รายงานตรวจบัคตั้งต้นก่อนการแก้ไข

ส่วนนี้คือจุดเริ่มต้นของงานทั้งหมด ก่อนแก้ code, dataset และ runner ใด ๆ โดยตรวจตั้งแต่ MRI
จนถึง Desktop inference ตามกระบวนการที่ผู้ใช้ระบุ:

```text
MRI -> extraction -> ICP -> SPHARM -> feature extraction -> augmentation
    -> training/evaluation -> Desktop inference
```

### 0.1 บัคและความเสี่ยงที่พบครั้งแรก

| รหัส | อาการ/หลักฐาน | สาเหตุที่ตรวจพบ | ผลกระทบก่อนแก้ |
|---|---|---|---|
| B01 | `ModuleNotFoundError: No module named 'slicer'` เมื่อใช้ `python run_spharm_batch.py` | ใช้ Python ปกติแทน SlicerSALT Python | SPHARM ไม่ได้รันด้วย runtime ที่ถูกต้อง |
| B02 | `ParaToSPHARMMeshCLP.exe` memory access error | input/reference/runtime ไม่ถูกควบคุมและตรวจไม่ครบ | batch SPHARM อาจหยุดกลางทางหรือสร้าง output ไม่ครบ |
| B03 | ผลซ้ายและขวาไม่เหมือนกันทั้งที่คิดว่าใช้โค้ดเดียวกัน | reference, path, flags หรือชุด output ของสองข้างไม่เป็นมาตรฐานเดียวกัน | coefficient และ mesh ของสองข้างเทียบกันไม่ได้อย่างมั่นใจ |
| B04 | มีไฟล์จากการรัน SPHARM เก่าและ dataset ใหม่ปะปนกัน | ใช้รายชื่อ/โฟลเดอร์เก่ากับชุดใหม่ที่มี left 373 และ right 377 | จำนวนแถวและ subject ที่ train/test อาจไม่ตรงกับชุดปัจจุบัน |
| B05 | augmentation/PLS-DA ถูกสร้างก่อนแบ่ง fold | synthetic rows จาก subject เดียวกันอาจเข้า validation/test | คะแนนสูงเกินจริงและเกิด data leakage |
| B06 | scaler/PLS-DA หรือ parameter อาจเห็น validation/test | fit transformation หรือเลือก model หลังเห็นข้อมูลประเมิน | ผล test ไม่ใช่การประเมินข้อมูลที่ไม่เคยเห็น |
| B07 | ไม่มีกติกากลางตรวจ subject/patient overlap | ชื่อไฟล์มี suffix และ side ที่ทำให้ระบุตัวบุคคลยาก | อาจมีคนเดียวกันอยู่ train และ test โดยไม่รู้ตัว |
| B08 | model entry point ซ้ำหลายชุด | แต่ละโฟลเดอร์มี `train_*.py` และ launcher ของตัวเอง | ใช้ contract, seed, augmentation และ output schema ไม่เท่ากัน |
| B09 | PointNet raw XYZ กับ PointNet-PLSDA ไม่ได้ถูกนิยามเป็น protocol แยก | input representation และการทำ PLS-DA ปะปนกัน | เปรียบเทียบผลไม่ยุติธรรมและอธิบายไม่ได้ว่า model เห็นอะไร |
| B10 | การเลือกโมเดลจาก test หรือการใช้ fold model รวมกันไม่ชัดเจน | ไม่มี OOF/manifest ที่บังคับกติกา | เสี่ยงเกิด test leakage และรายงานผลซ้ำซ้อน |
| B11 | log และ smoke output เก่าค้างอยู่ | การทดลองหลายรอบทิ้งไฟล์ไว้ใน project | สับสนว่าไฟล์ใดเป็นผล final และไฟล์ใดเป็นการทดลอง |
| B12 | test/regression บางรายการอ้าง protocol และ script รุ่นเก่า | test คาดหวัง protocol เก่ากับ training entry point ที่เลิกใช้แล้ว | หลัง cleanup test กลายเป็น false failure แม้ pipeline ใหม่ถูกต้อง |

### 0.2 ข้อสรุปจากรายงานตรวจบัคครั้งแรก

ก่อนแก้ไขยังไม่ควรนำ accuracy หรือผล test ไปอ้างว่าเป็นผลวิเคราะห์โรคที่ใช้ได้จริง เพราะยังมีความเสี่ยง
ด้าน runtime, reference consistency, dataset provenance และ leakage การแก้จึงต้องทำเป็นระบบเดียวกัน
ตั้งแต่ข้อมูลต้นทางจนถึง manifest ของผลลัพธ์ ไม่ใช่แก้เฉพาะ model ให้คะแนนสูงขึ้น

### 0.3 ลำดับการแก้ไขจากบัคแรกจนถึงสถานะปัจจุบัน

1. ตรวจโครงสร้าง pipeline และระบุจุดที่ใช้ runtime ผิด
2. ทำคำสั่ง SlicerSALT/PythonSlicer และมาตรฐาน SPHARM ซ้าย/ขวา
3. ตรวจ fixed reference และ output completeness ของ ICP/SPHARM
4. แยก dataset ใหม่ออกจากรายชื่อและ output เก่า
5. สร้าง `data_contract.py` และตรวจ schema, label, subject, patient group และ hash
6. สร้าง leakage-free runners และย้าย augmentation/PLS-DA เข้าไปอยู่ใน training fold
7. แยก raw coefficient, PLS-DA coefficient, raw PointNet, PointNet-PLSDA และ direct PLS-DA
8. เพิ่ม grouped 10-fold OOF, Optuna และการประเมินซ้ำด้วยสาม seed
9. ตรวจผล final, แก้ regression tests และลบ script/log/cache ที่ไม่อยู่ใน pipeline
10. อัปเดตคู่มือและสร้างรายงานฉบับนี้เพื่อให้ rerun และ audit ได้

## 1. ภาพรวมระบบเดิมและเป้าหมายที่ปรับ

ระบบมีลำดับการทำงานดังนี้:

```text
MRI
  -> FastSurfer / segmentation
  -> hippocampus mask extraction
  -> ICP alignment กับ fixed reference แยกซ้าย/ขวา
  -> SPHARM-PDM และ coefficient/mesh outputs
  -> feature extraction
  -> สร้าง cohort และ train/test split
  -> augmentation ภายใน training fold
  -> model training และ evaluation
  -> Desktop inference
```

ปัญหาหลักที่ตรวจพบก่อนปรับปรุงคือ:

1. การรัน SPHARM ด้วย Python ปกติทำให้เกิด `ModuleNotFoundError: No module named 'slicer'` เพราะโมดูล `slicer` เป็นของ SlicerSALT ไม่ใช่ Python ทั่วไป
2. ผลซ้ายและขวาเคยถูกสร้างด้วยคำสั่งหรือ reference ที่ไม่เป็นมาตรฐานเดียวกัน จึงเปรียบเทียบกันได้ไม่ถูกต้อง
3. `ParaToSPHARMMeshCLP.exe` อาจ crash ด้วย memory access error เมื่อ input/reference หรือ runtime ไม่ถูกต้อง
4. มี dataset และรายชื่อจากการรันเก่าอยู่ปะปนกับ dataset ใหม่ ทำให้เสี่ยงใช้จำนวนแถวหรือ test set ผิดชุด
5. การ augment และ PLS-DA รุ่นเก่าสามารถทำก่อนแบ่ง fold ทำให้ข้อมูลสังเคราะห์จากคนเดียวกันไหลเข้า validation/test ได้
6. การ fit scaler, PLS-DA หรือเลือก hyperparameter โดยเห็นข้อมูล validation/test ทำให้ผลประเมินสูงเกินจริง
7. runner รุ่นเก่ามี entry point ซ้ำจำนวนมากและใช้สัญญาข้อมูลไม่เหมือนกัน
8. มี smoke output, log เก่า, cache และ script ทดลองที่ไม่อยู่ใน pipeline ปัจจุบัน

เป้าหมายของการปรับคือให้ได้ pipeline ที่ reproducible, แยกซ้าย/ขวาชัดเจน, ไม่ใช้ test ตอนเลือกโมเดล,
ตรวจ patient-level leakage ได้ และมีผลลัพธ์ที่ตรวจสอบย้อนกลับได้ทุกขั้นตอน

## 2. การแก้ปัญหา SlicerSALT และการรัน SPHARM

### ปัญหา

คำสั่งเดิมที่ใช้ `python run_spharm_batch.py` ใช้ Python ปกติ ซึ่งไม่มีโมดูล `slicer` และไม่ใช่ runtime ของ SlicerSALT

### สิ่งที่ทำ

- กำหนดให้รันด้วย `PythonSlicer.exe` หรือ `SPHARM\run_spharm.bat` เท่านั้น
- เพิ่ม/ใช้ environment check ที่ `SPHARM\check_spharm_environment.py`
- ปรับ `SPHARM\run_spharm.bat` ให้ตรวจว่า SlicerSALT และ `bin\PythonSlicer.exe` มีอยู่จริงก่อนรัน
- wrapper รับเฉพาะ `left`, `right` หรือ `--check` เพื่อลดการสั่งผิด
- ส่ง `--slicer_exe`, `--num_workers`, input/output และ template เข้า `SPHARM\run_spharm_parallel.py` อย่างชัดเจน
- เพิ่มข้อความ error ที่บอกผู้ใช้ทันทีว่าต้องใช้ PythonSlicer เมื่อ runtime ไม่ครบ

### ข้อดี

- แก้สาเหตุของ `ModuleNotFoundError: slicer` โดยตรง
- ลดความเสี่ยงใช้ Python คนละ environment กับที่สร้าง SPHARM
- รันซ้ำจาก terminal ได้ด้วยคำสั่งเดียวกันทุกครั้ง
- ถ้า SlicerSALT หรือ PythonSlicer หาย จะหยุดก่อนสร้าง output ที่ไม่สมบูรณ์

คำสั่งมาตรฐาน:

```powershell
cmd /c ".\SPHARM\run_spharm.bat --check"
cmd /c ".\SPHARM\run_spharm.bat left"
cmd /c ".\SPHARM\run_spharm.bat right"
```

## 3. การทำให้ซ้ายและขวาใช้มาตรฐานเดียวกัน

### สิ่งที่กำหนดให้เหมือนกัน

- ใช้ pipeline SPHARM เดียวกัน
- ใช้จำนวน worker และ flags เดียวกัน
- ใช้ fixed reference แยกตาม hemisphere เพื่อให้จุด correspondence ไม่สลับกัน
- ใช้ `Templates\SPHARM\template_spharm_left.vtk` สำหรับซ้าย
- ใช้ `Templates\SPHARM\template_spharm_right.vtk` สำหรับขวา
- ตรวจ output ทั้งสองด้านด้วย `SPHARM\verify_bilateral_outputs.py`
- ใช้ชื่อไฟล์และ suffix เดียวกัน เช่น `_SPHARM.coef`, `_SPHARM_ellalign.coef`, `_SPHARM.vtk`

### เหตุผล

ซ้ายและขวาเป็นคนละ coordinate/reference space การใช้ template เดียวกันโดยไม่ระบุ hemisphere
อาจทำให้การ align ผิดทิศหรือทำให้ coefficient ของสองข้างเทียบกันไม่ได้ การแยก reference แต่ยังใช้
ค่าการรันเดียวกันทำให้แต่ละข้างสม่ำเสมอและตรวจสอบได้

### ข้อดี

- ผลซ้ายและขวาเกิดจากมาตรฐานเดียวกัน
- ลดความแตกต่างที่มาจากคำสั่งหรือ reference ไม่ใช่ความแตกต่างทางชีวภาพ
- ช่วยให้ feature extraction และ model comparison แยกต่อ side ได้ถูกต้อง
- ป้องกันการนำ output จากการรัน SPHARM เก่ามาปะปนกับ output ใหม่

## 4. การตรวจ ICP และ fixed reference

### สิ่งที่ทำ

- ตรวจการใช้ fixed reference แทนการให้แต่ละ subject เป็น reference ของกันและกัน
- คง reference contract และ metadata ของ reference ไว้สำหรับการ audit
- ตรวจว่าผลการ align ของ subject เดิมไม่เปลี่ยนเพราะมี subject อื่นเพิ่มเข้ามา
- ใช้ `ICP\reference_contract.py` สำหรับตรวจ physical scale, output spacing และ reference metadata
- ใช้ `ICP\ICP.py` และตรวจ output ตามขั้นตอนก่อนส่งต่อไป SPHARM

### เหตุผล

ถ้า reference ถูกสร้างใหม่จากชุดข้อมูลแต่ละรอบ ผล ICP จะเปลี่ยนตามลำดับไฟล์หรือจำนวน subject
ทำให้ rerun ไม่ deterministic และทำให้ผลซ้าย/ขวาหรือผลระหว่างรอบเทียบกันไม่ได้

### ข้อดี

- ผลเดิมอิง fixed reference เดิมและตรวจสอบย้อนกลับได้
- ลด batch-order dependence
- ตรวจจับ reference ที่ถูกแก้ไขหรือใช้ physical scale ไม่ตรงกันได้

## 5. การสร้าง dataset ใหม่และการเลิกใช้รายชื่อเก่า

สร้างและใช้ cohort ใหม่จากข้อมูลปัจจุบันแทนรายชื่อเก่า 264/67 แถว โดยใช้ชุดหลักดังนี้:

- `Model\All_coef`
- `Model\All_coef_Ds004469`
- `Model\All_coef_Ds005602`

แต่ละ cohort แยก `left` และ `right` เป็นคนละงานประเมิน ไม่รวมสองข้างเป็นโมเดลเดียว

### สัญญาของ raw coefficient dataset

`Model\data_contract.py` ตรวจสิ่งต่อไปนี้:

- feature coefficient ต้องมี `Coef_1` ถึง `Coef_507` ครบตามลำดับ
- ชื่อ subject และ label ต้องมีอยู่จริง
- `Class` กับ `BinaryClass` ต้องสอดคล้องกัน
- `DataType` ของ input raw/test ต้องเป็น original หรือ real
- ห้ามมี augmented/interpolated row ใน test
- train และ test ต้องไม่มี subject ซ้ำ
- train และ test ต้องไม่มี patient group ซ้ำ แม้ชื่อไฟล์จะต่างกัน
- train และ test ต้องไม่มี feature row ที่เหมือนกันทั้งแถว
- ค่า feature ต้องเป็น finite และ schema train/test ต้องเหมือนกัน
- manifest ต้องระบุ protocol `original-coef-no-pls-v1` และ `pls_da_applied_to_model_input=false`

### จำนวนข้อมูลที่ตรวจผ่าน

| Cohort | Side | Train | Test | Train class 0/1 | Test class 0/1 |
|---|---:|---:|---:|---:|---:|
| All_coef | left | 298 | 75 | 191/107 | 48/27 |
| All_coef | right | 302 | 75 | 220/82 | 55/20 |
| All_coef_Ds004469 | left | 54 | 12 | 37/17 | 10/2 |
| All_coef_Ds004469 | right | 53 | 12 | 42/11 | 11/1 |
| All_coef_Ds005602 | left | 244 | 63 | 154/90 | 38/25 |
| All_coef_Ds005602 | right | 249 | 63 | 178/71 | 44/19 |

### ข้อดี

- ทุกโมเดลใช้ข้อมูลใหม่ชุดเดียวกันและรู้แหล่งที่มาของข้อมูล
- ไม่ปะปนกับ split หรือ feature จากการรันเก่า
- ระบุ patient identity ได้จริง ทำ grouped CV ได้
- ตรวจ schema และ leakage ก่อนเริ่ม training ไม่ใช่ตรวจหลังได้คะแนนแล้ว

## 6. การแยก train/test และแก้ data leakage

### แนวทางเดิมที่มีความเสี่ยง

การสร้าง synthetic row ก่อนแบ่ง fold หรือ fit scaler/PLS-DA จากข้อมูลทั้งชุดทำให้ข้อมูลของ validation/test
มีอิทธิพลต่อการฝึกและการเลือกโมเดลได้

### แนวทางใหม่

- แบ่ง patient group ก่อน augmentation
- fit scaler เฉพาะ training fold
- fit PLS-DA เฉพาะ training fold ใน protocol ที่ใช้ PLS-DA
- สร้าง synthetic data เฉพาะ training fold
- validation ไม่ถูก augment
- test ไม่ถูก augment
- test ไม่ถูกใช้เลือก hyperparameter หรือ early stopping
- group เดียวกันไม่อยู่ทั้ง train และ validation ใน fold เดียวกัน
- final test ใช้โมเดลเดี่ยวที่ fit จาก train ทั้งหมดหลังเลือก parameter แล้ว

ไฟล์หลักที่ทำหน้าที่นี้คือ:

- `Model\leakage_free_coef_training.py`
- `Model\leakage_free_plsda_training.py`
- `Model\leakage_free_pointnet_training.py`
- `Model\leakage_free_pointnet_plsda_training.py`
- `Model\leakage_free_plsda_classifier.py`

### ข้อดี

- คะแนน test มีความหมายเป็นการประเมินข้อมูลที่ไม่เคยถูกเห็นจริง
- ลด optimistic bias จากการรั่วของข้อมูล
- เปรียบเทียบ protocol ได้อย่างยุติธรรม
- audit ได้ว่า feature transformation และ augmentation เกิดในขอบเขตใด

## 7. การออกแบบ augmentation ใหม่

มีการแยกสอง protocol เพื่อไม่ให้ความหมายของผลปะปนกัน:

### 7.1 `coef_raw`

- ใช้ raw coefficient 507 มิติ
- ใช้ fold-local balanced jitter/mixup ภายใน training fold
- ใช้ `children_per_pair=8`
- ใช้ `noise_scale=0.02`
- ไม่มี PLS-DA projection เข้าสู่ model input

### 7.2 `coef_plsda`

- fit PLS-DA ภายใน training fold เท่านั้น
- สร้าง augmented training feature จาก PLS-DA ที่ fit ใน fold นั้น
- validation/test ผ่าน transformation ที่ fit จาก train เท่านั้น

### 7.3 PointNet

- `pointnet_raw` ใช้ raw XYZ 1002 จุด x 3 แกน
- `pointnet_plsda` ใช้ PLS-DA ภายใน fold เพื่อสร้าง training cloud ตาม protocol ที่กำหนด
- PointNet ไม่ได้รับ coefficient เป็น input ใน raw XYZ baseline

### ข้อดี

- ได้ baseline raw ที่ตรงไปตรงมา
- ได้ protocol PLS-DA สำหรับเปรียบเทียบโดยไม่ใช้ PLS-DA แบบ global leakage
- จำนวน synthetic row และวิธีสร้างเท่ากันตาม protocol
- ผลของ PLS-DA และผลของ representation สามารถแยกอธิบายได้

## 8. การปรับปรุง model runner

ระบบเดิมมี training entry point ซ้ำหลายโฟลเดอร์และใช้ contract ไม่เหมือนกัน จึงทำให้รันซ้าย/ขวา
หรือคนละ cohort ได้ผลไม่สอดคล้องกัน

### runner ที่ใช้เป็นแกนกลาง

- `Model\leakage_free_coef_training.py` สำหรับ coefficient models
- `Model\leakage_free_plsda_training.py` สำหรับ coefficient fold-local PLS-DA
- `Model\leakage_free_pointnet_training.py` สำหรับ raw XYZ PointNet
- `Model\leakage_free_pointnet_plsda_training.py` สำหรับ PointNet PLS-DA
- `Model\leakage_free_plsda_classifier.py` สำหรับ direct PLS-DA classifier
- `Model\optuna_leakage_free_all.py` สำหรับรวมการ tuning และ final evaluation

### model ที่ถูกเปรียบเทียบ

- SVM
- MLP
- MobileNet
- ResNet
- ResNetAE
- SqueezeNet
- PointNet
- direct PLS-DA

### ค่าที่ทำให้ neural models เทียบกันได้

- ใช้ AdamW เป็น optimizer เดียวกัน
- tune learning rate, weight decay, batch size และ dropout ตาม model family
- กำหนด seed เดียวกันในรอบเปรียบเทียบ
- ประเมินซ้ำด้วย seed `42`, `123`, `2026`
- เลือกผู้ชนะต่อ cohort/side ไม่รวมซ้ายกับขวา
- ไม่ใช้ ensemble และไม่เฉลี่ย fold model ตอนรายงาน test

### ข้อดี

- ลดความแตกต่างที่เกิดจาก runner คนละชุด
- เปรียบเทียบ architecture ด้วย training contract เดียวกัน
- ผลของแต่ละ cohort/side ยังแยกชัดเจน
- ไม่มีโมเดลรวมซ้าย/ขวาที่อาจบดบังความแตกต่างของ hemisphere

## 9. Optuna และ 10-fold grouped OOF

เพิ่มไฟล์:

- `Model\optuna_leakage_free_all.py`
- `Model\run_optuna_leakage_free_all.ps1`
- `Model\OPTUNA_README_TH.md`

คำสั่งที่ใช้จริง:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\Model\run_optuna_leakage_free_all.ps1 `
  --protocol all --cohort all --side all --model all `
  --folds 10 --n_trials 10 --tune_epochs 20 --tune_patience 5 `
  --final_epochs 40 --eval_seeds 42,123,2026 `
  --device cuda --children_per_pair 8 --noise_scale 0.02 --pls_components 8 `
  --output_root .\Model\optuna_runs_10fold_all
```

การรันประกอบด้วย 5 protocol:

1. `coef_raw`
2. `coef_plsda`
3. `pointnet_raw`
4. `pointnet_plsda`
5. `plsda_direct`

รวม 90 studies:

- 36 coefficient raw studies
- 36 coefficient PLS-DA studies
- 6 raw PointNet studies
- 6 PointNet PLS-DA studies
- 6 direct PLS-DA studies

แต่ละ study ใช้ 10 grouped folds และ 10 trial records จากนั้นนำ parameter ที่ชนะไป fit final model
แยก seed 42, 123 และ 2026 รวม 270 final runs

### ช่วง parameter ที่ tune

| กลุ่ม | Parameter |
|---|---|
| SVM | `C` log `1e-2..1e2`, `gamma` เป็น `scale`, `auto` หรือ numeric log `1e-5..1e-1` |
| coefficient neural | `lr 1e-4..3e-3`, `weight_decay 1e-6..1e-2`, batch `8/16/32`, dropout `0.10..0.50` |
| ResNetAE | เพิ่ม `recon_weight 0.01..0.20` |
| PointNet | learning rate, weight decay, batch `8/16/32`, dropout `0.10..0.60` |
| direct PLS-DA | `n_components` จาก `2,4,6,8,12,16,24` |

### หลักการเลือก parameter

- objective คือ grouped 10-fold OOF balanced accuracy
- ไม่อ่าน test ระหว่าง Optuna tuning
- ไม่เลือกจาก test accuracy
- หลังเลือกแล้ว fit final model เดี่ยวจาก training data
- รายงาน test แยกตาม seed เพื่อเห็นความแปรปรวน

### ข้อดี

- ลดการเลือก hyperparameter แบบลองกับ test ซ้ำ ๆ
- OOF score มาจากข้อมูลที่ fold นั้นไม่ได้ใช้ fit
- สามารถรายงานความไม่แน่นอนจากหลาย seed
- ผู้ชนะถูกเลือกแยกต่อ protocol/cohort/side อย่างโปร่งใส

## 10. ผลลัพธ์ Optuna ที่สร้าง

โฟลเดอร์ผลลัพธ์: `Model\optuna_runs_10fold_all`

- `OPTUNA_TUNING_SUMMARY.csv` — trial และ OOF score ของ 90 studies
- `OPTUNA_FINAL_SUMMARY.csv` — ผลของ 270 final runs
- `OPTUNA_WINNERS_BY_COHORT_SIDE.csv` — ผู้ชนะภายในแต่ละ protocol/cohort/side
- `OPTUNA_WINNER_TEST_REPORT.csv` — ค่าเฉลี่ยและ SD ของผู้ชนะจาก 3 seeds
- `OPTUNA_GLOBAL_WINNERS_BY_COHORT_SIDE.csv` — ผู้ชนะสุดท้าย 6 กลุ่ม
- `OPTUNA_AUDIT.csv` — จำนวน trial, fold และ failure/prune ต่อ study
- `final\...\run_manifest.json` — manifest ราย final run
- `final\...\test_predictions.csv` — prediction ราย test subject

### ผู้ชนะสุดท้ายตาม OOF balanced accuracy

| Cohort | Side | Protocol | Model | OOF BA | Test accuracy | Test BA |
|---|---|---|---|---:|---:|---:|
| All_coef | left | coef_raw | MLP | 0.866908 | 0.791111 | 0.790895 |
| All_coef | right | coef_raw | MLP | 0.856874 | 0.857778 | 0.818182 |
| All_coef_Ds004469 | left | pointnet_raw | PointNet | 0.607313 | 0.722222 | 0.433333 |
| All_coef_Ds004469 | right | pointnet_plsda | PointNet | 0.906926 | 0.444444 | 0.242424 |
| All_coef_Ds005602 | left | coef_plsda | MLP | 0.907359 | 0.825397 | 0.816491 |
| All_coef_Ds005602 | right | coef_plsda | MLP | 0.901448 | 0.825397 | 0.810207 |

ตารางนี้แสดงว่าการเลือกจาก OOF ไม่ได้ทำให้ test สูงเสมอไป โดยเฉพาะ cohort ที่มี test ขนาดเล็ก
จึงควรรายงาน OOF, test, balanced accuracy และ SD ร่วมกัน ไม่ควรใช้ accuracy เพียงค่าเดียว

## 11. การแก้ Desktop inference และ schema

ตรวจและคง schema ที่ `DesktopApp\models\predictor.py` ใช้สำหรับรับ feature โดยให้:

- ชื่อ feature ถูกจัดเรียงตาม `feature_names_in_`
- input ที่เป็น DataFrame หรือ Series ให้ผลเหมือนกัน
- missing, extra หรือ duplicate columns ถูกปฏิเสธ
- model ที่มี feature contract ไม่ตรงกันจะไม่ถูกโหลดน้ำหนักอย่างเงียบ ๆ
- prediction เชื่อมกับ side และ feature provenance ที่ถูกต้อง

### ข้อดี

- ลดความเสี่ยง desktop inference ใช้ลำดับ coefficient ผิด
- แจ้ง error เมื่อ input ไม่ตรงกับ model แทนการคืนผลที่ดูเหมือนถูกต้อง
- ทำให้ training feature schema และ inference feature schema เป็น contract เดียวกัน

## 12. การปรับเอกสาร

### `RERUN_GUIDE_TH.md`

เพิ่มรายละเอียดต่อไปนี้:

- path และ environment ที่ต้องกำหนด
- คำสั่ง SlicerSALT/PythonSlicer
- ขั้น MRI, extraction, ICP, SPHARM, split และ feature extraction
- คำสั่ง runner รุ่น leakage-free
- Optuna 10-fold และ seed ที่ใช้
- output files และเกณฑ์ audit
- คำสั่ง regression test
- นโยบายว่าไม่ให้เรียก legacy launcher ที่ถูกลบ

### `README.md`

เปลี่ยนส่วน training เดิมที่เรียก `Model\run_everything.ps1` ให้ชี้ไปที่
`RERUN_GUIDE_TH.md` และคำสั่ง Optuna ปัจจุบัน เพื่อไม่ให้ผู้ใช้เริ่มจาก batch runner รุ่นเก่า

### `Model\OPTUNA_README_TH.md`

อธิบาย protocol, search space, OOF objective, seed และไฟล์สรุปของ Optuna

### ข้อดี

- มี single source of truth สำหรับการรันซ้ำ
- ผู้วิจัยคนอื่นสามารถตรวจได้ว่าใช้ dataset, seed, fold และ output ใด
- ลดการใช้คำสั่งเก่าที่อาจสร้าง leakage หรือใช้ dataset ผิดชุด

## 13. การแก้ regression tests

แก้ `tests\test_pipeline_regressions.py` สองส่วน:

1. เปลี่ยน synthetic prediction test ให้ใช้ `PROTOCOL` ปัจจุบันจาก `data_contract.py` แทน protocol เก่า
2. เปลี่ยน test ที่เคยคาดว่าจะมี `train_*.py` รุ่นเก่า 74 ไฟล์ ให้ตรวจว่า legacy entry point ถูกลบหมด และ current runner หลัก parse ได้

### เหตุผล

หลัง cleanup แล้ว test เดิมจะ fail เพราะคาดหวังไฟล์ที่ตั้งใจลบและใช้ protocol ที่ไม่ใช่ของระบบปัจจุบัน
การแก้ test ทำให้ test ตรวจ architecture ที่ใช้อยู่จริง ไม่ใช่สถานะเก่า

### ผล

```text
Ran 26 tests
OK
```

## 14. การลบไฟล์เก่าและสคริปต์ที่ไม่ใช้

### ลบ output/log เก่า

- `Model\optuna_smoke_direct`
- `Model\optuna_smoke_mlp`
- `Model\optuna_smoke_pointnet`
- `Model\optuna_smoke_svm`
- `Model\pointnet_plsda_smoke_runs`
- `Model\optuna_runs_10fold_all\CURRENT_PROGRESS_REPORT.csv`
- `legacy_svm_model_smoke.txt`
- `legacy_svm_smoke.txt`
- `verification_console.txt`
- `audit_console.txt`
- non-runtime `__pycache__` และ bytecode cache

### ลบ legacy model launchers

- training scripts ใต้ `Model\All_Augment_tain\**`
- training scripts และ `optimize_gpu.py` ใต้ `Model\Ds004469\**`
- training scripts และ `optimize_gpu.py` ใต้ `Model\Ds005602\**`
- source scripts ใต้ `Model\Legacy_PLS_Models_READONLY\**`
- source scripts ใต้ `Model\Legacy_PointNet_READONLY\**`
- `Model\run_balanced_training.ps1`
- `Model\run_dataset.ps1`
- `Model\run_everything.ps1`
- `Model\safe_legacy_entrypoint.py`
- `Model\safe_pointnet_entrypoint.py`
- `Model\train_all_6_datasets.py`
- `Model\resnet_gradcam_plsda_distancemap.py`

### ลบ splitter รุ่นเก่าและ maintenance scripts

- `SPHARM\split_all_dataset.py`
- `SPHARM\resplit_clean_data.py`
- `SPHARM\verify_all_split.py`
- `backup.py`
- `maintenance_finalize.py`
- `maintenance_migrate.py`
- `maintenance_wrappers.py`

การลบครั้งแรกมี 188 ไฟล์ ได้แก่ `.py` 96, `.ps1` 6 และ `.pyc` 86 จากนั้นลบ Grad-CAM
ที่ไม่มี reference เพิ่มอีก 1 ไฟล์ รวม 189 script/bytecode files ใน cleanup รอบนี้

### สิ่งที่ตั้งใจเก็บไว้

- dataset CSV/NPZ/PNG ทั้งหมด
- `Model\All_coef`, `Model\All_coef_Ds004469`, `Model\All_coef_Ds005602`
- `Model\Output_Dataset` และข้อมูลที่ใช้กับ Desktop inference
- `Model\optuna_runs_10fold_all`
- `Model\_training_site` และ `Model\_training_cuda_site`
- leakage-free runners และ audit scripts
- evaluation, report และ visualization scripts ที่ยังมีการอ้างอิงหรือมีหน้าที่วิเคราะห์ผล
- FastSurfer, DesktopApp, ICP และ SPHARM dependencies ที่ pipeline ต้องใช้

### ข้อดี

- ลดความเสี่ยงเรียก runner ผิดชุด
- ไม่ให้ผลเก่าถูกเข้าใจว่าเป็นผล final
- ลดจำนวน entry point ที่มี contract ต่างกัน
- ยังคงข้อมูลและผลที่จำเป็นต่อการวิจัยไว้ครบ

## 15. การตรวจสอบหลังการปรับทั้งหมด

ตรวจผ่านทั้งหมดดังนี้:

| รายการ | ผลตรวจ |
|---|---:|
| Python active pipeline files ที่ compile | 28/28 ผ่าน |
| Regression tests | 26/26 ผ่าน |
| Raw coefficient train/test contract | 6/6 cohort-side ผ่าน |
| Raw XYZ train/test contract | 6/6 cohort-side ผ่าน |
| Optuna tuning summary rows | 90 |
| Optuna final summary rows | 270 |
| Optuna audit rows | 90 |
| Final manifests | 270/270 ผ่าน |
| Audit failures | 0 |
| Final status `FAIL` | 0 |
| Prediction files | 270 |
| Prediction files ที่มี patient ซ้ำ | 0 |
| Pipeline scripts ที่จำเป็นหาย | 0 |
| Legacy target ที่ควรลบยังเหลือ | 0 |
| Non-runtime `__pycache__` ที่เหลือ | 0 |

Manifest final ทุกไฟล์ตรวจว่า:

- `folds=10`
- `optuna_trials=10`
- `test_used_during_tuning=false`
- `test_evaluation=single_final_model_no_fold_ensemble`

## 16. วิธีใช้งานระบบหลังปรับ

### ถ้ามี SPHARM และ feature อยู่แล้ว

1. ตรวจ dataset manifest และ raw train/test contract
2. ตรวจว่า `Model\All_coef*` เป็นชุดใหม่
3. รัน `Model\run_optuna_leakage_free_all.ps1`
4. ตรวจ `OPTUNA_AUDIT.csv`
5. ใช้ `OPTUNA_GLOBAL_WINNERS_BY_COHORT_SIDE.csv` ระบุผู้ชนะ 6 กลุ่ม
6. ใช้ `OPTUNA_WINNER_TEST_REPORT.csv` รายงาน mean/SD จาก 3 seeds

### ถ้าเริ่มจาก MRI ใหม่

1. FastSurfer/segmentation และ hippocampus extraction
2. ICP ด้วย fixed reference แยก left/right
3. SPHARM ผ่าน `SPHARM\run_spharm_parallel.py` โดยระบุ input/output/reference ตาม `RERUN_GUIDE_TH.md`
4. ตรวจ bilateral outputs และ split manifest
5. extract coefficient/XYZ features
6. สร้าง `All_coef` cohort และ manifest
7. รัน Optuna ตาม Section 9

### คำสั่งทดสอบหลังแก้โค้ด

```powershell
$Project = "C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
$Model = Join-Path $Project "Model"
$env:PYTHONPATH = "$Project;$Model;$(Join-Path $Model '_training_site')"
$PythonSlicer = "C:\Program Files\SlicerSALT 6.0.0\bin\PythonSlicer.exe"
& $PythonSlicer -W ignore::DeprecationWarning -m unittest discover -s (Join-Path $Project "tests") -v
```

## 17. ขอบเขตและข้อควรระวังในการตีความ

- การแก้ leakage ทำให้คะแนนมีความน่าเชื่อถือขึ้น แต่ไม่ได้รับประกันว่าโมเดลใช้วินิจฉัยทางคลินิกได้ทันที
- cohort ที่มี test ขนาดเล็ก เช่น `Ds004469` มี test balanced accuracy ผันผวนสูง ต้องรายงาน SD และ confidence interval
- OOF balanced accuracy ใช้เลือก parameter; test ใช้ประเมินครั้งสุดท้าย ไม่ควรนำ test กลับไป tune ซ้ำ
- ผลซ้ายและขวาต้องรายงานแยกกันตาม design ของงานนี้
- `pointnet_raw` และ `pointnet_plsda` เป็นคนละ protocol และไม่ควรอ้างว่าเป็นโมเดลเดียวกัน
- ไฟล์ใน `Legacy_*` ที่ถูกลบเป็น source script รุ่นเก่า ไม่ใช่การลบ dataset หรือ final model output
- runtime `_training_site` และ `_training_cuda_site` เป็นส่วนหนึ่งของ reproducibility contract และไม่ควรลบ

## 18. สรุปสถานะปัจจุบัน

โค้ดปัจจุบันรองรับ dataset ใหม่, fixed left/right reference, grouped 10-fold OOF, fold-local
augmentation/PLS-DA, Optuna tuning ที่ไม่ใช้ test, final single-model evaluation หลาย seed,
audit manifest และ regression tests ที่ผ่านแล้ว คู่มือและ README ชี้ไปยังคำสั่งเดียวกัน และไฟล์ legacy
ที่ไม่อยู่ใน pipeline ถูกลบออกโดยยังรักษา data, runtime และผล final ที่จำเป็นต่อการวิจัยไว้

รายละเอียดหน้าที่ของแต่ละโฟลเดอร์อยู่ที่ [FOLDER_GUIDE_TH.md](FOLDER_GUIDE_TH.md)

## 19. กู้ ICP reference เดิมและ cleanup สำหรับ rerun — 2026-09-21

- สร้าง `ICP/references/legacy_groupwise_all_381_v1/` จาก `mean_shape.ply` ของการรัน groupwise เดิม ตรวจสถานะสำเร็จ, 381 subject names, 381 transform matrices และ aligned NIfTI grid ครบทุกไฟล์ทั้งสองข้างก่อนคัดลอก
- กู้ scale จากค่าเฉลี่ย singular values ของ linear transform ทุก subject; ได้ซ้าย `0.030540622985912882`, ขวา `0.03180055903090514`; output grid ทั้งคู่ `128×128×128`, spacing `0.015625`
- สร้าง sidecar `fixed-legacy-groupwise-reference-v1` พร้อม hash/provenance และ `independent_test_reference=false`; reference รวมข้อมูลทั้ง 381 คน ใช้ทำซ้ำเดิมเท่านั้น ไม่ใช่ held-out test reference
- เพิ่ม `ICP/run_icp_with_reference.ps1` สำหรับ batch กับ single-file; เพิ่ม `--input_file` และระบุโหมด alignment ให้ชัดใน `ICP/ICP.py`; status ใหม่บันทึก version/provenance/reference hash/parameters
- เพิ่ม `ICP/run_icp_with_reference.py` ให้เรียก fixed-reference ICP ทีละข้าง/ทีละไฟล์จาก Python พร้อม preflight และ output verification; ปรับ PowerShell wrappers ให้รอ Slicer จบก่อนตรวจ status
- ตรวจรอบซ้ายที่เริ่มจาก terminal แล้วพบว่า Slicer ทำงานต่อหลัง PowerShell โยน error; status สำเร็จและ aligned NIfTI ครบ 381/381 แต่ runner เดิมออกก่อนเขียน `run_summary.json`
- fixed-reference job ไม่สร้าง batch mean ชื่อ `mean_shape.ply` อีก เพื่อไม่ให้สับสนกับ template ที่ใช้ align
- อัปเดต README, rerun guide, folder guide, ICP README และ improvements note; ลบ ICP/SPHARM outputs ที่สร้างซ้ำได้, launcher/path เก่า และ post-SPHARM copy/split helpers ที่ไม่อยู่ใน rerun ปัจจุบัน หลังเก็บ reference bundle
- cleanup ลบ 49,609 ไฟล์ รวมประมาณ 3.816 GiB; รายชื่อ path/จำนวนไฟล์/ขนาดและรายการที่เก็บไว้บันทึกใน `CLEANUP_MANIFEST_20260921.json`
- คง `SPHARM/split_data/current_split_manifest.json`, ALL left/right status CSV, source code, Templates/SPHARM, raw data และ historical model outputs ไว้; ไม่ได้รัน ICP/SPHARM ใหม่หรือฝึกโมเดลในขั้น cleanup นี้
- ตรวจหลัง cleanup ด้วย `verify_changes.py`: 32 tests ผ่าน, syntax 7,945 Python files ผ่าน ไม่มี syntax errors; การทดสอบนี้ไม่ใช่การรัน MRI/ICP/SPHARM/model เต็มชุดใหม่
