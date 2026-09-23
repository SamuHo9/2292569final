# สรุปโครงการ Hippocampal Shape Analysis for Epilepsy Detection

เอกสารนี้สรุปเส้นทางที่เริ่มจาก hippocampus masks ก่อนเข้า ICP จนถึงการทำนายด้วย Desktop App โดยถือว่า MRI/FastSurfer/segmentation ได้สร้าง masks มาแล้ว ส่วนต้นทางก่อน ICP อธิบายไว้เพียงเพื่อบอกที่มาของ input อ้างอิงโค้ดและผลรันปัจจุบันใน repository นี้ ณ วันที่ 23 กันยายน 2026

ระบบนี้เป็นระบบวิจัยสำหรับวิเคราะห์รูปร่างสามมิติของ hippocampus ซ้ายและขวาจาก MRI แบบ T1-weighted เพื่อจำแนกกลุ่ม Healthy/TLE ตาม label ในข้อมูล ไม่ใช่เครื่องมือวินิจฉัยทางคลินิกที่ผ่านการรับรอง

## 1. เป้าหมายของระบบ

ระบบมีเป้าหมายหลัก 4 ส่วน

1. รับภาพ MRI หรือ hippocampus mask
2. แยกและจัดแนวรูปร่าง hippocampus ให้มี coordinate frame เดียวกัน
3. แปลงรูปร่างเป็น SPHARM coefficients และ XYZ point-cloud features
4. ฝึกและประเมินโมเดลจำแนกแยกตาม dataset และข้างซ้าย/ขวา พร้อมนำโมเดลไปใช้ใน Desktop inference

ระบบไม่ได้รวม hippocampus ซ้ายและขวาเป็น feature vector เดียวในรอบเปรียบเทียบหลัก แต่สร้างโมเดลแยกซ้ายและขวาเพื่อดูว่าข้างใดและโมเดลใดให้ผลดีกว่า

## 2. ภาพรวมเส้นทางข้อมูล

MRI T1
-> image preprocessing และ segmentation
-> hippocampus masks ซ้าย/ขวา
-> mask preprocessing
-> ICP alignment
-> SPHARM-PDM
-> SPHARM ellalign mesh และ coefficient
-> coefficient/XYZ feature CSV
-> patient-level train/test split
-> grouped 10-fold cross-validation
-> fold-local augmentation หรือ PLS-DA augmentation
-> Optuna และการฝึกโมเดล
-> held-out test evaluation
-> ตาราง metric, prediction, visualization และ Desktop inference

แต่ละขั้นมีหน้าที่ต่างกัน:

- ICP แก้ตำแหน่ง แนวแกน ทิศทาง และสเกลของรูปร่าง
- SPHARM แทนรูปร่างให้เป็น parameterized mesh และ spherical-harmonic representation
- coefficient extraction แปลงรูปทรงเป็นตัวเลข 507 ค่า
- XYZ extraction เก็บจุดสามมิติ 1,002 จุด จุดละ 3 แกน
- PLS-DA ใน protocol ปัจจุบันใช้เรียนรู้โครงสร้างเพื่อสร้างข้อมูลฝึกสังเคราะห์ ไม่ได้เปลี่ยน input ของโมเดลให้เหลือเฉพาะ PLS scores
- classifier เช่น SVM, MLP, ResNet หรือ PointNet เป็นส่วนที่ทำนาย class สุดท้าย

## 3. ชั้นข้อมูล MRI และ preprocessing

### 3.1 MRI input

ข้อมูลต้นทางเป็น MRI T1-weighted ในรูปแบบ เช่น NIfTI หรือ MGZ ระบบมีทั้งโหมดที่เริ่มจาก MRI และโหมดที่รับ hippocampus mask ที่ผ่าน segmentation มาแล้ว

ไฟล์และส่วนสำคัญ:

- run_pipeline.py
- extract_hippocampus.py
- config.py
- FastSurfer/

### 3.2 Image preprocessing

FastSurfer จะ conform ภาพให้ตรงกับข้อกำหนดของ segmentation model เมื่อจำเป็น เช่น

- ตรวจ orientation
- จัดกริดและขนาด voxel
- ปรับรูปแบบข้อมูล
- บันทึกภาพที่เตรียมแล้ว เช่น orig.mgz

ขั้นตอนนี้เกิดก่อน segmentation และไม่ใช่ขั้นตอนเดียวกับ mask preprocessing

### 3.3 Segmentation และ hippocampus extraction

FastSurfer สร้าง brain segmentation จากนั้น extract_hippocampus.py แยกบริเวณ hippocampus ออกมาเป็น binary mask

ระบบอ้าง label โดยทั่วไป:

- hippocampus ซ้าย: label 17
- hippocampus ขวา: label 53

ผลลัพธ์จะถูกแยกเป็น:

- left_hippocampus
- right_hippocampus

### 3.4 Mask preprocessing

หลัง extraction mask อาจมีรู ช่องว่าง หรือส่วนประกอบเล็กที่ไม่ต้องการ จึงมีการปรับ mask เช่น

- binary closing
- fill holes
- เลือก connected component ที่ใหญ่ที่สุด
- ตัวเลือก dilation, smoothing หรือ topology fix ในบาง workflow

ค่าเริ่มต้นของ extraction workflow ใช้โหมด moderate การปรับ mask ไม่ควรสับสนกับ ICP เพราะ mask preprocessing แก้คุณภาพของ binary mask ส่วน ICP แก้ coordinate ของรูปร่าง

## 4. ICP alignment

### 4.1 ICP ทำหน้าที่อะไร

ICP หรือ Iterative Closest Point ใช้จัดรูปร่างแต่ละคนให้เทียบกับ reference frame เดียวกัน โดยจัดการเรื่อง:

- การเลื่อนจุดศูนย์กลาง
- ทิศทางแกน
- การหมุน
- การแก้ความกำกวมของ orientation
- การปรับสเกลตาม reference contract
- การคำนวณ transform matrix

ถ้าไม่จัดแนว รูปร่างของผู้ป่วยอาจแตกต่างกันเพราะตำแหน่งในภาพ ไม่ใช่เพราะรูปร่างจริง ทำให้ feature และโมเดลเรียนรู้สิ่งที่ไม่ต้องการ

### 4.2 Legacy reference ที่เก็บไว้

ใน repository มี reference ซ้าย/ขวาที่กู้คืนจาก groupwise run เดิม:

ICP/references/legacy_groupwise_all_381_v1/

ข้อมูล reference:

| ข้าง | Reference SHA-256 | Scale | Grid |
|---|---|---:|---|
| Left | e5db3de875fed2b774ef297526ba8f600742695e35415b1ebbca9d8f0063a94d | 0.030540622985912882 | 128 cubed, spacing 0.015625 |
| Right | 3f8c7896bca197a48955824eb37e90c91a3c48a6fc26b2ba893af7f216deb461 | 0.03180055903090514 | 128 cubed, spacing 0.015625 |

reference ซ้ายและขวาเป็นคนละไฟล์ เพราะ geometry ของแต่ละข้างไม่เหมือนกัน

### 4.3 จุดที่แก้ใน ICP

เดิมมีปัญหาหลายอย่าง:

- GUI ส่ง reference template แต่ ICP parser ไม่ได้ใช้งานจริง
- ICP บางโหมดสร้าง mean shape ใหม่จาก batch ที่กำลังรัน
- scale เปลี่ยนตามผู้ร่วม batch
- runner ตรวจสถานะก่อน Slicer ประมวลผลเสร็จ
- single-file และ batch ใช้ path/parameter ไม่สอดคล้องกัน
- output บางส่วนถูกมองว่าสำเร็จเพียงเพราะมีโฟลเดอร์เกิดขึ้น

สิ่งที่เพิ่มหรือแก้:

- รองรับ fixed reference อย่างชัดเจน
- เพิ่ม fit_reference สำหรับสร้าง reference จาก training masks
- บันทึก reference hash, scale, grid และ provenance
- เพิ่ม Python launcher ที่รัน batch หรือไฟล์เดียวได้
- ตรวจ icp_status.json, จำนวน output และ reference hash
- แก้ wrapper ให้รอ process tree ของ Slicer จบจริง
- แยก output ทุก run ไว้ใต้ reruns/
- ยกเลิกการรายงาน success หาก alignment ล้มเหลว

ไฟล์สำคัญ:

- ICP/ICP.py
- ICP/reference_contract.py
- ICP/run_icp_with_reference.py
- ICP/run_icp_legacy_reference_batch.sh
- ICP/rerun_fixed_reference.py

### 4.4 ข้อจำกัดของ reference ปัจจุบัน

legacy reference นี้สร้างจากผู้เข้าร่วมชุดเดิมทั้งหมด 381 ราย จึงเหมาะกับ:

- การทำซ้ำ coordinate convention เดิม
- การเปรียบเทียบกับผลเก่า
- การตรวจความสอดคล้องของ pipeline

แต่ไม่ควรอ้างว่าเป็น held-out independent test reference เพราะ test subjects เดิมอาจมีส่วนใน reference

สำหรับงานวิจัยที่ต้องการป้องกัน leakage ตั้งแต่ geometry ต้อง:

1. แบ่งผู้ป่วย train/test ก่อน
2. fit ICP reference จาก train เท่านั้น
3. align train และ test เข้าหา reference นั้น
4. รัน SPHARM และสกัด feature ใหม่
5. ฝึกโมเดลจาก feature ที่สร้างด้วย reference เดียวกัน

หากเปลี่ยน ICP reference ต้องสร้าง feature และฝึกโมเดลใหม่ ห้ามนำ model weights เดิมมาต่อกับ feature geometry ใหม่

## 5. SPHARM-PDM

### 5.1 SPHARM ทำหน้าที่อะไร

SPHARM-PDM แปลงรูปร่าง mesh ให้เป็น representation ที่เปรียบเทียบกันได้ โดยใช้ spherical parameterization และ spherical harmonic coefficients

ลำดับโดยทั่วไป:

1. รับ aligned mask
2. ตรวจ topology และคุณภาพ label
3. สร้าง parameterized mesh
4. สร้าง spherical harmonic representation
5. สร้าง grid mesh
6. ทำ ellalign/realignment
7. ส่งออก mesh และ coefficient

ไฟล์สำคัญ:

- SPHARM/run_spharm_parallel.py
- SPHARM/run_spharm_batch.py
- SPHARM/realign_spharm.py
- SPHARM/verify_spharm_outputs.py
- SPHARM/standard_production_config.json

ผลลัพธ์สำคัญ:

- mesh VTK
- SPHARM coefficient
- grid VTK
- ellalign VTK
- ellalign coefficient
- processing sidecar
- verification report และ log

### 5.2 ปัญหา ParaToSPHARMMeshCLP.exe

เคยพบ native access violation เช่น:

ParaToSPHARMMeshCLP.exe อ้าง memory address 0 และหยุดทำงาน

สาเหตุหลักที่ตรวจพบไม่ใช่ reference ICP อย่างเดียว แต่เกี่ยวข้องกับ:

- GenParaMesh สร้าง mesh ว่าง
- mask มี topology ผิดปกติ
- Euler characteristic ไม่ผ่าน
- label มีรูหรือส่วนประกอบผิดรูป
- native Slicer module crash

การใช้ reference เดิมไม่สามารถแก้ mesh ว่างได้ เพราะ error เกิดก่อนขั้นที่ใช้ reference template

สิ่งที่แก้:

- ตรวจผลลัพธ์จริงของแต่ละ module
- ไม่ถือว่า Completed with errors เป็น success โดยอัตโนมัติ
- ตรวจว่า VTK มี geometry และจำนวนจุดจริง
- ตรวจ .coef, .vtk, _grid.vtk และ _ellalign.coef
- แยก worker และบันทึก failure เป็นราย subject
- เพิ่ม Windows error mode ป้องกัน dialog native error ค้าง
- ใช้จำนวน worker มาตรฐานเดียวกันใน rerun
- เพิ่ม rerun_failed_spharm.py สำหรับ retry เฉพาะรายการที่ล้มเหลว

subject ที่ mesh ว่างหรือ topology ไม่ผ่านต้องแก้ segmentation/mask และทำ QC ก่อน ไม่ควรสร้าง feature ปลอมจาก output ที่เสีย

## 6. Feature extraction

### 6.1 Coefficient features

extract_ml_features_coef.py แปลง SPHARM coefficients เป็น CSV

รูปแบบหลัก:

- 1 แถวต่อ 1 subject
- Coef_1 ถึง Coef_507
- metadata เช่น Subject, Group, Class, BinaryClass, DataType

ดังนั้น coefficient model มี input dimensionality 507

### 6.2 XYZ features

extract_ml_features.py แปลง mesh เป็น XYZ point cloud

ปัจจุบัน PointNet ใช้:

- 1,002 points
- 3 axes ต่อ point
- รวม 3,006 scalar values
- reshape เป็น 3×1002

### 6.3 Feature contract และ sidecar

ทุก feature CSV ควรมี sidecar JSON ที่ระบุ:

- source mesh/coefficient variant
- ellalign หรือ variant ที่ใช้
- reference provenance
- input/output hash
- จำนวน feature
- จำนวน synthetic row
- label provenance

ตัว loader จะปฏิเสธ:

- schema ไม่ตรง
- feature ไม่ครบ
- ค่า non-finite
- label unknown
- train/test overlap
- output ที่ไม่มี sidecar
- feature ที่มาจาก variant คนละแบบ

## 7. Dataset split ปัจจุบัน

ชุดปัจจุบันอยู่ที่:

Model/current_spharm_80_20_20260922/

การแบ่ง train/test ทำระดับผู้ป่วยด้วย split seed 20260919 และใช้ assignment เดียวกันทั้งซ้ายและขวา

| Dataset/side | Train | Test | Train class 0/1 | Test class 0/1 |
|---|---:|---:|---:|---:|
| All_coef/left | 295 | 74 | 191/104 | 48/26 |
| All_coef/right | 302 | 75 | 220/82 | 55/20 |
| Ds004469/left | 54 | 12 | 37/17 | 10/2 |
| Ds004469/right | 53 | 12 | 42/11 | 11/1 |
| Ds005602/left | 241 | 62 | 154/87 | 38/24 |
| Ds005602/right | 249 | 63 | 178/71 | 44/19 |

หลักการสำคัญ:

- train และ test ไม่มี patient overlap
- test ใช้ข้อมูลจริงและไม่ทำ augmentation
- left และ right แยกกัน
- class count ของแต่ละ cohort/side ไม่เท่ากัน
- Ds004469 มี test class 1 น้อยมาก จึงทำให้ sensitivity และ balanced accuracy ไม่เสถียร

ไฟล์ตรวจสอบ split:

- SPHARM/split_data/current_spharm_80_20_20260922/split_manifest.json
- SPHARM/split_data/current_spharm_80_20_20260922/sample_manifest.csv
- SPHARM/split_data/current_spharm_80_20_20260922/summary_counts.csv
- Model/current_spharm_80_20_20260922/IMPORT_MANIFEST.json

## 8. PLS-DA augmentation

### 8.1 PLS-DA ในโครงการนี้ใช้ทำอะไร

PLS-DA ถูกใช้เพื่อสร้างข้อมูล training สังเคราะห์ โดย fit จาก training fold เท่านั้น

ลำดับคือ:

1. fit StandardScaler จาก fold training
2. fit PLS-DA จาก fold training
3. แปลงข้อมูลเป็น PLS scores
4. จับคู่ข้อมูล class เดียวกันที่อยู่ใกล้กัน
5. interpolation ใน score space
6. inverse transform กลับเป็น coefficient หรือ XYZ
7. เพิ่มเฉพาะ class ที่มีจำนวนน้อยกว่า
8. train classifier ด้วยข้อมูลจริงและข้อมูลสังเคราะห์

PLS-DA ไม่ได้ถูกส่งเข้า classifier เป็น feature 8 ค่าโดยตรง

### 8.2 n_components และ PLS components

n_components คือจำนวน latent axes ที่ PLS-DA เรียนรู้

fixed baseline และ PointNet PLS-DA ใช้:

- requested components = 8
- actual components = 8

architecture-optimized ให้ Optuna เลือกค่าจาก 2, 4, 6, 8, 12, 16, 24 ตาม model/dataset/side

ภายใน dataset/side/model เดียวกัน จำนวน components ที่เลือกจะคงที่ แต่ PLS weights และ scores จะ fit ใหม่ในแต่ละ fold

### 8.3 Synthetic ต่อ fold

จำนวน synthetic row คำนวณจาก:

synthetic = absolute(class 0 rows - class 1 rows)

ค่าปัจจุบัน:

| Dataset/side | Synthetic ต่อ fold | รวม 10 folds | Final fit |
|---|---:|---:|---:|
| All_coef/left | 77–79 | 783 | 87 |
| All_coef/right | 124–125 | 1,242 | 138 |
| Ds004469/left | 17–19 | 180 | 20 |
| Ds004469/right | 27–29 | 279 | 31 |
| Ds005602/left | 59–61 | 603 | 67 |
| Ds005602/right | 96–98 | 963 | 107 |

ยอดรวม 10 folds เป็นยอดสะสมระหว่าง CV ไม่ใช่ขนาด final dataset

Final fit ใช้ train ทั้งหมดครั้งเดียว เช่น All_coef/left:

- train เดิม 295
- class 0/1 = 191/104
- synthetic = 87
- final fit rows = 382

### 8.4 children_per_pair

children_per_pair=8 หมายถึงคู่ข้อมูลหนึ่งคู่ถูกใช้สร้าง interpolation pattern ได้ 8 ตำแหน่งก่อนวนกลับมาใช้คู่เดิม

ค่านี้ไม่ใช่จำนวน synthetic rows และไม่ใช่จำนวน PLS components

## 9. Protocol ของโมเดล

### 9.1 Fixed-architecture baseline

ใช้ model topology ที่กำหนดไว้เดิม แล้วปรับ hyperparameter ของการฝึก

โมเดลหลัก:

- SVM
- MLP
- ResNet
- ResNetAE
- MobileNet
- SqueezeNet

protocol coefficient ใช้ fold-local PLS-DA augmentation และ input 507 coefficients

ค่าหลัก:

- grouped 10-fold CV
- Optuna 10 trials ต่อ study
- PLS components 8
- children_per_pair 8
- seed หลัก 42
- ประเมินซ้ำด้วย seed 42, 123, 2026
- test ใช้หลังเลือกโมเดลและ threshold แล้วเท่านั้น

### 9.2 Architecture-optimized

ใช้ Optuna ค้นหา architecture และ training hyperparameter ภายใน search space ที่กำหนด

สิ่งที่ค้นหาได้ เช่น:

- learning rate
- weight decay
- batch size
- dropout
- จำนวน hidden width/depth
- kernel size
- residual blocks
- pooling
- bottleneck
- จำนวน PLS components
- SVM C, kernel และ gamma

ผลรัน:

- 36 studies = 3 cohorts × 2 sides × 6 models
- 108 final seed runs
- failures = 0
- test balanced accuracy tuned mean = 0.6099426
- test accuracy tuned mean = 0.7121704

การเลือก architecture ใช้ OOF balanced accuracy ไม่ใช้ test set

### 9.3 PointNet

มีสอง protocol แยกกัน:

1. PointNet raw XYZ baseline
2. PointNet PLS-DA

Raw XYZ:

- รับ normalized XYZ โดยตรง
- ไม่มี PLS-DA augmentation
- เป็น baseline ของ point-cloud model

PointNet PLS-DA:

- fit PLS-DA ใน flattened normalized XYZ
- สร้าง synthetic XYZ clouds
- reshape กลับเป็น 3×1002
- train PointNet ด้วยข้อมูลจริงและข้อมูลสังเคราะห์

ผลเฉลี่ยจาก 6 dataset/side และ 3 seeds:

- Raw PointNet test accuracy mean = 0.6959065
- Raw PointNet test balanced accuracy mean = 0.6198014
- PointNet PLS-DA test accuracy mean = 0.7263959
- PointNet PLS-DA test balanced accuracy mean = 0.6246764

ผลของ PLS-DA ไม่ได้ดีขึ้นทุก cohort/side จึงต้องรายงานแยก ไม่ควรสรุปจากค่าเฉลี่ยรวมเพียงค่าเดียว

## 10. Validation และการป้องกัน data leakage

สิ่งที่ทำเพื่อป้องกัน leakage:

- split ระดับผู้ป่วย
- ใช้ patient_group_id ใน grouped CV
- fit scaler เฉพาะ training fold
- fit PLS-DA เฉพาะ training fold
- สร้าง synthetic rows เฉพาะ training fold
- validation ไม่ถูก augment
- test ไม่ถูก augment
- threshold มาจาก training OOF เท่านั้น
- Optuna ใช้ OOF metric เท่านั้น
- test ถูกประเมินหลังเลือก model แล้ว
- ตรวจ train/test patient overlap
- ตรวจ identical feature row across split
- ตรวจ schema และ feature hash
- ใช้ manifest ระบุ test_used_during_tuning=false

OOF หมายถึงแต่ละแถวใน train ถูกทำนายโดยโมเดลที่ไม่ได้ฝึกด้วยแถวนั้นโดยตรง จึงเหมาะสำหรับเลือก threshold และเปรียบเทียบโมเดลมากกว่า train accuracy

## 11. ผลเปรียบเทียบผู้ชนะตาม OOF

จาก MODEL_WINNERS_FIXED_VS_ARCH.csv:

| Dataset/side | ผู้ชนะ | Track | OOF tuned balanced accuracy |
|---|---|---|---:|
| All_coef/left | MLP | Architecture-optimized | 0.8440 |
| All_coef/right | SVM | Architecture-optimized | 0.8670 |
| Ds004469/left | ResNetAE | Fixed-topology | 0.7138 |
| Ds004469/right | ResNet | Architecture-optimized | 0.7543 |
| Ds005602/left | MLP | Architecture-optimized | 0.9116 |
| Ds005602/right | MLP | Fixed-topology | 0.8917 |

ผลนี้เป็น OOF สำหรับเลือกโมเดล ไม่ใช่หลักฐานว่าโมเดลจะทำได้เท่านี้กับผู้ป่วยใหม่ทุกกลุ่ม

## 12. Desktop inference

Desktop App อยู่ใน DesktopApp/

ส่วนหลัก:

- DesktopApp/main.py
- DesktopApp/LeftUI/
- DesktopApp/RightUI/
- DesktopApp/models/predictor.py
- DesktopApp/models/left/
- DesktopApp/models/right/

หน้าที่ของแอป:

1. เลือก input MRI หรือ mask
2. เรียก preprocessing/segmentation ตาม workflow
3. เรียก ICP และ SPHARM
4. โหลด scaler, PLS และ model weights
5. สกัด feature ให้ตรง schema
6. ทำนายแยก left/right
7. แสดง probability และ class
8. แสดง mesh และผล 3D
9. บันทึก prediction summary และ log

Desktop inference ต้องใช้ preprocessing contract และ model weights ที่ตรงกัน หากเปลี่ยน ICP reference, SPHARM variant, feature order หรือ model input ต้องบรรจุ model ใหม่ให้ตรงกัน

weights ที่อยู่ใน DesktopApp/models/ เป็นเส้นทาง deployment เดิม ไม่ได้ถูกแทนที่อัตโนมัติด้วยผล Optuna รอบล่าสุด ดังนั้นผลวิจัยใน Model/ และโมเดลที่ Desktop ใช้ต้องตรวจ manifest ให้ตรงกันก่อนนำไปใช้งานจริง

## 13. การตรวจบัคและการแก้ไขหลัก

การตรวจ baseline พบปัญหาหลักในกลุ่มต่อไปนี้:

1. ขั้นตอน MRI/FastSurfer รายงาน success แม้บาง subject ล้มเหลว
2. ICP ไม่ใช้ fixed reference ตามที่ GUI ส่งมา
3. ICP คำนวณ scale จาก batch ทำให้ผลเปลี่ยนเมื่อเปลี่ยนสมาชิกใน batch
4. runner ตรวจสถานะก่อน Slicer จบ
5. SPHARM ถือ mesh ว่างหรือ Completed with errors เป็น success
6. ParaToSPHARMMesh native crash ไม่ถูกแยกเป็น failure ราย subject
7. label unknown อาจถูกตีความเป็น class 0
8. feature schema และ provenance ไม่ถูกตรวจครบ
9. augmentation และ scaler อาจอยู่นอก fold
10. threshold/AUC/prediction บางส่วนไม่ผูกกับ model output จริง
11. Desktop ไม่ตรวจ feature order, model contract และ failure propagation

การแก้ที่ทำ:

- เพิ่ม status และ exit-code propagation
- เพิ่ม fixed-reference contract
- เพิ่ม side-specific reference และ hash
- เพิ่ม single-file/batch Python runner
- เพิ่ม SPHARM verifier และ retry เฉพาะไฟล์ล้มเหลว
- เพิ่ม feature sidecar และ provenance
- ปฏิเสธ unknown label
- แยก raw input ออกจาก pre-augmented input
- ทำ fold-local PLS-DA augmentation
- ใช้ grouped CV ระดับผู้ป่วย
- แก้ ROC-AUC และ prediction validation
- ตรวจ Desktop feature ordering
- เพิ่ม requirements-training.txt และ requirements-desktop.txt
- แยก output run ใหม่เพื่อไม่ปนกับ log เก่า
- ลบ output/launcher เก่าที่ทำให้สับสนหลังเก็บ legacy reference bundle

cleanup รอบใหญ่ลบไฟล์ประมาณ 49,609 ไฟล์ หรือ 3.816 GiB โดยเก็บรายการไว้ใน CLEANUP_MANIFEST_20260921.json

## 14. ผลการตรวจสอบโค้ด

ผลตรวจที่มีหลักฐานใน repository:

- regression/audit tests ผ่าน 32 รายการในรอบแก้ไข
- syntax scan ไม่พบ syntax error ในชุดไฟล์ที่ตรวจ
- ตรวจ train/test identity และ schema
- ตรวจ unknown labels
- ตรวจ reference metadata/hash
- ตรวจ failure propagation
- ตรวจ bootstrap และ AUC
- ตรวจ augmentation scope
- ตรวจ cohort/dataset-side หลายชุด
- architecture audit: 36 studies, 108 final runs, failures 0
- PointNet raw audit: 6 studies × 3 seeds, failures 0
- PointNet PLS-DA audit: 6 studies × 3 seeds, failures 0

การตรวจเหล่านี้ยังไม่ใช่หลักฐานว่า MRI -> FastSurfer -> ICP -> SPHARM -> Desktop ทำงานครบวงจรบนภาพใหม่ทุกกรณี

## 15. ข้อจำกัดที่ต้องเขียนในงานวิจัย

1. legacy ICP reference สร้างจากข้อมูลเดิมทั้งชุด จึงไม่ใช่ independent held-out reference
2. OOF model CV ปลอด leakage ในระดับ feature/model แต่ geometry บางรอบถูกสร้างจาก reference ที่ fit ก่อน CV จึงยังไม่ใช่ fold-local ICP/SPHARM เต็ม pipeline
3. หากต้องการประเมินแบบเข้มงวด ต้องสร้าง ICP reference และ SPHARM feature ใหม่ภายในแต่ละ outer fold
4. SPHARM บาง subject ยังมี topology/mesh failure และต้อง QC หรือแก้ segmentation
5. SPHARM template ต้องตรวจแหล่งที่มาและผู้เข้าร่วมที่ใช้สร้าง template
6. Ds004469 test มีจำนวน class 1 ต่ำ ทำให้ sensitivity และ balanced accuracy แกว่ง
7. class 0 ต้องตีความตาม label contract เพราะบาง legacy workflow อาจ map contralateral เป็น class 0
8. ยังไม่มี external validation จาก cohort อิสระ
9. ยังไม่มีการรับรอง calibration และ clinical utility
10. Desktop weights เดิมต้องตรวจความตรงกับ feature/reference contract ของโมเดลวิจัยล่าสุด
11. ผลที่ได้เป็น research classification performance ไม่ใช่การยืนยันการวินิจฉัยโรคลมชักรายบุคคล

## 16. ไฟล์อ้างอิงหลัก

เอกสารภาพรวมและการ rerun:

- README.md
- RERUN_GUIDE_TH.md
- FOLDER_GUIDE_TH.md
- AUDIT_REPORT_TH.md
- IMPROVEMENTS_TH.md
- CHANGELOG_DETAILED_TH.md

การแบ่งข้อมูลและ feature:

- Model/current_spharm_80_20_20260922/IMPORT_MANIFEST.json
- SPHARM/split_data/current_spharm_80_20_20260922/README_TH.md
- SPHARM/split_data/current_spharm_80_20_20260922/split_manifest.json

ผลโมเดล:

- Model/optuna_current_spharm_80_20_20260922/
- Model/architecture_optuna_current_spharm_80_20_20260922/
- Model/pointnet_optuna_current_spharm_80_20_20260922/
- Model/pointnet_plsda_optuna_current_spharm_80_20_20260922/

ผล PLS-DA:

- Model/PLSDA_AUGMENTATION_VALUES_TH.md
- Model/PLSDA_AUGMENTATION_COUNTS.csv

การวิเคราะห์ผล:

- Model/MODEL_COMPARISON_FIXED_VS_ARCH.csv
- Model/MODEL_WINNERS_FIXED_VS_ARCH.csv
- Model/POINTNET_COMPARISON_RAW_VS_PLSDA.csv
- Model_Evaluation_and_Benchmarks/results_csv/bootstrap_1000_summary.csv

## 17. ลำดับการ rerun ที่ถูกต้อง

ถ้าจะสร้างผลใหม่จากต้นทาง ให้ทำตามลำดับนี้:

1. ตรวจ MRI/mask และ label
2. แบ่งผู้ป่วย train/test ก่อน ICP
3. fit train-only ICP reference แยกซ้าย/ขวา
4. align train/test ด้วย reference ที่ตรงข้าง
5. รัน SPHARM train/test
6. verify mesh, coefficient, grid และ sidecar
7. extract coefficient และ XYZ
8. build model input และตรวจ manifest
9. รัน fixed baseline
10. รัน architecture-optimized
11. รัน PointNet raw และ PointNet PLS-DA
12. รัน audit outputs
13. เลือกผู้ชนะจาก OOF เท่านั้น
14. รายงาน test metric แยก seed/cohort/side
15. จึงค่อย package model สำหรับ Desktop inference

ห้ามทำลำดับกลับกัน เช่น สร้าง PLS-DA จาก test, augment test, เลือก model จาก test accuracy หรือใช้ feature จาก ICP reference คนละชุดกับ model ที่ฝึกไว้

## 18. สรุปสถานะปัจจุบัน

ส่วนที่ทำแล้ว:

- ตรวจและแก้ failure propagation
- เพิ่ม fixed-reference ICP workflow
- เก็บ legacy reference พร้อม hash/provenance
- ตรวจและแยก SPHARM failure
- สร้าง current 80/20 split
- สร้าง coefficient และ XYZ inputs
- ทำ fold-local PLS-DA augmentation
- รัน fixed baseline และ architecture-optimized
- รัน PointNet raw และ PointNet PLS-DA
- ใช้ grouped 10-fold, OOF threshold และหลาย seed
- ตรวจ output manifests และ leakage controls
- สร้างรายงานจำนวน PLS/synthetic
- push ผลและเอกสารขึ้น repository

ส่วนที่ยังต้องทำหากต้องการอ้างผลเป็นระบบใช้งานจริง:

- สร้าง train-only ICP/SPHARM ใหม่แบบ fold-local สำหรับการประเมินเข้มงวด
- แก้หรือ QC subject ที่ SPHARM topology ไม่ผ่าน
- ตรวจ provenance ของ SPHARM template
- เลือกและ package model ที่ตรง feature contract สำหรับ Desktop
- ทดสอบ MRI ใหม่แบบ end-to-end
- ทำ external validation และ clinical calibration

