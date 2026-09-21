# รายงานตรวจบัคกระบวนการ Hippocampal Shape Analysis

> รายงานนี้เก็บข้อค้นพบก่อนแก้ไขไว้เป็นหลักฐาน สถานะการปรับปรุง ผลทดสอบจริง และข้อจำกัดล่าสุดอยู่ใน [IMPROVEMENTS_TH.md](IMPROVEMENTS_TH.md)

วันที่ตรวจ: 18 กันยายน 2026

> **สถานะปัจจุบัน (21 กันยายน 2026):** เนื้อหาและคำสั่งในรายงานนี้เป็น baseline ก่อนแก้ โดยเฉพาะผล `exploratory_groupwise` และ launcher ที่ระบุไว้ไม่ได้เป็น entry point ปัจจุบัน ผล ICP/SPHARM เก่าและ launcher ที่ชี้ไปยัง output เหล่านั้นถูกลบหลังตรวจและเก็บ legacy reference bundle แล้ว ดูคำสั่งปัจจุบันที่ [RERUN_GUIDE_TH.md](RERUN_GUIDE_TH.md) และรายการแก้ไขที่ [IMPROVEMENTS_TH.md](IMPROVEMENTS_TH.md)

รายงานนี้เป็นผลตรวจ baseline ก่อนรอบแก้ไข ตั้งแต่ MRI → image preprocessing → segmentation → extraction → mask preprocessing → ICP → SPHARM → features และเส้นทาง augmentation/training/evaluation กับ Desktop inference พบ 11 ประเด็นที่ควรแก้ เรียงตามผลกระทบด้านล่าง ผลการแก้และสถานะล่าสุดอยู่ใน `IMPROVEMENTS_TH.md`; คำสั่ง rerun ที่ยังใช้ได้อยู่ใน `RERUN_GUIDE_TH.md`

## สรุปว่าระบบทำอะไรบ้าง

ระบบนี้มีเป้าหมายวิเคราะห์ **รูปร่างสามมิติของฮิปโปแคมปัสซ้ายและขวาจากภาพ MRI สมอง** แล้วใช้ข้อมูลรูปร่างสร้างโมเดลจำแนกกลุ่ม Healthy/TLE รวมถึงแสดงผลทำนายแยกข้างและภาพรูปร่างประกอบการวิเคราะห์ มีทั้งส่วนเตรียมข้อมูลและทดลองฝึกโมเดลสำหรับงานวิจัย กับ Desktop App สำหรับประมวลผลข้อมูลและเรียกใช้โมเดลที่ฝึกไว้

เส้นทางเตรียมรูปร่างคือ **MRI → เตรียมภาพ (image preprocessing) → FastSurfer segmentation → แยกฮิปโปแคมปัส → เตรียม mask (mask preprocessing) → ICP → ทำความสะอาด label ก่อน SPHARM → SPHARM/realignment → ฟีเจอร์** จากนั้นส่วนทดลองนำฟีเจอร์ไปฝึกและประเมินโมเดล ส่วน Desktop inference ใช้ preprocessing และโมเดลที่ฝึกไว้แล้ว

### Preprocessing อยู่ตรงไหน

Preprocessing กระจายอยู่หลายขั้นตอนของโค้ด โดยแต่ละส่วนเตรียมข้อมูลคนละชนิด:

| ส่วน | อยู่ก่อนขั้นตอนใด | สิ่งที่โค้ดทำ |
|---|---|---|
| **Image preprocessing** | ก่อน FastSurfer segmentation | FastSurfer ตรวจและ conform ภาพให้ตรงกับข้อกำหนดของโมเดล ปรับกริด/ขนาด voxel แนวภาพและชนิดข้อมูลตามพารามิเตอร์เมื่อจำเป็น แล้วบันทึกภาพที่เตรียมแล้วเป็น `orig.mgz`; อยู่ใน `FastSurfer/FastSurferCNN/run_prediction.py` ฟังก์ชัน `conform_and_save_orig()` |
| **Mask preprocessing** | หลัง extraction และก่อน ICP | `extract_hippocampus.py` เรียก `apply_mode()` กับ binary mask ซ้าย/ขวา ค่าเริ่มต้น `moderate` ทำ binary closing 1 รอบและ fill holes; โหมด dilation, Gaussian smoothing หรือ topology_fix เป็นตัวเลือกเพิ่มเติม ไม่ได้ทำพร้อมกันทั้งหมด |
| **Shape preprocessing** | ก่อนและระหว่างเตรียม SPHARM | ICP จัดศูนย์ แนวแกน ทิศทาง และสเกลของรูปร่าง; จากนั้น `SPHARM/run_spharm_batch.py` เรียก `improve_label_quality()` เพื่อ fill holes, closing และเก็บ connected component ที่ใหญ่ที่สุด ก่อน SegPostProcess และ spherical parameterization |
| **Feature preprocessing** | ก่อน classifier | สคริปต์โมเดลสาย PLS ใช้ StandardScaler และ PLS เพื่อลดมิติ; Desktop predictor เรียก `.transform()` ด้วย scaler/PLS ที่บันทึกไว้จากการฝึก ส่วน augmentation เป็นการสร้างตัวอย่างฝึกเพิ่มเติม |

ดังนั้น **extraction คือการเลือกบริเวณฮิปโปแคมปัส ส่วน mask preprocessing คือการปรับ mask ที่เลือกมาให้พร้อมวิเคราะห์** แม้ทั้งสองงานจะอยู่ในสคริปต์เดียวกัน การแยกชื่อขั้นตอนนี้ทำให้เห็นชัดว่ามีการปรับข้อมูลตรงไหนบ้าง

| ขั้นตอน | ระบบทำอะไร | ข้อมูลเข้าและผลลัพธ์หลัก | โค้ดที่รับผิดชอบ |
|---|---|---|---|
| 1. รับและจัดการข้อมูล | เลือกไฟล์หรือโฟลเดอร์ MRI และประมวลผลเป็นรายไฟล์หรือ batch; สามารถใช้ segmentation ที่มีอยู่ผ่านโหมดข้าม FastSurfer | MRI `.nii`, `.nii.gz`, `.mgz` หรือ segmentation ที่มีอยู่ | `run_pipeline.py`, `DesktopApp/LeftUI/import_panel.py` |
| 2. เตรียมภาพและแบ่งส่วนสมอง | FastSurfer conform ภาพเมื่อจำเป็นก่อนเรียกโมเดล segmentation เพื่อระบุบริเวณต่าง ๆ ในสมอง | ภาพ conformed `orig.mgz` และ segmentation เช่น `aparc.DKTatlas+aseg.deep.mgz` | `run_pipeline.py`, `FastSurfer/` |
| 3. แยกฮิปโปแคมปัสและเตรียม mask | ดึง label 17 ฝั่งซ้ายและ 53 ฝั่งขวา สร้าง binary mask แล้วทำ mask preprocessing เช่น closing/fill holes; ค่าเริ่มต้นเป็น moderate | mask `.nii.gz` ใน `left_hippocampus` และ `right_hippocampus`; โค้ดตั้งใจบันทึก MRI คู่กันด้วย แต่มีบัคตามข้อ 5 | `extract_hippocampus.py`, `config.py` |
| 4. จัดแนวด้วย ICP | สร้าง surface mesh จัดศูนย์และแนวแกน แก้ความกำกวมของทิศทาง และทำ groupwise rigid ICP ก่อนส่งออก mask ที่จัดแนวแล้ว | aligned meshes, `aligned_nifti`, `T_matrices.npy` และประวัติ convergence | `ICP/ICP.py` |
| 5. แปลงรูปร่างด้วย SPHARM-PDM | ทำความสะอาด mask สร้าง spherical parameterization และแทนรูปร่างด้วย spherical harmonics เพื่อเตรียมการเปรียบเทียบรูปร่าง; มีขั้น resample และ realignment เพิ่มเติม | mesh `.vtk`, สัมประสิทธิ์ `.coef`, grid mesh และ realigned mesh | `SPHARM/run_spharm_batch.py`, `SPHARM/resample_spharm_grid.py`, `SPHARM/realign_spharm.py` |
| 6. เตรียมฟีเจอร์และเพิ่มข้อมูล | แปลง mesh เป็นพิกัด XYZ/ขอบเชื่อมต่อ หรือแปลง `.coef` เป็นตัวเลขใน CSV; สร้างตัวอย่างสังเคราะห์ด้วย interpolation ในพื้นที่ PLS และมีสคริปต์เพิ่มข้อมูลเพื่อปรับสมดุลคลาส | CSV ฟีเจอร์, synthetic meshes/coefficients และ metadata การสร้างข้อมูล | `Data_Processing/` |
| 7. ฝึกและประเมินโมเดล | ทดลอง SVM, MLP, ResNet, ResNet ร่วม autoencoder, MobileNet, SqueezeNet และ PointNet ตามสคริปต์ที่มี; ใช้ชุดข้อมูล ds004469/ds005602 ทั้งแบบภายในชุด รวมชุด และทดสอบข้ามชุด | ผลทำนายที่บันทึกไว้, accuracy, confusion matrix, sensitivity/specificity, F1, ROC/AUC และ bootstrap summaries | `Model/`, `Model_Evaluation_and_Benchmarks/` |
| 8. ทำนายและแสดงผลในแอป | โหลด scaler + PLS + ResNet1D ที่ฝึกไว้ รับฟีเจอร์ `.coef` และคำนวณผลซ้าย/ขวา; เมื่อมีสองข้างใช้ค่าเฉลี่ย probability เป็นผลรวม พร้อมตารางผลและมุมมอง 3D | ผลจำแนกและ probability รายข้าง/รวม, `predictions_summary.csv`, ภาพ mesh และข้อมูลประกอบการแสดงผล | `DesktopApp/models/predictor.py`, `DesktopApp/LeftUI/result_panel.py`, `DesktopApp/RightUI/` |

### ผู้ใช้ใช้งานผ่านอะไรได้บ้าง

- **Desktop App:** เลือกโฟลเดอร์ต้นทาง/ปลายทาง สั่งขั้นตอน FastSurfer, ICP, SPHARM และ prediction ดู log ความคืบหน้า ตารางผล และโมเดลสามมิติ มีแผงควบคุมค่าพารามิเตอร์และประวัติโฟลเดอร์ที่ใช้
- **สคริปต์ประมวลผล:** `run_pipeline.py` ทำ segmentation/extraction; `run_everything.bat` ทำ ICP → SPHARM → realignment และเปิดกราฟ convergence ส่วนฝึกโมเดลเรียกผ่านสคริปต์ใน `Model/` แยกต่างหาก
- **เครื่องมือวิเคราะห์ผล:** `Visualize/` มีกราฟ PLS-DA, ROC, violin และตัวดู mesh; มีสคริปต์สร้าง/จัดระเบียบผล Grad-CAM และแผนที่ระยะห่าง รวมถึงสคริปต์รวบรวมผลเป็น Excel/Word ใน `Model_Results_Excel/04_Scripts/`

### สิ่งที่ต้องแยกระหว่างการฝึกกับการใช้งานจริง

ส่วนฝึกโมเดลใช้ข้อมูลที่มี label และอาจมี augmentation ส่วน Desktop inference ใช้โมเดลที่บันทึกไว้แล้ว โดยไม่ฝึกใหม่ทุกครั้ง โค้ดตัวสกัดฟีเจอร์บางตัวอนุมาน label จากชื่อไฟล์ และ map ฝั่ง contralateral เป็นคลาส 0 สำหรับงานจำแนกแบบ binary ดังนั้นความหมายของคลาสขึ้นกับการจัดข้อมูลด้วย ไม่ได้มีเฉพาะ healthy controls เสมอไป

ภาพ Grad-CAM/PLS บางส่วนที่แอปแสดงมาจากผลที่คำนวณไว้ล่วงหน้าและใช้ประกอบกับ mesh ของผู้ป่วย จึงควรแยกจาก probability ที่คำนวณจากฟีเจอร์ของผู้ป่วยคนนั้นโดยตรง

ภาพรวมนี้อธิบายความสามารถและเส้นทางที่พบในโค้ดปัจจุบัน ส่วนข้อผิดพลาดที่ทำให้บางขั้นไม่ทำงานตามเป้าหมายและขอบเขตการทดสอบจริงระบุไว้ในรายงานด้านล่าง

## ขอบเขตและหลักฐาน

- ตรวจ syntax ด้วย AST ของ Python 140 ไฟล์ รวมสคริปต์ audit ที่เพิ่มใหม่ 1 ไฟล์: ไม่พบ syntax error
- อ่านและเทียบข้อมูล train/test จากเส้นทางที่ SVM และ PointNet ใช้จริง 20 คู่: พบไฟล์ครบ และไม่พบ Subject หรือแถวฟีเจอร์ซ้ำแบบตรงกันระหว่าง train/test ในคู่ที่ตรวจ ข้อนี้ไม่ได้พิสูจน์ว่าไม่มีความเกี่ยวข้องระหว่างข้อมูลสังเคราะห์กับต้นฉบับ
- ทดสอบฟังก์ชันเดิมแบบแยกส่วน โดยใช้ mock แทน Slicer, nibabel และ mesh เฉพาะจุดที่ระบุ ไม่ได้จำลองการทำงานของ MRI/โมเดลทั้งหมด
- ตรวจผลทำนาย SVM ที่บันทึกอยู่แล้ว 12 ชุด
- เครื่องมือทดสอบ: Python 3.9.10 ที่มากับ SlicerSALT พร้อม numpy/pandas/scikit-learn
- ข้อจำกัด: shell นี้ไม่มีคำสั่ง `python`/`py`; interpreter ที่ใช้ทดสอบไม่มี nibabel, torch และ PyQt6 และการ import VTK ล้มเหลวจาก DLL จึงยังไม่ได้ทดสอบ GUI, FastSurfer, resampling จริง หรือโหลดโมเดล inference แบบครบเส้นทาง
- มีการแก้ `ICP/ICP.py` และ `SPHARM/run_spharm_batch.py` อยู่ก่อนแล้ว การตรวจอ้างอิงไฟล์ปัจจุบันและรักษาการแก้เดิมไว้

## 1. P1 — ICP ไม่ใช้ reference template และเปลี่ยนสเกลตามสมาชิกใน batch

**ตำแหน่ง:** `DesktopApp/LeftUI/icp_panel.py:114`, `ICP/ICP.py:517`, `ICP/ICP.py:632`, `ICP/ICP.py:753`

แอปส่ง `--reference_template` แต่ parser ของ ICP ไม่มี argument นี้ และ `parse_known_args()` ทิ้งค่าที่ไม่รู้จัก จากนั้น ICP สร้าง mean จาก batch ปัจจุบัน และคำนวณ `global_scale` จาก bounding box รวมของ batch

**พิสูจน์:** ส่ง `--reference_template official.vtk` เข้า parser เดิม ไม่เกิด error แต่ไม่มีค่า reference_template ในผลลัพธ์ ทดสอบฟังก์ชัน normalization เดิมด้วย mesh stub รูปลูกบาศก์เดียวกัน: รันเดี่ยวได้ขอบเขต −1 ถึง 1; เพิ่มเพื่อนที่ใหญ่เป็นสองเท่าแล้วขอบเขตของตัวเดิมกลายเป็น −0.5 ถึง 0.5

**ผลกระทบ:** ฟีเจอร์ผู้ป่วยเดิมอาจเปลี่ยนเมื่อเปลี่ยนผู้ร่วม batch โดยเฉพาะการใช้โมเดลที่ฝึกด้วย cohort ใหญ่กับผู้ป่วยรายเดียว ขนาดสัมพัทธ์ภายใน batch ยังรักษาไว้ แต่หน่วยสเกลข้าม batch ไม่คงที่

**แนวแก้:** รองรับ fixed training template จริง เก็บ transform/หน่วยสเกลที่ใช้ตอนฝึกและใช้ซ้ำตอน inference ห้ามคำนวณหน่วยสเกลใหม่จากผู้ป่วยที่เข้ามาทำนาย

## 2. P1 — งานล้มเหลวแต่ขั้นตอนถัดไปได้รับสถานะสำเร็จ

**ตำแหน่ง:** `run_pipeline.py:406`, `run_pipeline.py:434`, `SPHARM/run_spharm_batch.py:370`, `SPHARM/run_spharm_batch.py:382`, `SPHARM/run_spharm_batch.py:417`, `DesktopApp/LeftUI/fastsurfer_panel.py:74`, `DesktopApp/LeftUI/spharm_panel.py:185`

- MRI pipeline เพิ่มตัวนับ failed แต่จบ `main()` ตามปกติ แม้ทุก segmentation ล้มเหลว; GUI ใช้ return code 0 เป็น success
- `process_single_subject()` จับ exception แล้วไหลไป `return True`
- ตัว batch ของ SPHARM ไม่ใช้ return value ของแต่ละ subject และไม่รวมจำนวนสำเร็จ/ล้มเหลว
- GUI รอ realignment จบแล้วพิมพ์ SUCCESS โดยไม่ตรวจ `re_proc.returncode`
- `run_everything.bat` ตรวจการมีโฟลเดอร์เป็นหลัก ทั้งที่โฟลเดอร์อาจถูกสร้างก่อนงานสำเร็จ หรือค้างจากรอบเดิม

**พิสูจน์:** mock FastSurfer ให้ล้มเหลว 1/1 งานแล้ว `main()` ไม่ raise และพิมพ์ PIPELINE COMPLETE; inject exception ตอนโหลด volume แล้ว SPHARM คืน True; mock ทุก subject ให้คืน False แล้ว batch ยังพิมพ์ ALL BATCH PROCESSING COMPLETED

**แนวแก้:** ส่งต่อสถานะผิดพลาดและ exit code ให้ครบทุกชั้น ตรวจผลลัพธ์ราย subject ที่สร้างในรอบปัจจุบัน และแสดงสถานะ partial success เมื่อสำเร็จไม่ครบ

## 3. P1 — Cross-validation แบ่งข้อมูลหลัง augmentation ทำให้ validation ไม่เป็นอิสระ

**ตำแหน่ง:** `Data_Processing/augment_plsda_balanced.py:400`, `Model/All_Augment_tain/left/SVM/train_svm_pls.py:37`, `Model/All_Augment_tain/left/SVM/train_svm_pls.py:70`, `Model/Ds005602/left/ResNet/train_resnet_pls.py:198`

ตัว augmentation fit PLS ด้วยข้อมูลทั้งหมดที่รับเข้ามา ขณะที่ training scripts อ่าน CSV ที่ augment มาแล้วและใช้ StratifiedKFold แบ่งตามแถว การ fit scaler/PLS ใหม่ใน fold ของ classifier จึงไม่ได้ย้อนแก้ความสัมพันธ์และการเรียนรู้ที่เกิดใน augmentation ก่อนหน้า

**พิสูจน์จากข้อมูลจริง:** All_Augment_tain/left มี 1,480 แถว โดย 1,250 แถวมีชื่อ synthetic/augmented เมื่อแบ่งด้วย CV แบบเดียวกับ SVM พบ synthetic ใน validation แต่ละ fold 250, 244, 243, 257, 256 แถว

**ผลกระทบ:** คะแนน CV ไม่ใช่การวัดบนผู้ป่วยจริงที่กันไว้อย่างอิสระ และอาจสูงเกินจริง ไม่มี parent metadata ในชุดไฟล์ที่ค้นพบ จึงยังระบุคู่ parent/child ที่ข้าม fold จริงแต่ละคู่ไม่ได้ และไม่ได้สรุปว่า external test ปนเปื้อนด้วย

**แนวแก้:** แบ่ง subject จริงก่อนทุกอย่าง; fit preprocessing/PLS และสร้าง augmentation เฉพาะ train fold; validation/test ใช้ข้อมูลจริง; เก็บ Parent_A/Parent_B และ dataset namespace เพื่อสอบย้อนกลับ

## 4. P1 — ข้อมูลที่ไม่รู้ label ถูกเขียนเป็น Healthy

**ตำแหน่ง:** `Data_Processing/extract_ml_features_coef.py:26`, `Data_Processing/extract_ml_features_coef.py:148`

`classify_subject()` คืน Class = −1 สำหรับชื่อที่ไม่มีคำระบุโรค แต่ผู้เขียน CSV ใช้ `1 if group_label == 1 else 0` จึงเปลี่ยน Unknown เป็น BinaryClass = 0

**พิสูจน์:** ชื่อสมมติ `lh_sub-001_hippocampus_aligned` ได้ Group = Unknown, Class = −1 แต่ถูกแปลงเป็น BinaryClass = 0 ตามโค้ดผู้เขียน CSV

**ผลกระทบ:** นำชื่อแบบ BIDS ที่ไม่ได้ฝัง diagnosis มาใช้แล้วอาจสร้าง label ฝึกผิดอย่างเงียบ ๆ

**แนวแก้:** อ่าน label จาก metadata ที่จับคู่ด้วย subject/dataset ID และปฏิเสธหรือแยกแถว Unknown ออกจาก supervised training

## 5. P2 — MRI คู่กับ mask ไม่ถูกบันทึกเพราะไม่ได้ import numpy

**ตำแหน่ง:** `run_pipeline.py:295`

`organize_output()` เรียก `np.asarray` แต่ไฟล์ไม่ได้ import numpy เป็น np จากนั้นจับ exception เป็น WARNING และยังคืน success หาก copy mask ครบ

**พิสูจน์:** เรียกฟังก์ชันเดิมพร้อม mock nibabel ได้ `name 'np' is not defined`, ไม่มี MRI ถูกสร้าง และคืน True

**แนวแก้:** เพิ่ม import ที่ขาดและกำหนดว่า paired MRI เป็นผลลัพธ์บังคับหรือไม่ให้ชัดเจน พร้อมตรวจผลบันทึก ไม่ควรนับสำเร็จเกินสิ่งที่สร้างได้จริง

## 6. P2 — Predictor สลับความหมายของฟีเจอร์เมื่อ DataFrame เรียงคอลัมน์ต่างกัน

**ตำแหน่ง:** `DesktopApp/models/predictor.py:264`, `DesktopApp/models/predictor.py:272`, `DesktopApp/models/predictor.py:294`

โค้ดเปลี่ยน DataFrame เป็น `.values` ก่อน แล้วนำชื่อฟีเจอร์ที่ scaler คาดหวังไปครอบค่าตามลำดับเดิม แทนที่จะเรียงค่าตามชื่อคอลัมน์

**พิสูจน์:** input เรียง `[Coef_2=20, Coef_1=10]`; scaler ต้องการ `[Coef_1, Coef_2]`; ผลเตรียมข้อมูลกลับเป็น `Coef_1=20, Coef_2=10`

**ผลกระทบ:** เปลี่ยนลำดับคอลัมน์ CSV/DataFrame แล้วผลทำนายเปลี่ยนได้โดยไม่เกิด error เส้นทางอ่าน `.coef` โดยตรงไม่ใช่ตัวกระตุ้นกรณีนี้

**แนวแก้:** ตรวจชื่อที่ขาด/เกินและ reorder DataFrame ด้วย expected feature names ก่อนแปลงเป็น array

## 7. P2 — Bootstrap AUC จัดการคะแนนที่เท่ากันผิด

**ตำแหน่ง:** `Model/run_bootstrap_on_all_trained_models.py:63`

`argsort(argsort(ypr))` แจก rank ไม่เท่ากันให้คะแนนที่เท่ากัน ทำให้ AUC ขึ้นกับลำดับแถว แทนที่จะใช้ average ranks

**พิสูจน์:** y = [0,0,1,1], ทุก probability = 0.5; AUC ที่ถูกต้องคือ 0.5 สำหรับ resample ที่มีทั้งสองคลาส แต่ฟังก์ชันเดิมรายงาน mean = 0.5097 และ 95% CI = [0.0000,1.0000] เมื่อ seed = 42, 1,000 รอบ

**แนวแก้:** ใช้ `roc_auc_score` ในแต่ละรอบ หรือ average rank ที่รองรับ ties และกำหนดวิธีจัดการรอบที่มีคลาสเดียวให้ชัดเจน

## 8. P2 — Bootstrap ของ SVM เปลี่ยนกฎตัดสินจากผลทำนายที่ประเมินจริง

**ตำแหน่ง:** `Model/Ds004469/left/SVM/train_svm_pls.py:138`, `Model/Ds004469/left/SVM/train_svm_pls.py:190`, `Model/run_bootstrap_on_all_trained_models.py:16`

คะแนนหลักใช้ `svm.predict()` แต่ bootstrap สร้าง prediction ใหม่ด้วย `predict_proba > 0.5` ซึ่งไม่จำเป็นต้องตรงกันสำหรับ SVC

**หลักฐานในผลบันทึกจริง:** Ds004469/left มีคำตัดสินเปลี่ยน 1 จาก 16 ราย; Ds004469Train_Ds005602test/right เปลี่ยน 1 จาก 284 ราย พบ 2 ชุดที่ต่างจาก 12 ชุด SVM ที่ตรวจ

**แนวแก้:** bootstrap y_true, y_pred และ y_prob ด้วย index เดียวกัน ใช้ y_pred เดิมสำหรับ accuracy/confusion matrix และใช้ y_prob สำหรับ ROC/AUC หรือกำหนด threshold policy เดียวกันทั้งระบบแล้วประเมินใหม่

## 9. P2 — ตารางเปรียบเทียบโมเดลใช้ชุดทดสอบคนละชุด

**ตำแหน่ง:** `Model/Ds005602/left/SVM/train_svm_pls.py:38`, `Model/Ds005602/left/PointNet/train_pointnet.py:141` และสคริปต์เทียบเท่าของ dataset/side อื่น

อ่านจำนวนแถวจาก CSV ตาม path ที่โค้ดใช้จริง:

| Dataset/Side | SVM test | PointNet test |
|---|---:|---:|
| Ds005602/left | 84 | 56 |
| Ds005602/right | 86 | 57 |
| Ds004469/left | 16 | 11 |
| Ds004469/right | 17 | 11 |
| All_Augment_tain/left | 99 | 67 |
| All_Augment_tain/right | 102 | 68 |

**ผลกระทบ:** คะแนนที่รวมภายใต้ dataset เดียวกันไม่ได้วัดบน cohort เดียวกัน จึงใช้จัดอันดับความสามารถของโมเดลโดยตรงไม่ได้

**แนวแก้:** ใช้ split manifest กลางและสร้างทั้ง XYZ/coef ตาม Subject ID เดียวกัน ตรวจความเท่ากันของ ID และ label ก่อนรวม benchmark พร้อมเก็บ ID ใน prediction artifacts

## 10. P2 — พารามิเตอร์ realignment ที่แอปส่งไม่ตรงกับ CLI

**ตำแหน่ง:** `DesktopApp/LeftUI/spharm_panel.py:165`, `DesktopApp/LeftUI/spharm_panel.py:172`, `SPHARM/realign_spharm.py:172`

แอปส่ง `--target_template`, `--tolerance`, `--max_iterations` แต่สคริปต์รับเพียง `--spharm_dir` กับ `--template` และทิ้งค่าที่ไม่รู้จักด้วย parse_known_args

**ผลกระทบ:** ตัวเลือกที่ส่งจาก UI ไม่มีผลตามที่ระบุ ปัจจุบัน auto-detect อาจพบ official template เดียวกันและบดบังบัคนี้ แต่การระบุ template ผ่าน flag ดังกล่าวไม่ได้ถูกใช้งานจริง

**แนวแก้:** ทำ CLI/UI contract ให้ตรงกัน รองรับพารามิเตอร์ที่มีผลจริง และแสดง error สำหรับ argument ที่ไม่รองรับ

## 11. P2 — ติดตั้งตาม root requirements แล้วยังเปิดระบบหลักไม่ได้ครบ

**ตำแหน่ง:** `requirements.txt:1`, `DesktopApp/models/predictor.py:4`, `DesktopApp/models/predictor.py:7`, `Model/Ds005602/left/SVM/train_svm_pls.py:4`

Root requirements ไม่มี torch, scikit-learn, joblib และ seaborn แม้ Desktop predictor และ training scripts import โดยตรง สคริปต์ training บางตัวใช้ `uv --with` ช่วยได้เฉพาะเส้นทางนั้น แต่ไม่ครอบคลุม `python DesktopApp/main.py` หลังติดตั้งตาม README/root setup

**แนวแก้:** แยก requirements สำหรับ desktop, training, FastSurfer และ Slicer อย่างชัดเจน พร้อม smoke import ของ entry point แต่ละตัวหลังติดตั้ง

## ลำดับดำเนินการที่แนะนำ

1. แก้สถานะ failure, unknown labels และ paired MRI เพื่อให้ระบบไม่รายงานผลครบทั้งที่ข้อมูลไม่ครบ
2. ทำ template/scale และ schema ของฟีเจอร์ให้คงที่ระหว่าง training/inference
3. สร้าง split กลางและย้าย augmentation เข้า train fold แล้วฝึก/ประเมินใหม่ใน output ใหม่
4. แก้ bootstrap และสร้างรายงานเปรียบเทียบใหม่บน test cohort เดียวกัน
5. ตรวจ GUI และ MRI → prediction จริงหลังเตรียม runtime ที่มี dependencies ครบ

## ทดสอบซ้ำ

จากโฟลเดอร์โปรเจกต์:

```powershell
& 'C:\Program Files\SlicerSALT 6.0.0\bin\PythonSlicer.exe' audit_checks.py
```

สคริปต์อ่านโค้ดและ CSV/NPZ เดิม สร้าง mock fixtures ใน temporary directory และเขียนผลลง `audit_results.json` เท่านั้น ไม่มีการฝึกโมเดล เปิด GUI หรือประมวลผล MRI เดิมซ้ำ ไฟล์ `audit_console.txt` เป็น console output จากรอบตรวจล่าสุด

การตรวจนี้เป็นการตรวจข้อผิดพลาดของซอฟต์แวร์และกระบวนการประเมิน ไม่ใช่การรับรองประสิทธิภาพโมเดลจากการทดลองครบทุกขั้นตอน
