# ค่าการทำ PLS-DA และจำนวนข้อมูลสังเคราะห์

ค่าด้านล่างคำนวณจาก current split, grouped 10-fold และ seed `42` ซึ่งเป็น protocol เดียวกับรอบ training ล่าสุด

| Dataset/Side | Train/Test | Train class 0/1 | PLS components จริง | Synthetic ต่อ fold | รวม synthetic ใน 10 folds | Synthetic ตอน fit final |
|---|---:|---:|---:|---:|---:|---:|
| All_coef / left | 295 / 74 | 191 / 104 | 8 | 77–79 | 783 | 87 |
| All_coef / right | 302 / 75 | 220 / 82 | 8 | 124–125 | 1,242 | 138 |
| All_coef_Ds004469 / left | 54 / 12 | 37 / 17 | 8 | 17–19 | 180 | 20 |
| All_coef_Ds004469 / right | 53 / 12 | 42 / 11 | 8 | 27–29 | 279 | 31 |
| All_coef_Ds005602 / left | 241 / 62 | 154 / 87 | 8 | 59–61 | 603 | 67 |
| All_coef_Ds005602 / right | 249 / 63 | 178 / 71 | 8 | 96–98 | 963 | 107 |

ค่าจำนวน synthetic ต่อ fold ของแต่ละ fold ตามลำดับคือ:

- `All_coef/left`: `77, 79, 79, 79, 79, 78, 78, 78, 78, 78`
- `All_coef/right`: `125, 125, 124, 124, 124, 124, 124, 124, 124, 124`
- `All_coef_Ds004469/left`: `18, 18, 18, 18, 17, 17, 17, 19, 19, 19`
- `All_coef_Ds004469/right`: `27, 27, 29, 28, 28, 28, 28, 28, 28, 28`
- `All_coef_Ds005602/left`: `60, 59, 59, 59, 61, 61, 61, 61, 61, 61`
- `All_coef_Ds005602/right`: `98, 96, 96, 96, 96, 96, 96, 96, 96, 97`

จำนวน synthetic ในตารางเป็นจำนวนที่สร้างเพื่อทำให้ class 0 และ class 1 สมดุลในแต่ละ training fold สำหรับทั้ง `coef_plsda` และ PointNet PLS-DA โดยจำนวนแถวเท่ากัน แต่ค่าคุณลักษณะสังเคราะห์แตกต่างกันเพราะใช้ feature space คนละแบบ

- coefficient PLS-DA: สร้างกลับมาเป็น 507 coefficient features
- PointNet PLS-DA: สร้างกลับมาเป็น normalized XYZ ขนาด `3 × 1002`

ค่า loading/score ของ PLS-DA ไม่ใช่ตัวเลขเดียว แต่เป็นพารามิเตอร์ของ PLS model ที่ fit แยกต่อ dataset, side และ fold ไฟล์ CSV รายละเอียดอยู่ที่ `PLSDA_AUGMENTATION_COUNTS.csv`

## ตำแหน่งค่าที่ใช้เลือก n_components

ไม่มีค่า n_components เดียวที่ดีที่สุดสำหรับทุก dataset และทุก model การเลือก architecture-optimized ใช้ OOF balanced accuracy จาก training เท่านั้น โดยค่า n_components ของ best trial อยู่ที่:

- `Model/architecture_optuna_current_spharm_80_20_20260922/ARCH_OPT_TUNING_SUMMARY.csv` ในคอลัมน์ `BestParams` และ key `pls_components`
- `Model/architecture_optuna_current_spharm_80_20_20260922/ARCH_OPT_WINNERS_BY_COHORT_SIDE.csv` สำหรับโมเดลที่ชนะในแต่ละ cohort/side
- `studies/coef_plsda/<cohort>/<side>/<model>/best_params.json` สำหรับ best trial ราย study
- `studies/coef_plsda/<cohort>/<side>/<model>/trials.csv` สำหรับค่า n_components ของทุก Optuna trial

สำหรับ fixed baseline และ PointNet PLS-DA กำหนด n_components เป็น 8 ในคำสั่งและ manifest ไม่ได้ค้นหา n_components แยกด้วย Optuna

## ค่าอะไรถูกบันทึกและอะไรยังไม่ได้บันทึก

มีการบันทึกแล้ว:

- จำนวน PLS components ใน `best_params.json`, `threshold.json` และ `run_manifest.json`
- จำนวน synthetic ต่อ fold ใน `threshold.json` และ `PLSDA_AUGMENTATION_COUNTS.csv`
- จำนวน train ก่อน/หลัง augmentation และจำนวนของแต่ละ class ใน `threshold.json` และ `run_manifest.json`
- OOF prediction ใน `best_oof_predictions.csv`
- test prediction และ metric ใน `test_predictions.csv` และ `metrics.json`
- Optuna objective และ hyperparameter ใน `trials.csv`

ยังไม่ได้บันทึกเป็นไฟล์ในรอบเดิม:

- scaler mean และ standard deviation ที่ใช้ก่อน fit PLS-DA
- PLS x_weights, x_loadings, y_loadings และ score matrix ของทุก fold
- รายชื่อ parent pairs ที่ถูกเลือกในแต่ละ class
- ค่า feature จริงของ synthetic rows แต่ละแถว

ค่ากลุ่มหลังถูกสร้างในหน่วยความจำภายใน `leakage_free_plsda_training.py` และ `leakage_free_pointnet_plsda_training.py` แล้วถูกใช้ train model ต่อทันที จึงไม่สามารถอ่านค่ารายตัวกลับจาก `metrics.json` หรือ `test_predictions.csv` ได้

ไฟล์ `scaler.pkl` ที่อยู่ใน final model เป็น scaler ของ classifier หลัง augmentation ไม่ใช่ scaler ที่ใช้ภายใน PLS-DA ส่วน SVM เก็บ classifier scaler ไว้ภายใน `model.pkl` เช่นเดียวกัน

หากต้องการรายงานค่า loading/score หรือ synthetic feature รายตัว ต้องเพิ่ม export ในฟังก์ชัน PLS-DA แล้ว rerun ด้วย protocol และ seed เดิม จึงจะได้ artifact ที่ตรวจสอบย้อนกลับได้ครบทุก fold
