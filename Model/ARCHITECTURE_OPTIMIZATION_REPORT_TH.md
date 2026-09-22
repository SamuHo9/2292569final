# รายงานการปรับปรุงและรันโมเดล

## ขอบเขตการรันรอบนี้

รอบนี้เปรียบเทียบสองแนวทางบนข้อมูลชุดเดียวกันที่ `Model/current_spharm_80_20_20260922`:

1. **Fixed-topology baseline** ใช้โครงสร้างโมเดลเดิม แล้วปรับเฉพาะ hyperparameter ของการฝึก/ตัวจำแนก
2. **Architecture-optimized** ใช้ Optuna ค้นหาโครงสร้างภายใน search space ที่กำหนด โดยยังใช้ protocol, split และกติกาป้องกัน leakage เดียวกับ baseline

ทั้งสองชุดเป็น `coef_plsda` และแยกผลตาม cohort กับ side (`left`/`right`) ไม่รวมซ้ายขวาเข้าด้วยกัน

## สิ่งที่แก้ไข

- เพิ่ม `optuna_architecture_optimized.py` สำหรับค้นหา architecture ของ MLP, ResNet, ResNetAE, MobileNet, SqueezeNet และ hyperparameter ของ SVM
- เพิ่ม `tune_fixed_baseline_thresholds.py` เพื่อเลือก threshold จาก training OOF ของ baseline โดยไม่ใช้ test set
- เพิ่ม `audit_architecture_outputs.py` สำหรับตรวจจำนวน study, seed, trial, test isolation, threshold source และความครบถ้วนของผลลัพธ์
- เพิ่มตารางเปรียบเทียบ `MODEL_COMPARISON_FIXED_VS_ARCH.csv` และตารางผู้ชนะตาม OOF `MODEL_WINNERS_FIXED_VS_ARCH.csv`
- เพิ่มคู่มือการรันใน `ARCHITECTURE_OPTIMIZED_README_TH.md`

## ค่าที่ใช้เหมือนกัน

- 10-fold grouped cross-validation ระดับผู้ป่วย
- Optuna 10 trials ต่อโมเดล
- seed ประเมินซ้ำ `42, 123, 2026`
- `children_per_pair=8`
- PLS-DA และ augmentation ทำเฉพาะ training fold
- threshold ที่ปรับแล้วมาจาก training OOF เท่านั้น
- test ใช้ครั้งสุดท้ายเพื่อรายงานผล ไม่ใช้เลือก trial, architecture หรือ threshold

## ผลการรัน

- 36 studies: 3 cohorts × 2 sides × 6 models
- 108 final seed runs: 36 studies × 3 seeds
- failures: 0
- test balanced accuracy เฉลี่ยของ architecture-optimized หลังปรับ threshold: `0.6099426`
- test accuracy เฉลี่ยของ architecture-optimized หลังปรับ threshold: `0.7121704`

ผลราย cohort/side และผลของแต่ละโมเดลอยู่ในไฟล์ CSV ที่ระบุด้านบน ส่วนรายละเอียดทุก study, OOF prediction, test prediction และโมเดลที่บันทึกไว้ อยู่ที่:

`Model/architecture_optuna_current_spharm_80_20_20260922`

## หลักการเลือกโมเดล

ตารางผู้ชนะเลือกจาก **OOF tuned balanced accuracy** เพื่อไม่ให้ test set กลายเป็นเครื่องมือเลือกโมเดล ค่า test ในตารางเป็นผลประเมินภายหลังและใช้รายงานความสามารถในการ generalize เท่านั้น

## ข้อจำกัดที่ต้องระบุในรายงานวิจัย

ICP reference ที่ใช้ในข้อมูลชุดนี้ยังเป็น legacy reference เดิม จึงไม่ได้แก้หรือสร้างใหม่ในรอบนี้ การอ้างผลเป็น held-out ที่เข้มงวดที่สุดควรสร้าง ICP reference จาก training fold เท่านั้นแล้วทำ ICP/SPHARM ใหม่ก่อนประเมิน test

รอบนี้ใช้ข้อมูล coefficient เป็นหลัก จึงยังไม่ได้ rerun raw XYZ PointNet ในชุดคำสั่งนี้ เพราะต้องมี XYZ export และ split ที่ตรงกับชุดปัจจุบันก่อน การไม่รัน PointNet รอบนี้ไม่ได้เปลี่ยนผลของหกโมเดล coefficient ข้างต้น
