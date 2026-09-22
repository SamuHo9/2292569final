# รายงานการรัน PointNet ชุดปัจจุบัน

## ข้อมูลนำเข้า

สร้าง XYZ จากไฟล์ `_SPHARM_ellalign.vtk` ในผลรัน ICP/SPHARM เดิม โดยใช้รายชื่อ, label และ train/test assignment จาก `current_spharm_80_20_20260922` เป็นแหล่งอ้างอิงเดียวกัน ไม่ใช้ไฟล์ XYZ เก่า 264/67 แถว และไม่ใช้ข้อมูลที่ทำ PLS-DA ล่วงหน้า

ชุด XYZ อยู่ที่:

`Model/current_spharm_80_20_20260922/Output_Dataset`

จำนวนแถวตรงกับ split coefficient ปัจจุบันทุก cohort/side และตรวจผ่าน `data_contract` แล้ว

## การรัน

รันแยกสอง protocol:

- `pointnet_raw`: PointNet รับ raw XYZ ที่ normalize แยกต่อ mesh และใช้ balanced jitter augmentation ใน training fold
- `pointnet_plsda`: PointNet รับ XYZ เหมือนเดิม แต่สร้าง augmentation จาก PLS-DA ภายใน training fold เท่านั้น

ค่าที่ใช้เหมือนกันทั้งสอง protocol:

- 10-fold grouped CV ระดับผู้ป่วย
- Optuna 10 trials ต่อ cohort/side
- final seeds `42, 123, 2026`
- `children_per_pair=8`
- `tune_epochs=20`, `final_epochs=40`
- test ไม่ถูกใช้เลือก trial, hyperparameter หรือ augmentation

## ผลรวมจาก audit

| Protocol | Studies | Final seed runs | Test BA เฉลี่ย | Test accuracy เฉลี่ย |
|---|---:|---:|---:|---:|
| PointNet raw XYZ | 6 | 18 | 0.6198 | 0.6959 |
| PointNet PLS-DA | 6 | 18 | 0.6247 | 0.7264 |

ค่าในตารางเป็นค่าเฉลี่ยจากสาม seed และใช้เพื่อรายงานหลังการเลือก trial จาก OOF เท่านั้น

## ไฟล์ผลลัพธ์

- Raw XYZ: `Model/pointnet_optuna_current_spharm_80_20_20260922`
- PLS-DA PointNet: `Model/pointnet_plsda_optuna_current_spharm_80_20_20260922`
- ตารางเปรียบเทียบ: `Model/POINTNET_COMPARISON_RAW_VS_PLSDA.csv`
- Audit raw: `Model/pointnet_optuna_current_spharm_80_20_20260922/POINTNET_OPTUNA_AUDIT.json`
- Audit PLS-DA: `Model/pointnet_plsda_optuna_current_spharm_80_20_20260922/POINTNET_OPTUNA_AUDIT.json`

แต่ละ seed มี `model_state.pt`, `metrics.json`, `test_predictions.csv` และ `run_manifest.json` ใต้ `final/pointnet_raw/...` หรือ `final/pointnet_plsda/...`

## ข้อควรระบุในงานวิจัย

PLS-DA ใน protocol นี้ใช้สร้าง augmentation ใน training fold เท่านั้น และ PointNet ยังคงรับ XYZ ไม่ใช่ค่า PLS score เป็น input การประเมินยังอิง legacy ICP reference เดิม จึงควรระบุข้อจำกัดนี้เมื่อรายงานผล held-out
