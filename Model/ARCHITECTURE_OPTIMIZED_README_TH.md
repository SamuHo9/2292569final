# Fixed-topology และ Architecture-optimized

โปรเจกต์นี้แยกการทดลองเป็นสองชุดเพื่อให้เปรียบเทียบได้อย่างยุติธรรม

## Fixed-topology baseline

`optuna_leakage_free_all.py` ใช้โครงสร้าง MLP/ResNet/ResNetAE/MobileNet/
SqueezeNet เดิม และ SVM แบบ RBF โดยค้นเฉพาะ hyperparameter การฝึก ค่า PLS-DA
และค่า SVM ตัวอย่างคำสั่ง:

```bash
python Model/optuna_leakage_free_all.py \
  --protocol coef_plsda --cohort all --side all --model all \
  --folds 10 --n_trials 10 --children_per_pair 8 \
  --pls_components 8 --seed 42 --eval_seeds 42,123,2026 \
  --data_root Model/current_spharm_80_20_20260922 \
  --output_root Model/optuna_current_spharm_80_20_20260922
```

## Architecture-optimized

`optuna_architecture_optimized.py` ใช้ protocol และ split เดียวกัน แต่ให้
Optuna ค้นจำนวนชั้น ความกว้างของชั้น จำนวน residual block, kernel, pooling,
width multiplier และ bottleneck ภายใน search space ที่กำหนดไว้ล่วงหน้า

```bash
python Model/optuna_architecture_optimized.py \
  --protocol coef_plsda --cohort all --side all --model all \
  --folds 10 --n_trials 10 --tune_epochs 20 --tune_patience 5 \
  --final_epochs 40 --children_per_pair 8 \
  --seed 42 --eval_seeds 42,123,2026 --device cuda \
  --data_root Model/current_spharm_80_20_20260922 \
  --output_root Model/architecture_optuna_current_spharm_80_20_20260922
```

ทั้งสอง runner ทำ PLS-DA และ augmentation ภายใน training fold เท่านั้น ไม่ทำ
augmentation กับ validation หรือ test และไม่ใช้ test เลือก trial, architecture
หรือ threshold

หลัง baseline เสร็จ สามารถปรับ threshold จาก OOF ได้ด้วย:

```bash
python Model/tune_fixed_baseline_thresholds.py \
  --run_root Model/optuna_current_spharm_80_20_20260922 \
  --data_root Model/current_spharm_80_20_20260922 \
  --protocol coef_plsda --folds 10 --children_per_pair 8 \
  --pls_components 8 --device cuda
```

การเลือกโมเดลใช้ OOF balanced accuracy เป็นหลัก ส่วน test ใช้รายงานผลครั้ง
สุดท้ายเท่านั้น ควรรายงานค่า threshold 0.5 และค่า threshold ที่เลือกจาก OOF
แยกกัน

ข้อจำกัดของชุดปัจจุบัน: ICP reference เป็น legacy reference ที่สร้างจากข้อมูล
รวม จึงควรสร้าง train-only reference และรัน ICP/SPHARM ใหม่ก่อนสรุปผล held-out
สำหรับงานวิจัย
