# All_coef pairwise augmentation (80/20, no validation)

สร้างเมื่อ: 2026-09-24T12:51:29+07:00

ชุดนี้ใช้เฉพาะ `All_coef` ซ้าย/ขวา จาก split 80/20 ที่มีอยู่แล้ว ไม่สร้าง validation และ augment เฉพาะ train เท่านั้น
test ถูกคัดลอก byte ต่อ byte จาก input และไม่ถูกใช้คำนวณ scaler, จับคู่ หรือสร้างข้อมูลสังเคราะห์

## โปรโตคอล

- `between_class`: จับคู่ class 0 กับ class 1 ที่ใกล้กันที่สุดแบบหนึ่งต่อหนึ่งใน coefficient space หลัง standardize ด้วย train เท่านั้น; คู่หนึ่งสร้างได้ไม่เกิน 8 แถว: alpha 0.10, 0.20, 0.30, 0.40 ติดป้าย class 0 และ alpha 0.60, 0.70, 0.80, 0.90 ติดป้าย class 1 ซึ่งเป็นคลาสของ endpoint ที่ใกล้กว่า
- `within_class`: จับคู่เฉพาะแถวใน class เดียวกันแบบหนึ่งต่อหนึ่ง; คู่หนึ่งสร้างได้ไม่เกิน 8 แถวและรับ label ของคลาสนั้น
- ไม่ใช้ PLS-DA และไม่ใช้คู่เดิมซ้ำ; จำนวนแถวถูกเลือกให้ได้ขนาดสองคลาสเท่ากันสูงสุดภายใต้ข้อจำกัดดังกล่าว

## โฟลเดอร์ที่ใช้เทรนและเทส

| protocol | side | train (ใช้เทรน) | test (ใช้ประเมินครั้งสุดท้าย) |
|---|---|---|---|
| `between_class` | `left` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\between_class\left\train\All_coef_Left_train_coef_features.csv` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\between_class\left\test\All_coef_Left_test_coef_features.csv` |
| `between_class` | `right` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\between_class\right\train\All_coef_Right_train_coef_features.csv` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\between_class\right\test\All_coef_Right_test_coef_features.csv` |
| `within_class` | `left` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\within_class\left\train\All_coef_Left_train_coef_features.csv` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\within_class\left\test\All_coef_Left_test_coef_features.csv` |
| `within_class` | `right` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\within_class\right\train\All_coef_Right_train_coef_features.csv` | `C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924\within_class\right\test\All_coef_Right_test_coef_features.csv` |

## จำนวนที่ได้

| protocol | side | original train (0/1) | synthetic (0/1) | final train (0/1) | pairs |
|---|---|---:|---:|---:|---:|
| `between_class` | `left` | 191/104 | 329/416 | 520/520 | 104 |
| `between_class` | `right` | 220/82 | 190/328 | 410/410 | 82 |
| `within_class` | `left` | 191/104 | 329/416 | 520/520 | 94 |
| `within_class` | `right` | 220/82 | 190/328 | 410/410 | 65 |

ไฟล์ตรวจสอบในแต่ละ side: `pair_manifest.csv` (หนึ่งแถวต่อคู่และไม่ซ้ำ), `synthetic_manifest.csv` (หนึ่งแถวต่อลูก), `scaler_stats.csv` และ `augmentation_manifest.json` ที่อยู่ข้าง CSV train/test

คำสั่ง rerun:
```powershell
$env:PYTHONPATH="$PWD\Model\_training_cuda_site;$PWD\Model\_training_site"
& 'C:\Program Files\SlicerSALT 6.0.0\bin\PythonSlicer.exe' Data_Processing\augment_all_pairwise_balanced.py --data-root "C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\current_spharm_80_20_20260922" --output-root "C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\augment_all_pairwise_80_20_20260924" --protocol all --side all --children-per-pair 8 --seed 42
```

คำเตือนสำหรับการใช้ในงานวิจัย: ชุด train นี้เป็น pre-augmented fixed-split สำหรับการทดลองแบบ 80/20 ตามคำขอ หากทำ cross-validation ต้องย้ายการจับคู่เข้าไปภายในแต่ละ training fold เพื่อไม่ให้ข้อมูลสังเคราะห์รั่วข้าม fold
