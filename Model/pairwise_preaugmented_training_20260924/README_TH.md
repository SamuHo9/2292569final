# รายงานการเทรน pairwise pre-augmented

รันด้วย train/test fixed split ของ `All_coef` ที่สร้างจาก pairwise augmentation ก่อนหน้านี้ โดยไม่สร้าง validation, ไม่ augment ซ้ำระหว่างเทรน และไม่ใช้ PLS-DA

พารามิเตอร์ร่วม:

- Models: SVM, MLP, ResNet, ResNetAE, MobileNet, SqueezeNet
- Seeds: `42, 123, 2026`
- Neural optimizer: AdamW, learning rate `0.001`, weight decay `0.001`, batch size `32`, dropout `0.25`, `20` epochs
- SVM: RBF, `C=1.0`, `gamma=scale`, ไม่มี probability calibration ภายใน เพราะไม่ต้องการ validation เพิ่ม
- Scaler ของ neural models fit จาก train ที่ augment แล้วเท่านั้น; test ใช้แค่ตอนประเมินครั้งสุดท้าย

## ผลเฉลี่ยข้าม 3 seeds (balanced accuracy)

| protocol | side | model | mean | SD | min | max |
|---|---|---|---:|---:|---:|---:|
| `between_class` | `left` | `SqueezeNet` | 0.5403 | 0.0559 | 0.4760 | 0.5769 |
| `between_class` | `left` | `SVM` | 0.8133 | 0.0000 | 0.8133 | 0.8133 |
| `between_class` | `left` | `ResNetAE` | 0.6181 | 0.0561 | 0.5553 | 0.6635 |
| `between_class` | `left` | `ResNet` | 0.5801 | 0.0116 | 0.5673 | 0.5897 |
| `between_class` | `left` | `MobileNet` | 0.5876 | 0.0193 | 0.5689 | 0.6074 |
| `between_class` | `left` | `MLP` | 0.7794 | 0.0257 | 0.7596 | 0.8085 |
| `between_class` | `right` | `SqueezeNet` | 0.6750 | 0.0402 | 0.6386 | 0.7182 |
| `between_class` | `right` | `SVM` | 0.7386 | 0.0000 | 0.7386 | 0.7386 |
| `between_class` | `right` | `ResNetAE` | 0.5652 | 0.0839 | 0.4977 | 0.6591 |
| `between_class` | `right` | `ResNet` | 0.6409 | 0.0532 | 0.5795 | 0.6727 |
| `between_class` | `right` | `MobileNet` | 0.6561 | 0.0667 | 0.5795 | 0.7023 |
| `between_class` | `right` | `MLP` | 0.8568 | 0.0357 | 0.8182 | 0.8886 |
| `within_class` | `left` | `SqueezeNet` | 0.5718 | 0.0143 | 0.5561 | 0.5841 |
| `within_class` | `left` | `SVM` | 0.8325 | 0.0000 | 0.8325 | 0.8325 |
| `within_class` | `left` | `ResNetAE` | 0.5884 | 0.0560 | 0.5553 | 0.6530 |
| `within_class` | `left` | `ResNet` | 0.6162 | 0.0514 | 0.5865 | 0.6755 |
| `within_class` | `left` | `MobileNet` | 0.6557 | 0.0644 | 0.5865 | 0.7139 |
| `within_class` | `left` | `MLP` | 0.7810 | 0.0257 | 0.7612 | 0.8101 |
| `within_class` | `right` | `SqueezeNet` | 0.6811 | 0.0191 | 0.6591 | 0.6932 |
| `within_class` | `right` | `SVM` | 0.7568 | 0.0000 | 0.7568 | 0.7568 |
| `within_class` | `right` | `ResNetAE` | 0.6902 | 0.0539 | 0.6455 | 0.7500 |
| `within_class` | `right` | `ResNet` | 0.7303 | 0.0773 | 0.6545 | 0.8091 |
| `within_class` | `right` | `MobileNet` | 0.6697 | 0.0400 | 0.6273 | 0.7068 |
| `within_class` | `right` | `MLP` | 0.8386 | 0.0000 | 0.8386 | 0.8386 |

## ผู้ชนะตามค่าเฉลี่ย balanced accuracy

| protocol | side | model | mean | SD |
|---|---|---|---:|---:|
| `between_class` | `left` | **SVM** | 0.8133 | 0.0000 |
| `between_class` | `right` | **MLP** | 0.8568 | 0.0357 |
| `within_class` | `left` | **SVM** | 0.8325 | 0.0000 |
| `within_class` | `right` | **MLP** | 0.8386 | 0.0000 |

## ตำแหน่งไฟล์

- สรุปทุก run: `PAIRWISE_TRAIN_SUMMARY.csv`
- สรุปแยก model: `MODEL_SUMMARY_BY_PROTOCOL_SIDE.csv`
- ผู้ชนะ: `BEST_BY_PROTOCOL_SIDE.csv`
- แต่ละ run อยู่ที่ `protocol/side/model/seed_<seed>/` และมี `model.pkl` หรือ `model_state.pt`, `training_log.csv`, `test_predictions.csv`, `metrics.json`, `run_manifest.json`

การตรวจสอบรอบนี้ยืนยัน 72/72 runs สำเร็จ, validation rows เป็นศูนย์, ไม่มี augmentation ซ้ำ และไม่มี PLS-DA ใน model input
