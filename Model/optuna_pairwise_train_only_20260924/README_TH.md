# รายงาน Optuna: pairwise augmentation แบบ train-only

รอบนี้ค้นหา hyperparameters ด้วย grouped 10-fold CV ภายใน train เท่านั้น โดยสร้าง pairwise augmentation ใหม่ภายในแต่ละ fold; test ไม่ถูกอ่านระหว่าง Optuna และ final fit ไม่มี validation เพิ่ม

- Studies: 24 (2 protocol × 2 side × 6 models)
- Trials: 10 ต่อ study
- Folds: 10 แบบ grouped
- Tuning epochs: 3; final epochs: 20
- Final seeds: `42, 123, 2026`
- PLS-DA: ไม่ใช้
- Final train: fit จาก train เดิมทั้งหมดหลังสร้าง pairwise augmentation ใหม่; test ใช้ประเมินครั้งสุดท้ายเท่านั้น

## Best hyperparameters จาก OOF

| protocol | side | model | OOF balanced accuracy | parameters |
|---|---|---|---:|---|
| `between_class` | `left` | **MLP** | 0.8441 | `{"batch_size": 32, "dropout": 0.4233589392465845, "lr": 0.0007896186801026692, "weight_decay": 4.809461967501575e-06}` |
| `between_class` | `right` | **MLP** | 0.8516 | `{"batch_size": 32, "dropout": 0.3099025726528951, "lr": 0.0016967533607196547, "weight_decay": 7.068974950624607e-06}` |
| `within_class` | `left` | **MLP** | 0.8515 | `{"batch_size": 8, "dropout": 0.4579309401710595, "lr": 0.000642033033629786, "weight_decay": 5.4880470007660465e-06}` |
| `within_class` | `right` | **SVM** | 0.8471 | `{"C": 0.14618962793704965, "gamma_mode": "auto"}` |

## ค่า test เฉลี่ยของผู้ชนะ (3 seeds)

| protocol | side | model | accuracy | balanced accuracy | sensitivity | specificity |
|---|---|---|---:|---:|---:|---:|
| `between_class` | `left` | **MLP** | 0.7883 | 0.7839 | 0.7692 | 0.7986 |
| `between_class` | `right` | **MLP** | 0.8533 | 0.8311 | 0.7833 | 0.8788 |
| `within_class` | `left` | **MLP** | 0.7658 | 0.7548 | 0.7179 | 0.7917 |
| `within_class` | `right` | **SVM** | 0.8533 | 0.8364 | 0.8000 | 0.8727 |

## ไฟล์ผลลัพธ์

- `OPTUNA_TUNING_SUMMARY.csv`: best OOF และพารามิเตอร์ของทั้ง 24 studies
- `OPTUNA_FINAL_SUMMARY.csv`: ผล test ของ final model ทุก model/seed รวม 72 runs
- `OPTUNA_WINNERS_BY_PROTOCOL_SIDE.csv`: ผู้ชนะจาก OOF balanced accuracy เท่านั้น
- `OPTUNA_WINNER_TEST_MEANS.csv`: ค่า test เฉลี่ยของผู้ชนะเพื่อรายงานแยกต่างหาก
- `studies/<protocol>/<side>/<model>/trials.csv`: รายละเอียดทุก trial
- `final/<protocol>/<side>/<model>/seed_<seed>/`: model, scaler, predictions, metrics และ run manifest

ตรวจสอบแล้ว: 24/24 studies, 72/72 final runs, validation ของ final fit = 0, augmentation อยู่ภายในแต่ละ CV training fold และไม่มี test leakage
