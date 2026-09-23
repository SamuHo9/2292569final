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
