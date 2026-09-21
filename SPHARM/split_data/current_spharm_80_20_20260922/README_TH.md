# ชุดข้อมูล SPHARM แบ่ง Train/Test 80/20

สร้างเมื่อ 2026-09-22 โดยใช้ผล SPHARM จาก `reruns/icp_legacy_reference_20260921_172727`

แบ่งระดับผู้ป่วยด้วย seed `20260919` และใช้ assignment เดียวกันทั้งซ้ายและขวา จึงไม่มี Subject เดียวกันอยู่ทั้ง train และ test ไฟล์ coefficients ที่คัดลอกเป็น `_SPHARM_ellalign.coef` พร้อม `_processing.json` และ `labels.csv` ส่วนโฟลเดอร์ train มี mesh ตัวอย่างจาก train ใน `_template/` สำหรับใช้เป็น topology ตอนทำ PLS-DA augmentation

| ชุดข้อมูล | ด้าน | Train Healthy/TLE | Test Healthy/TLE | รวม |
|---|---|---:|---:|---:|
| All_coef | left | 191/104 (295) | 48/26 (74) | 369 |
| All_coef | right | 220/82 (302) | 55/20 (75) | 377 |
| All_coef_Ds004469 | left | 37/17 (54) | 10/2 (12) | 66 |
| All_coef_Ds004469 | right | 42/11 (53) | 11/1 (12) | 65 |
| All_coef_Ds005602 | left | 154/87 (241) | 38/24 (62) | 303 |
| All_coef_Ds005602 | right | 178/71 (249) | 44/19 (63) | 312 |

ไฟล์ SPHARM ฝั่งซ้าย 4 รายการไม่มี coefficient เพราะขั้น SPHARM ไม่ผ่าน แต่ coefficient ฝั่งขวาของผู้ป่วยทั้ง 4 รายยังอยู่ในชุดขวา ไม่ได้ตัดออก

**ข้อจำกัดการประเมิน:** ICP reference ที่ใช้รอบนี้เป็น legacy reference ที่สร้างจากผู้ป่วยทั้งหมด 381 ราย (`independent_test_reference=false`) ดังนั้นชุดนี้พร้อมสำหรับจัดข้อมูลและทดลองขั้นตอน แต่คะแนนจาก test ยังอ้างว่าเป็นการประเมินอิสระไม่ได้ หากต้องการรายงานผลวิจัยแบบ held-out ต้อง fit ICP reference จาก train เท่านั้น แล้วรัน ICP/SPHARM ใหม่ให้ train และ test ก่อน

ใช้ `summary_counts.csv` ดูจำนวน, `sample_manifest.csv` ตรวจรายชื่อและ hash ของไฟล์, และ `split_manifest.json` ตรวจ assignment ระดับผู้ป่วย ห้ามทำ augmentation กับโฟลเดอร์ test
