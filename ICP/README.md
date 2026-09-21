# ICP: reference, batch และ single-file runner

ใช้คู่มือหลัก [RERUN_GUIDE_TH.md](../RERUN_GUIDE_TH.md) เพื่อเลือกระหว่าง legacy reference กับ reference ที่ fit จาก training masks เท่านั้น คู่มือนี้สรุปไฟล์ที่ใช้จริงในโฟลเดอร์ ICP

## Reference ที่กู้คืนจากผลเดิม

`references/legacy_groupwise_all_381_v1/` มี mean shape ซ้าย/ขวาจาก ICP groupwise เดิม 381 รายต่อข้าง พร้อม `.json` contract และ `reference_manifest.json` ระบุ hash, scale, grid และแหล่งที่มา ค่า scale ซ้ายคือ `0.030540622985912882`; ขวา `0.03180055903090514`; grid ทั้งคู่ `128³`, spacing `0.015625`.

reference นี้รวมรูปร่างจากผู้เข้าร่วมทั้งชุดเดิม; audit IDs ยืนยันว่า status ปัจจุบันซ้าย 373/373 และขวา 377/377 อยู่ในกลุ่มที่ใช้สร้าง reference แล้ว จึงใช้ตรึง coordinate convention/รัน single ใหม่เท่านั้น ห้ามใช้กับ test ของชุดนี้เพื่ออ้าง independent test evaluation การ rerun แบบ pairwise กับ mean shape ที่ freeze จะไม่รับประกัน transform เดิมทุกไบต์จากการรัน groupwise

## คำสั่ง Git Bash สำหรับ batch ซ้าย/ขวา

เปิด Git Bash จากโฟลเดอร์ใดก็ได้:

```bash
bash ~/Desktop/17-9-2569/Hippocampal-Shape-Analysis-for-Epilepsy-Detection/ICP/run_icp_legacy_reference_batch.sh
```

ตัว launcher ตรวจ preflight ของสองข้างก่อนเริ่ม และสร้าง run ID เดียวกันให้อัตโนมัติ สามารถส่ง run ID เองเป็นอาร์กิวเมนต์ตัวแรกได้

ถ้าไม่ระบุ `--input-dir` สคริปต์ใช้ input ที่พบใน workspace นี้โดยอัตโนมัติ:

```text
../merged_ds005602_ds004469/left_hippocampus   (381 masks)
../merged_ds005602_ds004469/right_hippocampus  (381 masks)
```

ฟังก์ชัน `python()` มีผลเฉพาะ Git Bash หน้าต่างนี้ ถ้า input อยู่ที่อื่น ให้เติม `--input-dir "PATH_TO_MASK_FOLDER"`; สำหรับไฟล์เดียวใช้ `--input-file "PATH_TO_MASK.nii.gz"` และระบุ `--side` ให้ตรงกับ hemisphere. สคริปต์ Python เปิด SlicerSALT โดยตรง รอให้ทำงานจบก่อนตรวจ status/จำนวน aligned volumes และเขียน `run_summary.json`. output จะถูกสร้างใหม่ใต้ `reruns/` อัตโนมัติ; log อยู่ใน `icp_debug_log.txt` ของ job

## ไฟล์โค้ด

- `ICP.py`: core alignment; ต้องระบุหนึ่งโหมดให้ชัด (`--reference_template`, `--fit_reference` หรือ `--exploratory_groupwise`) และรองรับ `--input_dir`/`--input_file`
- `run_icp_with_reference.py`: Python launcher สำหรับรัน fixed legacy reference ทีละข้างหรือทีละไฟล์ พร้อม preflight, รอ Slicer และตรวจ output
- `run_icp_legacy_reference_batch.sh`: Git Bash runner สำหรับตรวจและรัน batch ซ้าย/ขวาใน run เดียวกัน
- `reference_contract.py`: ตรวจ sidecar, template hash, scale, spacing และ grid; รองรับ training-only และ legacy provenance versions
- `create_legacy_reference_from_previous_run.py`: validator/builder ที่สร้าง reference bundle จากโฟลเดอร์ output groupwise เดิม หาก output เก่าถูกลบแล้วให้ใช้ bundle ที่สร้างไว้ ไม่สามารถสร้างซ้ำจากไฟล์ที่ไม่มีได้
- `rerun_fixed_reference.py`: Git Bash/Python workflow สำหรับแบ่ง mask train/test, fit reference จาก train เท่านั้น และตรวจ output ทั้งหก ICP jobs
- `prepare_mask_split.py`: ตรวจ subject IDs/labels กับ split manifest แล้ว copy mask โดยไม่ย้ายหรือลบต้นฉบับ
- `plot_icp_convergence.py`: สร้างกราฟจาก convergence history ของ exploratory/groupwise run

ไฟล์ `output_left_hippocampus/`, `output_right_hippocampus/` และ `icp_debug_log.txt` ที่เคยมีเป็นผลเก่าซึ่งจัดเก็บเฉพาะ reference bundle/manifest ที่ตรวจแล้วก่อนล้าง ไม่ใช่ output ปัจจุบัน
