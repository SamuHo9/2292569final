"""
Launcher สำหรับรัน run_spharm_batch.py แบบขนานหลาย Slicer instance
(ไม่ต้องรันผ่าน Slicer เอง - รันด้วย python ปกติจากนอก Slicer)

ทำไมต้องแยกเป็นหลาย process แทนที่จะ async ภายใน process เดียว:
  MRML scene ของ Slicer ไม่ thread-safe การสร้าง/ลบ node จากหลาย thread
  พร้อมกันในกระบวนการเดียวเสี่ยง race condition และ crash การแยกเป็น
  Slicer.exe instance คนละตัว (คนละ process, คนละ MRML scene) จึงปลอดภัยกว่า
  และตัวที่กินเวลาจริง (SegPostProcessCLP.exe, GenParaMeshCLP.exe,
  ParaToSPHARMMeshCLP.exe) เป็น subprocess ของตัวเองอยู่แล้ว ไม่ใช้ GPU/
  ทรัพยากรร่วมที่ต้อง sync กันระหว่าง instance

คุณภาพผลลัพธ์เหมือนเดิม 100% เพราะไม่ได้แตะ numIterations/subdivLevel/
spharmDegree เลย แค่กระจาย "จำนวน subject" ไปรันบนหลาย core พร้อมกัน

วิธีใช้ (auto - แนะนำ ไม่ต้องคิดเลขเอง):
    python run_spharm_parallel.py \
        --slicer_exe "C:/Program Files/SlicerSALT 6.0.0/SlicerSALT.exe" \
        --input_dir path/to/aligned_nifti \
        --output_dir path/to/output \
        --reference_template path/to/template_spharm_left.vtk

ไม่ต้องใส่ --num_workers เลย - script จะตรวจ physical CPU core และ RAM ว่างจริง
ของเครื่อง ณ ตอนรัน แล้วเลือกจำนวน worker ที่ปลอดภัยที่สุดให้เอง (เอาค่าที่น้อย
กว่าระหว่างข้อจำกัดของ CPU กับ RAM) พร้อมพิมพ์เหตุผลที่มาของตัวเลขให้เห็น

ถ้าอยากบังคับจำนวนเอง ใส่ --num_workers <จำนวน> ได้ตามปกติ (ยังโดนเช็ค RAM
ป้องกันไว้เหมือนเดิม เว้นแต่ใส่ --force)
"""
import os
import sys
import argparse
import subprocess
import threading
import time
import platform
from datetime import datetime


def get_ram_info_gb():
    """
    คืน (total_gb, available_gb) ของ RAM ทั้งเครื่อง
    ลองใช้ psutil ก่อน (แม่นยำสุด, cross-platform) ถ้าไม่มีใช้คำสั่งของแต่ละ OS แทน
    คืน (None, None) ถ้าหาไม่ได้เลย (จะข้ามการเช็คแทนที่จะบล็อกการรัน)
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)
    except ImportError:
        pass

    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/Value"],
                text=True, stderr=subprocess.DEVNULL)
            vals = dict(line.split("=") for line in out.strip().splitlines() if "=" in line)
            total_gb = int(vals["TotalVisibleMemorySize"]) / (1024**2)
            avail_gb = int(vals["FreePhysicalMemory"]) / (1024**2)
            return total_gb, avail_gb
        elif platform.system() == "Linux":
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":")
                    info[k.strip()] = int(v.strip().split()[0])  # kB
            total_gb = info.get("MemTotal", 0) / (1024**2)
            avail_gb = info.get("MemAvailable", info.get("MemFree", 0)) / (1024**2)
            return total_gb, avail_gb
        elif platform.system() == "Darwin":
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            return total / (1024**3), None  # macOS ไม่มี "available" ตรงๆ แบบง่าย
    except Exception:
        pass

    return None, None


def get_cpu_core_info():
    """
    คืน (physical_cores, logical_cores)
    physical_cores อาจเป็น None ถ้าหาไม่ได้ (เช่นไม่มี psutil) - ตอนนั้นจะประมาณจาก logical/2
    """
    logical = os.cpu_count()  # มีเสมอใน stdlib ไม่ต้องพึ่ง psutil
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return physical, logical


def recommend_num_workers(ram_per_worker_gb):
    """
    คำนวณจำนวน worker ที่แนะนำจาก CPU core และ RAM ว่างพร้อมกัน
    คืน (n, reason_lines) - reason_lines คือคำอธิบายสำหรับ print ให้ผู้ใช้เห็นที่มาของตัวเลข
    """
    lines = []
    physical, logical = get_cpu_core_info()
    if physical:
        cpu_n = max(1, physical - 1)  # เผื่อ 1 core ให้ OS/งานอื่น
        lines.append(f"  CPU: {physical} physical core ({logical} logical) -> เผื่อ 1 core ให้ OS = {cpu_n}")
    else:
        cpu_n = max(1, logical // 2)  # ไม่รู้ physical ให้ประมาณว่า logical มี hyperthread 2 เท่า
        lines.append(f"  CPU: ไม่พบจำนวน physical core (ติดตั้ง psutil จะแม่นกว่า) "
                     f"-> ประมาณจาก {logical} logical core / 2 = {cpu_n}")

    total_gb, avail_gb = get_ram_info_gb()
    if avail_gb is not None:
        ram_n = max(1, int((avail_gb * 0.85) / ram_per_worker_gb))
        lines.append(f"  RAM: ว่าง {avail_gb:.1f} GB จาก {total_gb:.1f} GB -> ที่ {ram_per_worker_gb:.1f} GB/worker "
                     f"เผื่อ margin 15% = {ram_n}")
    else:
        ram_n = cpu_n
        lines.append(f"  RAM: ตรวจไม่ได้ (ติดตั้ง psutil จะช่วยได้) -> ใช้ตัวเลขจาก CPU แทน")

    n = max(1, min(cpu_n, ram_n))
    bottleneck = "CPU" if cpu_n <= ram_n else "RAM"
    lines.append(f"  => เลือก {n} worker (ตัวจำกัดคือ {bottleneck})")
    return n, lines


def sum_children_rss_gb(pids):
    """รวม RSS (หน่วยความจำจริงที่ใช้อยู่) ของ process ทั้งหมด รวมลูกหลาน ถ้ามี psutil"""
    try:
        import psutil
        total = 0
        for pid in pids:
            try:
                p = psutil.Process(pid)
                total += p.memory_info().rss
                for child in p.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        pass
            except Exception:
                pass
        return total / (1024**3)
    except ImportError:
        return None


def ram_monitor_loop(procs, stop_event, interval_sec=20):
    """แสดงหน่วยความจำที่ใช้จริงทุก N วินาทีระหว่างรัน (ต้องมี psutil)"""
    pids = [p.pid for p in procs]
    while not stop_event.wait(interval_sec):
        used_gb = sum_children_rss_gb(pids)
        _, avail_gb = get_ram_info_gb()
        if used_gb is not None:
            msg = f"[RAM monitor] worker ทั้งหมดใช้จริงรวม ~{used_gb:.1f} GB"
            if avail_gb is not None:
                msg += f"  |  ว่างเหลือในระบบ ~{avail_gb:.1f} GB"
            print(msg)


def parse_args():
    default_slicer = r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
    if not os.path.exists(default_slicer):
        cand5 = r"C:\Program Files\SlicerSALT 5.0\SlicerSALT.exe"
        if os.path.exists(cand5):
            default_slicer = cand5

    p = argparse.ArgumentParser(description="รัน run_spharm_batch.py แบบขนานหลาย Slicer instance")
    p.add_argument("--slicer_exe", default=default_slicer, help="path ไปยัง Slicer.exe หรือ SlicerSALT.exe")
    p.add_argument("--script", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "run_spharm_batch.py"),
                   help="path ไปยัง run_spharm_batch.py")
    p.add_argument("--num_workers", type=int, default=None,
                   help="จำนวน Slicer instance ที่รันพร้อมกัน ไม่ระบุ = ตรวจ CPU core + RAM ว่างอัตโนมัติแล้วเลือกให้เอง")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reference_template", default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--num_iterations", type=int, default=None)
    p.add_argument("--subdiv_level", type=int, default=None)
    p.add_argument("--spharm_degree", type=int, default=None)
    p.add_argument("--extra_args", default="", help="อาร์กิวเมนต์เพิ่มเติมที่จะส่งต่อไปยัง run_spharm_batch.py ตรงๆ")
    p.add_argument("--ram_per_worker_gb", type=float, default=1.0,
                   help="ประมาณการ RAM ที่แต่ละ Slicer instance ใช้ (GB) สำหรับเช็คก่อนรัน ค่าเริ่มต้น=1.0 "
                        "(Slicer เปล่าๆ กินราว 0.3-0.5GB บวก CLI subprocess ราว 0.2-0.3GB รวม ~0.6-0.8GB)")
    p.add_argument("--force", action="store_true",
                   help="ข้ามการเช็ค RAM แล้วรันตามจำนวน worker ที่ขอไว้เลย ถึงจะดูไม่พอก็ตาม")
    return p.parse_args()


def stream_output(proc, shard_idx, lock):
    """อ่าน stdout ของแต่ละ worker แบบ real-time พร้อม prefix บอกว่าเป็น shard ไหน"""
    for line in iter(proc.stdout.readline, ''):
        if not line:
            break
        with lock:
            print(f"[shard {shard_idx}] {line.rstrip()}")
    proc.stdout.close()


def main():
    args = parse_args()
    args.slicer_exe = os.path.abspath(args.slicer_exe)
    args.script = os.path.abspath(args.script)
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.reference_template:
        args.reference_template = os.path.abspath(args.reference_template)

    if not os.path.isfile(args.slicer_exe):
        print(f"!!! ไม่พบ Slicer executable ที่: {args.slicer_exe}")
        sys.exit(1)
    if not os.path.isfile(args.script):
        print(f"!!! ไม่พบ run_spharm_batch.py ที่: {args.script}")
        sys.exit(1)

    if args.num_workers is None:
        print("=== ไม่ได้ระบุ --num_workers -> ตรวจ CPU core + RAM ว่างเพื่อเลือกให้อัตโนมัติ ===")
        n, reason_lines = recommend_num_workers(args.ram_per_worker_gb)
        for line in reason_lines:
            print(line)
        was_auto = True
    else:
        n = max(1, args.num_workers)
        was_auto = False

    # ---- เช็ค RAM ก่อนรันจริง (เช็คซ้ำเสมอ แม้ตอน auto ก็ตาม เผื่อสถานะ RAM เปลี่ยนระหว่างคำนวณ) ----
    total_gb, avail_gb = get_ram_info_gb()
    if total_gb is None:
        print("! ไม่สามารถตรวจสอบ RAM ของเครื่องได้ (ติดตั้ง psutil ด้วย `pip install psutil` "
              "จะช่วยให้เช็ค/มอนิเตอร์ RAM ได้แม่นยำขึ้น) ข้ามการเช็คไปเลย")
    else:
        need_gb = n * args.ram_per_worker_gb
        print(f"RAM ทั้งเครื่อง: {total_gb:.1f} GB  |  ว่างตอนนี้: {avail_gb:.1f} GB")
        print(f"ประมาณการที่ {n} worker จะใช้: ~{need_gb:.1f} GB (ที่ {args.ram_per_worker_gb:.1f} GB/worker)")
        if need_gb > avail_gb * 0.85:
            safe_n = max(1, int((avail_gb * 0.85) / args.ram_per_worker_gb))
            print(f"!!! คำเตือน: {n} worker อาจใช้ RAM เกินที่ว่างอยู่ (เหลือ margin น้อยเกินไป)")
            print(f"    แนะนำลดเหลือ {safe_n} worker หรือปิดโปรแกรมอื่นก่อนรัน")
            if not args.force:
                print(f"    ลดจำนวน worker จาก {n} เหลือ {safe_n} อัตโนมัติ (ใช้ --force เพื่อบังคับใช้ {n} "
                      f"ตามที่ขอไว้ หรือ --num_workers {safe_n} เพื่อตั้งค่าที่แนะนำแบบชัดเจน)")
                n = safe_n
            else:
                print(f"    --force ถูกระบุไว้ -> รัน {n} worker ตามที่ขอ (เสี่ยงเครื่องอืด/ค้างถ้า RAM ไม่พอจริง)")

    if was_auto:
        print(f"=== ตัวเลขสุดท้ายที่ใช้จริง: {n} worker (ถ้าอยากบังคับเอง ใส่ --num_workers <จำนวน>) ===")
    print(f"=== เริ่มรัน {n} Slicer instance พร้อมกัน (shard 0..{n-1}) ===")
    print(f"เริ่มเวลา: {datetime.now()}")
    t0 = time.time()

    procs = []
    threads = []
    lock = threading.Lock()

    for shard_idx in range(n):
        cmd = [
            args.slicer_exe, "--no-splash", "--no-main-window",
            # SPHARM needs Slicer core + CLI modules, but not optional scripted
            # modules that try to install unrelated packages at startup.
            "--disable-scripted-loadable-modules",
            "--python-script", args.script,
            "--", 
            "--input_dir", args.input_dir,
            "--output_dir", args.output_dir,
            "--num_shards", str(n),
            "--shard_index", str(shard_idx),
        ]
        if args.reference_template:
            cmd += ["--reference_template", args.reference_template]
        if args.fast:
            cmd += ["--fast"]
        if args.num_iterations is not None:
            cmd += ["--num_iterations", str(args.num_iterations)]
        if args.subdiv_level is not None:
            cmd += ["--subdiv_level", str(args.subdiv_level)]
        if args.spharm_degree is not None:
            cmd += ["--spharm_degree", str(args.spharm_degree)]
        if args.extra_args:
            cmd += args.extra_args.split()

        print(f"[shard {shard_idx}] เริ่ม: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
        procs.append(proc)
        th = threading.Thread(target=stream_output, args=(proc, shard_idx, lock), daemon=True)
        th.start()
        threads.append(th)

    exit_codes = None
    stop_event = threading.Event()
    monitor_th = threading.Thread(target=ram_monitor_loop, args=(procs, stop_event), daemon=True)
    monitor_th.start()
    try:
        exit_codes = [p.wait() for p in procs]
    finally:
        stop_event.set()
        monitor_th.join(timeout=2)
    for th in threads:
        th.join(timeout=5)

    elapsed = time.time() - t0
    print(f"\n=== ทุก shard เสร็จแล้ว ใช้เวลารวม {elapsed/60:.1f} นาที ===")
    for i, code in enumerate(exit_codes):
        status = "OK" if code == 0 else f"!!! exit code={code}"
        print(f"  shard {i}: {status}")

    # รวม log จากทุก shard เป็นไฟล์เดียว เรียงตามเวลา
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    merged_path = os.path.join(log_dir, "spharm_debug_log_merged.txt")
    log_lines = []
    for i in range(n):
        shard_log = os.path.join(log_dir, f"spharm_debug_log_shard{i}of{n}.txt")
        if os.path.isfile(shard_log):
            with open(shard_log, encoding="utf-8", errors="replace") as f:
                log_lines.extend(f.readlines())
    log_lines.sort(key=lambda ln: ln[1:9] if len(ln) > 9 and ln[0] == "[" else "")
    with open(merged_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)
    print(f"รวม log ทุก shard ไว้ที่: {merged_path}")

    if any(c != 0 for c in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
