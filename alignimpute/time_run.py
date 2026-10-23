from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def gpu_used(dev):
    out = subprocess.run(["nvidia-smi", "-i", str(dev), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--log-dir", type=Path, default=Path("logs/eswa/time"))
    a = ap.parse_args(); a.log_dir.mkdir(parents=True, exist_ok=True); a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for line in open(a.out):
            r = json.loads(line); done.add((r["method"], r["setting"]))
    for line in open(a.jobs):
        if not line.strip() or line.startswith("#"):
            continue
        method, setting, cmd = line.rstrip("\n").split("\t")
        if (method, setting) in done:
            continue
        base = gpu_used(a.gpu); peak = [base]; stop = threading.Event()

        def poll():
            while not stop.is_set():
                try:
                    peak[0] = max(peak[0], gpu_used(a.gpu))
                except Exception:
                    pass
                time.sleep(0.2)
        th = threading.Thread(target=poll, daemon=True); th.start()
        log = open(a.log_dir / f"{method}_{setting}.log", "w")
        t0 = time.time()
        p = subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(p.pid, 0)
        sec = time.time() - t0; stop.set(); th.join(); log.close()
        rec = {"method": method, "setting": setting, "seconds": round(sec, 1), "peak_rss_gb": round(ru.ru_maxrss / 2 ** 20, 2),
               "peak_gpu_gb": round((peak[0] - base) / 1024, 2), "exit": os.waitstatus_to_exitcode(status), "cmd": cmd}
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("method", "setting", "seconds", "peak_rss_gb", "peak_gpu_gb", "exit")}), flush=True)


if __name__ == "__main__":
    main()
