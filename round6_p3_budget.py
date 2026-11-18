"""Round 6 phase 3 budget (sacct, all of this user's jobs since the phase-3 start). Writes results/round6/budget.json.

GPU hours = sum over jobs of gres/gpu x elapsed; CPU core hours = AllocCPUS x elapsed for jobs without a GPU
(the CPU cores of GPU jobs are listed separately). Limits: 60 GPU h, 3000 CPU core h; stop at 80%.
"""
import json
import re
import subprocess
import sys

START = "2026-11-18T23:20"
LIMIT_GPU, LIMIT_CPU = 60.0, 3000.0


def main():
    out = subprocess.run(["sacct", "-X", "-S", START, "-E", "now", "-o", "JobID,JobName,AllocCPUS,AllocTRES,ElapsedRaw,State",
                          "-P", "-n"], capture_output=True, text=True, check=True).stdout
    gpu_h = cpu_h = cpu_of_gpu_jobs_h = 0.0
    jobs = []
    for line in out.splitlines():
        jid, name, ncpu, tres, el, state = line.split("|")
        if not name.startswith("iso-"):
            continue
        el, ncpu = int(el or 0), int(ncpu or 0)
        m = re.search(r"gres/gpu=(\d+)", tres)
        ng = int(m.group(1)) if m else 0
        if ng:
            gpu_h += ng * el / 3600
            cpu_of_gpu_jobs_h += ncpu * el / 3600
        else:
            cpu_h += ncpu * el / 3600
        jobs.append({"job": jid, "name": name, "gpus": ng, "cpus": ncpu, "elapsed_s": el, "state": state})
    res = {"phase": "round 6 phase 3", "since": START, "gpu_hours": round(gpu_h, 3), "cpu_core_hours": round(cpu_h, 2),
           "cpu_core_hours_of_gpu_jobs": round(cpu_of_gpu_jobs_h, 2), "limit_gpu_hours": LIMIT_GPU,
           "limit_cpu_core_hours": LIMIT_CPU, "frac_gpu": round(gpu_h / LIMIT_GPU, 4), "frac_cpu": round(cpu_h / LIMIT_CPU, 4),
           "stop_at_80pct": bool(gpu_h >= 0.8 * LIMIT_GPU or cpu_h >= 0.8 * LIMIT_CPU), "n_jobs": len(jobs), "jobs": jobs}
    json.dump(res, open("results/round6/budget.json", "w"), indent=1)
    print(f"GPU {gpu_h:.2f} / {LIMIT_GPU} h ({100 * gpu_h / LIMIT_GPU:.1f}%), CPU {cpu_h:.1f} / {LIMIT_CPU} core h "
          f"({100 * cpu_h / LIMIT_CPU:.1f}%), {len(jobs)} jobs" + ("  ** 80% reached: stop **" if res["stop_at_80pct"] else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
