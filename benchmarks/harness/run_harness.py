import json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
RESULTS_DIR = ROOT / "benchmarks" / "results"
PROFILES_DIR = ROOT / "benchmarks" / "profiles"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def safe_run(cmd, cwd):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    dt = time.time() - t0
    return p.returncode, dt, out, err

def try_git(args):
    try:
        return subprocess.check_output(["git"] + args, cwd=str(ROOT), text=True).strip()
    except Exception:
        return ""

def env_fingerprint():
    # NOTE: we do not assert CUDA exists; we record what we can.
    py = sys.version.replace("\n"," ")
    gpu_query = ""
    try:
        gpu_query = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    except Exception:
        gpu_query = ""

    # packages: small + safe snapshot
    pkgs = ""
    try:
        pkgs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).strip()
    except Exception:
        pkgs = ""

    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": py,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES",""),
        "gpu_query": gpu_query,
        "packages": pkgs.splitlines() if pkgs else []
    }

def main():
    ts = utc_now()
    stamp = ts.replace(":","").replace("-","")
    out_json = RESULTS_DIR / f"results_{stamp}.json"

    # Known bench scripts (repo-truth: do not guess beyond these)
    benches = [
        ("microbench_spherical_harmonics", "benchmarks/microbench/bench_spherical_harmonics.py", "microbench"),
        ("microbench_tensor_product",      "benchmarks/microbench/bench_tensor_product.py",      "microbench"),
        ("microbench_neighbor_list",       "benchmarks/microbench/bench_neighbor_list.py",       "microbench"),
        ("end2end_message_passing",        "benchmarks/end2end/bench_message_passing.py",        "end2end"),
    ]

    runs = []
    for name, rel, kind in benches:
        path = ROOT / rel
        run = {
            "name": name,
            "path": rel,
            "kind": kind,
            "status": "FAILED_BUILD",
            "exit_code": None,
            "elapsed_sec": None,
            "stdout_path": "",
            "stderr_path": ""
        }

        if not path.exists():
            # keep FAILED_BUILD but record reason in stderr artifact
            stderr_p = RESULTS_DIR / f"{name}_{stamp}.stderr.txt"
            stderr_p.write_text(f"missing file: {rel}\n", encoding="utf-8")
            run["stderr_path"] = str(stderr_p.relative_to(ROOT))
            runs.append(run)
            continue

        cmd = [sys.executable, str(path)]
        rc, dt, out, err = safe_run(cmd, cwd=ROOT)

        stdout_p = RESULTS_DIR / f"{name}_{stamp}.stdout.txt"
        stderr_p = RESULTS_DIR / f"{name}_{stamp}.stderr.txt"
        stdout_p.write_text(out or "", encoding="utf-8")
        stderr_p.write_text(err or "", encoding="utf-8")

        run["exit_code"] = int(rc)
        run["elapsed_sec"] = float(dt)
        run["stdout_path"] = str(stdout_p.relative_to(ROOT))
        run["stderr_path"] = str(stderr_p.relative_to(ROOT))

        if rc == 0:
            run["status"] = "OK"
        else:
            run["status"] = "FAILED_RUNTIME"

        runs.append(run)

    ok = sum(1 for r in runs if r["status"] == "OK")
    fail = len(runs) - ok

    payload = {
        "contract": "batmobile.next.benchmarks",
        "version": os.environ.get("BATMOBILE_TAG",""),
        "timestamp_utc": ts,
        "git": {
            "commit": try_git(["rev-parse","HEAD"]),
            "branch": try_git(["rev-parse","--abbrev-ref","HEAD"]),
            "status": try_git(["status","--porcelain"])
        },
        "environment": env_fingerprint(),
        "runs": runs,
        "summary": {
            "ok": ok,
            "failed": fail
        },
        "notes": "Artifact truth only. No performance claims implied by existence of results."
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[BATMOBILE] wrote → {out_json}")

if __name__ == "__main__":
    # allow PS to inject version tag
    os.environ.setdefault("BATMOBILE_TAG", "")
    main()
