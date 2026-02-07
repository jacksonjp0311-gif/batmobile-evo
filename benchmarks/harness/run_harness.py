import json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
RESULTS_DIR = ROOT / "benchmarks" / "results"
PROFILES_DIR = ROOT / "benchmarks" / "profiles"
LOGS_DIR = ROOT / "logs"
ARTIFACTS_DIR = ROOT / "artifacts"
REGISTRY_PATH = ROOT / "benchmarks" / "bench_registry.json"
SCHEMA_PATH = ROOT / "benchmarks" / "results_schema.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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

def load_bench_registry():
    if not REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    benches = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        path = entry.get("path")
        kind = entry.get("kind", "unknown")
        if name and path:
            benches.append((name, path, kind))
    return benches

def load_schema():
    if not SCHEMA_PATH.exists():
        return {}
    try:
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def validate_payload(payload, schema):
    errors = []
    required_top = schema.get("required_top_level", [])
    if required_top:
        missing = [key for key in required_top if key not in payload]
        if missing:
            errors.append(f"missing top-level fields: {missing}")

    env_fields = schema.get("environment_fields", [])
    environment = payload.get("environment", {})
    missing_env = [key for key in env_fields if key not in environment]
    if missing_env:
        errors.append(f"missing environment fields: {missing_env}")

    run_fields = schema.get("run_fields", [])
    status_enum = set(schema.get("status_enum", []))
    for idx, run in enumerate(payload.get("runs", [])):
        missing_run = [key for key in run_fields if key not in run]
        if missing_run:
            errors.append(f"run[{idx}] missing fields: {missing_run}")
        if status_enum and run.get("status") not in status_enum:
            errors.append(f"run[{idx}] invalid status: {run.get('status')}")

    return errors

def main():
    ts = utc_now()
    stamp = ts.replace(":","").replace("-","")
    out_json = RESULTS_DIR / f"results_{stamp}.json"

    benches = load_bench_registry()
    if not benches:
        print("[BATMOBILE] No benchmark registry found or it is empty.")
        print(f"[BATMOBILE] Expected: {REGISTRY_PATH}")
        return 1

    runs = []
    log_lines = []
    log_lines.append(f"[BATMOBILE] registry: {REGISTRY_PATH}")
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
            log_lines.append(f"[{name}] MISSING: {rel}")
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

        log_lines.append(f"[{name}] cmd: {' '.join(cmd)}")
        log_lines.append(f"[{name}] exit_code: {rc} elapsed_sec: {dt:.4f}")
        if out:
            log_lines.append(f"[{name}] stdout:\n{out}")
        if err:
            log_lines.append(f"[{name}] stderr:\n{err}")

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

    schema = load_schema()
    contract_errors = validate_payload(payload, schema) if schema else []
    if contract_errors:
        log_lines.append("[BATMOBILE] contract validation FAILED")
        for err in contract_errors:
            log_lines.append(f"[BATMOBILE] contract error: {err}")
    else:
        log_lines.append("[BATMOBILE] contract validation OK")

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[BATMOBILE] wrote → {out_json}")
    log_path = LOGS_DIR / f"run_{stamp}.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    latest_path = ARTIFACTS_DIR / f"latest_{stamp}.txt"
    latest_path.write_text(str(out_json.relative_to(ROOT)) + "\n", encoding="utf-8")
    print(f"[BATMOBILE] log → {log_path}")
    print(f"[BATMOBILE] latest → {latest_path}")

    if contract_errors:
        return 2
    if fail > 0:
        return 1
    return 0

if __name__ == "__main__":
    # allow PS to inject version tag
    os.environ.setdefault("BATMOBILE_TAG", "")
    raise SystemExit(main())
