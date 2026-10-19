#!/usr/bin/env python3
"""Steps of the "OCR job" workflow (.github/workflows/ocr-job.yml).

  plan   read params.json → GITHUB_OUTPUT shards=[0, 1, …]
  shard  fetch the job, OCR shard $SHARD of $SHARDS, upload shards/shard-<i>.json
         (+ progress-<i>.json every ~30 s, log-<i>.txt)
  merge  fetch the job + every shard, assemble with --merge-shards, upload
         derived/*, log.txt and status.json (always — "failed" if a shard failed)

Talks to the bucket with the AWS CLI (preinstalled on GitHub's runners), using
ENDPOINT, BUCKET, JOB_ID and the AWS_* credentials from the environment. Prints
nothing about the game: OCR output goes to log files in the bucket.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

JOB = Path("job")
INPUTS = ["game.json", "roi.json", "markers.json", "app/data.json",
          "defaults/roi.json", "defaults/roi_templates.json", "params.json"]


def env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        sys.exit(f"{name} is not set")
    return v


JOB_ID = env("JOB_ID")
if not re.fullmatch(r"[0-9a-f]{32}", JOB_ID):
    sys.exit("job_id must be 32 hex characters")
PREFIX = f"s3://{env('BUCKET')}/ocr-jobs/{JOB_ID}/"
S3 = ["aws", "s3", "cp", "--endpoint-url", env("ENDPOINT"), "--only-show-errors"]


def get(rel: str, dest: Path, required: bool = False) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = subprocess.run(S3 + [PREFIX + rel, str(dest)], capture_output=True).returncode == 0
    if required and not ok:
        sys.exit(f"could not fetch {rel}")
    return ok


def put(src: Path, rel: str, wait: bool = True) -> None:
    cmd = S3 + [str(src), PREFIX + rel]
    if wait:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd)


def params() -> dict:
    try:
        return json.loads((JOB / "params.json").read_text())
    except (OSError, ValueError):
        return {}


def fetch_inputs() -> None:
    for rel in INPUTS:
        get(rel, JOB / rel, required=(rel == "game.json"))


def ocr_cmd(*extra: str) -> tuple[list[str], dict]:
    p = params()
    e = dict(os.environ)
    for name, var in (("roi.json", "PLYOCR_DEFAULT_ROI"), ("roi_templates.json", "PLYOCR_ROI_TEMPLATES")):
        if (JOB / "defaults" / name).exists():
            e[var] = str((JOB / "defaults" / name).resolve())
    # Every shard and the merge must cut the video into the same chunks, so the
    # worker count is explicit (never "auto", which depends on the machine).
    cmd = [sys.executable, "-m", "plyocr.run_tesseract_ocr", "--game", str(JOB / "game.json"),
           "--fps", str(p.get("fps", 0.5)), "--workers", str(p.get("workers") or 4),
           "--decoder", str(p.get("decoder", "keyframe")), "--derived", str(JOB / "derived"), *extra]
    return cmd, e


def run_logged(cmd: list[str], e: dict, log_path: Path, progress_rel: str | None) -> int:
    """Run the OCR with its output in log_path; relay PROGRESS to the bucket
    (the first one at once — it carries this shard's total — then every ~30 s)."""
    last, prog = 0.0, None
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            m = re.match(r"PROGRESS\s+(\d+)/(\d+)", line)
            if m:
                prog = {"done": int(m.group(1)), "total": int(m.group(2))}
            if progress_rel and prog and time.time() - last >= 30:
                last = time.time()
                (JOB / "progress.json").write_text(json.dumps(prog))
                put(JOB / "progress.json", progress_rel, wait=False)
        rc = proc.wait()
    if progress_rel and prog:
        (JOB / "progress.json").write_text(json.dumps(prog))
        put(JOB / "progress.json", progress_rel)
    return rc


def plan() -> None:
    JOB.mkdir(exist_ok=True)
    get("params.json", JOB / "params.json")
    n = max(1, min(20, int(params().get("shards") or 6)))
    with open(os.environ["GITHUB_OUTPUT"], "a") as out:
        out.write(f"shards={json.dumps(list(range(n)))}\n")
    print(f"shards={n}")


def shard() -> None:
    i, n = int(env("SHARD")), int(env("SHARDS"))
    fetch_inputs()
    cmd, e = ocr_cmd("--shard", f"{i}/{n}")
    t0 = time.time()
    rc = run_logged(cmd, e, JOB / "log.txt", f"shards/progress-{i}.json")
    put(JOB / "log.txt", f"shards/log-{i}.txt")
    print(f"shard {i + 1}/{n}: exit={rc} seconds={round(time.time() - t0)}")
    if rc != 0:
        sys.exit(rc)
    put(JOB / "derived" / f"shard-{i}.json", f"shards/shard-{i}.json")


def merge() -> None:
    t0 = time.time()
    status: dict = {"status": "failed", "exitCode": None, "rows": 0}
    try:
        if os.environ.get("SHARD_RESULT") != "success":
            status["reason"] = "a shard failed"
            return
        fetch_inputs()
        n = max(1, min(20, int(params().get("shards") or 6)))
        files = [JOB / "shards" / f"shard-{i}.json" for i in range(n)]
        for i, f in enumerate(files):
            get(f"shards/shard-{i}.json", f, required=True)
        cmd, e = ocr_cmd("--merge-shards", *map(str, files))
        rc = run_logged(cmd, e, JOB / "merge-log.txt", None)
        # One log: every shard's, then the merge.
        parts = []
        for i in range(n):
            if get(f"shards/log-{i}.txt", JOB / f"log-{i}.txt"):
                parts.append(f"── shard {i + 1}/{n} ──\n" + (JOB / f"log-{i}.txt").read_text())
        parts.append("── merge ──\n" + (JOB / "merge-log.txt").read_text())
        (JOB / "log.txt").write_text("\n".join(parts))
        put(JOB / "log.txt", "log.txt")
        rows = 0
        try:
            rows = len(json.loads((JOB / "derived" / "timeline.json").read_text()))
        except (OSError, ValueError):
            pass
        status = {"status": "done" if rc == 0 else "failed", "exitCode": rc, "rows": rows, "shards": n}
        if rc == 0:
            for f in (JOB / "derived").iterdir():
                if f.is_file() and not f.name.startswith("shard-"):
                    put(f, f"derived/{f.name}")
    finally:
        status["seconds"] = round(time.time() - t0)
        status["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        JOB.mkdir(exist_ok=True)
        (JOB / "status.json").write_text(json.dumps(status))
        put(JOB / "status.json", "status.json")
        print(f"merge: {status['status']} rows={status['rows']}")
    if status["status"] != "done":
        sys.exit(1)


if __name__ == "__main__":
    {"plan": plan, "shard": shard, "merge": merge}[sys.argv[1]]()
