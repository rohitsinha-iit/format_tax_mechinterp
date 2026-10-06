"""Shared helpers: paths, configs, jsonl I/O with metadata headers, git hash."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
ACTS = Path(os.environ.get("FTMI_ACTS", ROOT / "acts"))
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figs"
MODEL_KEYS = ("qwen3-4b", "qwen3-8b")


def load_cfg(name: str) -> dict:
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def model_info(key: str) -> dict:
    cfg = load_cfg("models")
    info = dict(cfg["models"][key])
    info["key"] = key
    return info


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def meta_record(**kw) -> dict:
    return {"_meta": {"time": now(), "git": git_hash(), "argv": sys.argv, **kw}}


def read_jsonl(path, with_meta: bool = False):
    """Read records, skipping `_meta` header lines unless with_meta."""
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # a torn last line from a killed run; the uid will simply be regenerated
                continue
            if "_meta" in r and not with_meta:
                continue
            rows.append(r)
    return rows


def done_uids(path) -> set:
    return {r["uid"] for r in read_jsonl(path) if "uid" in r}


class JsonlWriter:
    """Append-mode writer; writes a `_meta` header each time it is opened."""

    def __init__(self, path, meta: dict, flush_every: int = 25):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resumed = self.path.exists() and self.path.stat().st_size > 0
        self.f = open(self.path, "a", encoding="utf-8")
        self.flush_every = flush_every
        self.n = 0
        self.write(meta_record(resumed=resumed, **meta))
        self.f.flush()

    def write(self, rec: dict):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1
        if self.n % self.flush_every == 0:
            self.flush()

    def flush(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.flush()
        self.f.close()


def write_jsonl(path, rows, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if meta is not None:
            f.write(json.dumps(meta_record(**meta), ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def log(msg: str, logfile=None):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")
