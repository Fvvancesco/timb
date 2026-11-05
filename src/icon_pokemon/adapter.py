"""Small JSON-only boundary to the pinned local Showdown installation."""
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def showdown(script, payload, timeout=60):
    started = perf_counter()
    result = subprocess.run(
        ["node", str(ROOT / "tools/showdown" / script)],
        input=json.dumps(payload), text=True, capture_output=True, timeout=timeout,
        check=True, cwd=ROOT,
    )
    parsed = json.loads(result.stdout)
    if script == "simulate.cjs":
        parsed["subprocess_wall_seconds"] = perf_counter() - started
    return parsed


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
