"""Verify artifacts and execute real Jittor CPU operations from an installed wheel."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def check(artifact_dir: Path, work_dir: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    wheels = list(artifact_dir.glob("*.whl"))
    archives = list(artifact_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            top = name.split("/")[0]
            if top != "scifigbench" and not top.endswith(".dist-info"):
                raise ValueError(f"Unexpected wheel member: {name}")
        for kind in ("qa", "prediction", "report"):
            if f"scifigbench/schemas/{kind}.schema.json" not in names:
                raise ValueError(f"Packaged schema missing: {kind}")
        if not any(name.endswith("/LICENSE") for name in names):
            raise ValueError("Packaged license missing")
    with tarfile.open(archives[0], "r:gz") as archive:
        names = archive.getnames()
        for name in names:
            parts = Path(name).parts
            if any(part in {".git", ".venv", "annotation", "remote_patch"} for part in parts):
                raise ValueError(f"Unexpected source distribution member: {name}")
        if not any(name.endswith("examples/synthetic/expected.json") for name in names):
            raise ValueError("Source distribution lacks synthetic examples")
    work_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment.update({"use_cuda": "0", "use_mkl": "0", "DISABLE_MULTIPROCESSING": "1"})
    environment.setdefault("UV_CACHE_DIR", str(work_dir / "uv-cache"))
    with tempfile.TemporaryDirectory(prefix="wheel-check-", dir=work_dir) as temp:
        scratch = Path(temp).resolve()
        venv = scratch / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, env=environment)
        bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        command = bin_dir / ("scifigbench.exe" if os.name == "nt" else "scifigbench")
        constraints = scratch / "constraints.txt"
        subprocess.run([
            "uv", "export", "--frozen", "--no-dev", "--no-emit-project",
            "--no-hashes", "--output-file", str(constraints),
        ], check=True, cwd=repo, env=environment, stdout=subprocess.DEVNULL)
        subprocess.run([
            "uv", "pip", "install", "--python", str(python),
            "--constraint", str(constraints), str(wheel),
        ], check=True, cwd=scratch, env=environment)
        subprocess.run([str(command), "--version"], check=True, cwd=scratch, env=environment)
        probe = """
import importlib.util
import sys
from pathlib import Path
import scifigbench
from scifigbench import backend
from scifigbench.io import schema
from scifigbench.metrics import anls, set_anls
assert Path(scifigbench.__file__).resolve().is_relative_to(Path(sys.prefix))
assert all(schema(kind)['type'] == 'object' for kind in ('qa', 'prediction', 'report'))
assert all(importlib.util.find_spec(name) is None for name in ('torch', 'accelerate', 'xformers'))
assert anls('abc', 'abc') == 1
assert set_anls(['A', 'A'], ['A']) == 0.5
assert backend.exact_match(10**100 + 1, 10**100) == 0
info = backend.metadata()
assert info['name'] == 'jittor' and info['device'] == 'cpu' and info['dtype'] == 'float64'
assert not any(name in sys.modules for name in ('torch', 'accelerate', 'xformers'))
"""
        subprocess.run([str(python), "-c", probe], check=True, cwd=scratch, env=environment)
        subprocess.run([
            str(python), str(repo / "examples" / "synthetic" / "run_demo.py"),
            "--data-root", str(repo / "examples" / "synthetic"),
            "--out-dir", str(scratch / "demo"),
        ], check=True, cwd=scratch, env=environment)
    print("Wheel/source inspection, Jittor CPU execution and isolated CLI demo passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("dist"))
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    check(args.artifact_dir.resolve(), args.work_dir.resolve())
