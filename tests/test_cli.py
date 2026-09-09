import json
import subprocess
import sys


def test_installed_cli_from_another_directory(examples, tmp_path):
    cli = [sys.executable, "-m", "scifigbench"]
    version = subprocess.run(cli + ["--version"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert version.stdout.strip() == "scifigbench 0.2.0"
    task = "edge_level_verification"
    args = [
        "evaluate", "--task", task, "--qa", str(examples / "qa" / f"{task}.jsonl"),
        "--pred", str(examples / "predictions" / "correct" / f"{task}.jsonl"),
        "--report-dir", str(tmp_path / "reports"),
    ]
    success = subprocess.run(cli + args, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(success.stdout)["primary_metric"]["value"] == 1
    refusal = subprocess.run(cli + args, cwd=tmp_path, capture_output=True, text=True)
    assert refusal.returncode == 2
    assert "--overwrite" in refusal.stderr
    invalid = subprocess.run(cli + ["validate", "--qa", "missing.jsonl"], cwd=tmp_path, capture_output=True, text=True)
    assert invalid.returncode == 2 and "error:" in invalid.stderr
