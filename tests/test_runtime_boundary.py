"""Data and image operations stay usable without initializing the scoring runtime."""
import os
import subprocess
import sys

import pytest


def test_import_data_and_payload_do_not_initialize_jittor(dataset):
    script = '''
import importlib.abc, sys
class BlockJittor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jittor' or fullname.startswith('jittor.'):
            raise ImportError('blocked-runtime')
sys.meta_path.insert(0, BlockJittor())
from scifigbench import load_qa, prepare_model_payload
from scifigbench.backend import BackendError, metadata
rows = load_qa(sys.argv[1])
payload = prepare_model_payload(rows[0], data_root=sys.argv[1])
assert set(payload) == {'prompt', 'image'}
payload['image'].close()
assert 'jittor' not in sys.modules
try:
    metadata()
except BackendError:
    pass
else:
    raise AssertionError('backend silently fell back')
'''
    result = subprocess.run([sys.executable, "-c", script, str(dataset[0])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="Native unsupported-platform error")
def test_native_windows_cli_has_actionable_error(dataset, predictions, tmp_path):
    import json
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps({"qid": k, "response": v}) + "\n"
                            for k, v in predictions.items()), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "scifigbench", "evaluate", "--data-root",
                             str(dataset[0]), "--predictions", str(path)], capture_output=True, text=True)
    assert result.returncode == 2 and not result.stdout
    assert "Linux or WSL2" in result.stderr and "Traceback" not in result.stderr


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Compiler initialization requires Linux")
def test_missing_compiler_is_actionable(tmp_path):
    env = dict(os.environ, cc_path=str(tmp_path / "missing-cxx"),
               JITTOR_HOME=str(tmp_path / "jittor"), use_mkl="0")
    script = '''
from scifigbench.backend import BackendError, metadata
try:
    metadata()
except BackendError as error:
    assert 'compiler' in str(error) and 'No fallback' in str(error)
else:
    raise AssertionError('missing compiler did not stop scoring')
'''
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_no_bundled_model_framework():
    import importlib.util
    assert all(importlib.util.find_spec(name) is None for name in ("torch", "accelerate", "xformers"))
    assert not any(name in sys.modules for name in ("torch", "accelerate", "xformers"))
