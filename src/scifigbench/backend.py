"""Lazy, single-process Jittor CPU execution for evaluation arithmetic."""
from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import sysconfig
import threading
from collections.abc import Sequence


class BackendError(RuntimeError):
    """The required evaluation runtime could not execute."""


_lock = threading.RLock()
_jittor = None
_active = False


@contextlib.contextmanager
def _compiler_output_to_stderr():
    """Keep Python and native compiler messages out of the CLI's JSON stdout."""
    saved_fd = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except (OSError, ValueError):
        if saved_fd is not None:
            os.close(saved_fd)
            saved_fd = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        if saved_fd is not None:
            sys.stderr.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)


@contextlib.contextmanager
def cpu_runtime():
    """Execute and synchronize CPU float64 operations without a fallback backend."""
    global _jittor, _active
    with _lock:
        if _active:
            yield _jittor
            return
        with _compiler_output_to_stderr():
            try:
                if not sys.platform.startswith("linux"):
                    raise BackendError("Jittor evaluation requires Linux or WSL2 in this release")
                if _jittor is None:
                    settings = {name: os.environ.get(name)
                                for name in ("nvcc_path", "use_mkl", "python_config_path")}
                    # venv does not copy pythonX.Y-config beside its interpreter.
                    # Jittor needs that helper even when Python.h is already installed.
                    if "python_config_path" not in os.environ:
                        bindir = Path(sysconfig.get_config_var("BINDIR") or sys.base_prefix)
                        candidates = [
                            bindir / f"python{sys.version_info.major}.{sys.version_info.minor}-config",
                            Path(sys.base_prefix) / "bin" / f"python3.{sys.version_info.minor}-config",
                        ]
                        for candidate in candidates:
                            if candidate.is_file():
                                os.environ["python_config_path"] = str(candidate)
                                break
                    os.environ.update({"nvcc_path": "", "use_mkl": "0"})
                    try:
                        jt = importlib.import_module("jittor")
                    except (Exception, SystemExit) as exc:
                        # Jittor's compiler discovery also raises plain Exception.
                        raise BackendError(
                            "Jittor CPU initialization failed. Check the locked dependencies, "
                            f"C++ compiler and Python development headers. No fallback was used. Details: {exc}"
                        ) from exc
                    finally:
                        for name, previous in settings.items():
                            if previous is None:
                                os.environ.pop(name, None)
                            else:
                                os.environ[name] = previous
                else:
                    jt = _jittor
                with jt.no_grad(use_cuda=0, auto_convert_64_to_32=0, amp_reg=0):
                    if _jittor is None:
                        probe = jt.array([1.0, 2.0], dtype="float64").sum()
                        if probe.dtype != "float64" or probe.item() != 3.0:
                            raise BackendError("Jittor CPU float64 initialization failed")
                        _jittor = jt
                    _active = True
                    try:
                        yield jt
                        jt.sync_all()
                    finally:
                        _active = False
            except BackendError:
                raise
            except (ImportError, OSError, RuntimeError, AssertionError, SystemExit) as exc:
                raise BackendError(
                    "Jittor CPU execution failed. Install the locked dependencies and a C++ "
                    f"compiler as described in the README. No fallback was used. Details: {exc}"
                ) from exc


def metadata() -> dict:
    with cpu_runtime() as jt:
        return {
            "name": "jittor", "version": jt.__version__, "device": "cpu",
            "dtype": "float64",
            "assignment_solver": {"name": "scipy", "version": importlib.metadata.version("scipy")},
        }


def mean(values: Sequence[float]) -> float:
    with cpu_runtime() as jt:
        return float(jt.array(list(values) or [0.0], dtype="float64").mean().item())


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    with cpu_runtime() as jt:
        return float((jt.array([numerator], dtype="float64") / denominator).item())


def exact_match(reference: int | str, prediction: int | str) -> float:
    # Canonical decimal strings preserve Python integers beyond int64/float64 ranges.
    left, right = str(reference), str(prediction)
    size = max(len(left), len(right), 1)
    left_codes = [ord(c) for c in left] + [-1] * (size - len(left))
    right_codes = [ord(c) for c in right] + [-1] * (size - len(right))
    with cpu_runtime() as jt:
        equal = jt.array(left_codes, dtype="int32") == jt.array(right_codes, dtype="int32")
        return float(equal.all().float64().item())


def edit_distances(jt, pairs: Sequence[tuple[Sequence[str], Sequence[str]]]):
    """One Jittor CPU code op computes exact token distances for a nonempty batch."""
    if not pairs:
        raise ValueError("edit_distances requires at least one pair")
    vocabulary: dict[str, int] = {}
    left, right, left_offsets, right_offsets = [], [], [0], [0]
    for a, b in pairs:
        for tokens, packed, offsets in ((a, left, left_offsets), (b, right, right_offsets)):
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
                packed.append(vocabulary[token])
            offsets.append(len(packed))
    inputs = [jt.array(values or [0], dtype="int32")
              for values in (left, right, left_offsets, right_offsets)]
    return jt.code(
        [len(pairs)], "int32", inputs,
        cpu_header="#include <algorithm>\n#include <numeric>\n#include <vector>",
        cpu_src=r"""
        for (int p = 0; p < out_shape0; ++p) {
            const int a0 = @in2(p), a1 = @in2(p + 1);
            const int b0 = @in3(p), b1 = @in3(p + 1);
            const int n = a1 - a0, m = b1 - b0;
            std::vector<int> previous(m + 1), current(m + 1);
            std::iota(previous.begin(), previous.end(), 0);
            for (int i = 1; i <= n; ++i) {
                current[0] = i;
                for (int j = 1; j <= m; ++j) {
                    const int substitution = previous[j - 1]
                        + (@in0(a0 + i - 1) == @in1(b0 + j - 1) ? 0 : 1);
                    current[j] = std::min(std::min(current[j - 1] + 1,
                                                  previous[j] + 1), substitution);
                }
                previous.swap(current);
            }
            @out(p) = previous[m];
        }
        """,
    )


def similarities(jt, pairs, *, threshold: float):
    distances = edit_distances(jt, pairs).float64()
    lengths = jt.array([max(len(a), len(b)) for a, b in pairs], dtype="float64")
    # Both empty -> 1; one empty -> 0. A safe denominator avoids invalid intermediate values.
    scores = 1.0 - distances / jt.maximum(lengths, 1.0)
    thresholded = jt.where(scores >= threshold, scores, jt.zeros_like(scores))
    return jt.where(lengths == 0, jt.ones_like(scores), thresholded)
