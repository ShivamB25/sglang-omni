# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github/scripts/delete_gpu_process.sh"


def cleanup_environment(tmp_path: Path, memory: str, child_pid: int) -> dict[str, str]:
    probe = tmp_path / "nvidia-smi"
    probe.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if '--query-compute-apps=pid' in sys.argv:\n"
        "    print(os.environ['OWNED_TEST_PID'])\n"
        "elif '--query-gpu=memory.used' in sys.argv:\n"
        "    print(os.environ['TEST_GPU_MEMORY'])\n"
        "else:\n"
        "    sys.exit(99)\n"
    )
    probe.chmod(0o755)
    return os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMNI_CI_GPU_CLEAN_WAIT_SECONDS": "0",
        "OMNI_CI_GPU_CLEAN_POLL_SECONDS": "1",
        "OWNED_TEST_PID": str(child_pid),
        "TEST_GPU_MEMORY": memory,
    }


@pytest.mark.parametrize("flags", [["--wait-only"], ["--wait-only", "--kill-orphans"]])
def test_wait_only_never_terminates_reported_process(
    tmp_path: Path, flags: list[str]
) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT), *flags],
            env=cleanup_environment(tmp_path, "0", child.pid),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


@pytest.mark.parametrize("memory", ["", "N/A", "-1", "4096"])
def test_wait_only_rejects_unknown_or_busy_memory(tmp_path: Path, memory: str) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT), "--wait-only"],
            env=cleanup_environment(tmp_path, memory, child.pid),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)
