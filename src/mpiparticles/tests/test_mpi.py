"""Run the MPI job under ``mpiexec -n 2`` and ``-n 4`` (skipped without MPI)."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JOB = Path(__file__).with_name("_mpi_job.py")
SRC = str(Path(__file__).resolve().parents[2])  # the package, also when not installed
MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")

pytestmark = pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
    reason="needs mpiexec and mpi4py",
)


def run(size, *args):
    command = [MPIEXEC, "-n", str(size)]
    version = subprocess.run([MPIEXEC, "--version"], capture_output=True, text=True)
    if "Open MPI" in version.stdout + version.stderr:
        command.append("--oversubscribe")
    command += [sys.executable, str(JOB), *args]
    path = os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")]))
    environment = {**os.environ, "PYTHONPATH": path}
    return subprocess.run(
        command, capture_output=True, text=True, timeout=180, env=environment
    )


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("args", [(), ("big",)], ids=["ids", "ids-above-2**53"])
def test_migration_job(size, args):
    result = run(size, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"OK size={size}" in result.stdout
