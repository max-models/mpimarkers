---
title: Installation
description: Create a Python environment and install mpiparticles.
---

mpiparticles needs Python 3.10 or newer. Create and activate an environment:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the package and its requirements (`numpy`, `cunumpy`, `maybempi` and `mpiarray`) with pip:

```bash
pip install -e .
```

Without an MPI launcher, mpiparticles runs on one rank with [maybempi](https://github.com/max-models/maybempi)'s
stand-in for `mpi4py`, and makes no MPI call. For several ranks install an MPI library and `mpi4py`, and
start the program with `mpiexec`:

```bash
pip install -e ".[mpi]"
mpiexec -n 4 python my_program.py
```

## Optional extras

| Extra  | Installs                                                     |
| ------ | ------------------------------------------------------------ |
| `mpi`  | `mpi4py`, needed to run on more than one rank                |
| `test` | `pytest` and `coverage`                                      |
| `docs` | the notebook runner and griffe for the API reference         |
| `dev`  | formatters and linters, plus the `mpi`, `test` and `docs` extras |

```bash
pip install -e ".[dev]"
```

The tests that start `mpiexec -n 2` and `-n 4` jobs are skipped when `mpiexec` or `mpi4py` is missing.
