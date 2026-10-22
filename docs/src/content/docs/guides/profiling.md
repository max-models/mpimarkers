---
title: Profiling
description: The scope-profiler regions mpiparticles ships, off unless the application turns them on.
sidebar:
  order: 4
---

## Opt-in profiling regions

mpiparticles times its expensive operations with [scope-profiler](https://github.com/max-models/scope-profiler)
regions named `mpiparticles.<operation>`: `migrate`, `keep_owned`, `gather_columns`, and the store's
`reserve` (only when the storage grows), `compact`, `sort_by_key`, `snapshot` and `restore`. They are
tagged `"mpiparticles"`, an opt-in tag, so they stay off (and record nothing) when an application
profiles its own code, until it asks for them:

```python
import scope_profiler as sp

sp.setup(enable_tags=["mpiparticles"])
```

`session(enable_tags=...)` and `ProfilingOptions(enable_tags=...)` take the same argument.

The same works without touching the code: `enable_tags = ["mpiparticles"]` in the TOML
configuration, `SCOPE_PROFILER_ENABLE_TAGS=mpiparticles` in the environment, or
`scope-profiler run --enable-tag mpiparticles script.py`. `"*"` enables every opt-in tag.

A region that is off costs a few tenths of a microsecond per call, so only operations over all
markers are instrumented, never per-marker work.
