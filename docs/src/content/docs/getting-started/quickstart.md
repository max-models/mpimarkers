---
title: Quickstart
description: Store markers, decompose a box between ranks and migrate the markers to their owners.
---

First, ensure that `mpimarkers` is [installed](/mpimarkers/getting-started/installation/).

## One rank

A `ParticleStore` keeps the markers of one rank in component-major arrays: positions `(ndim, n)`,
velocities `(nvel, n)`, weights, alive flags and int64 ids.

```python
import cunumpy as xp

from mpimarkers import ParticleStore

store = ParticleStore(ndim=2, nvel=3)
ids = store.add(xp.asarray([[0.1, 0.9], [0.2, 0.8]]), xp.zeros((3, 2)), weights=1.0)

store.deactivate(xp.asarray([False, True]))  # dead markers stay until compacted
store.compact_if_needed()
print(store.num_alive, store.length)
```

## Several ranks

Split a box between the ranks with a `Decomposition`, push the markers, and call `migrate` after
every push. It is collective: every rank calls it, also a rank with nothing to send.

```python
import maybempi

from mpimarkers import BoxDecomposition, IdAllocator, ParticleStore, migrate

comm = maybempi.get_mpi().COMM_WORLD
decomposition = BoxDecomposition(comm, [0.0, 0.0], [1.0, 1.0], proc_sizes=[2, 2])
store = ParticleStore(2, 3, id_allocator=IdAllocator.from_comm(comm))

# ... add the markers of this rank, push them ...
sent, received = migrate(store, decomposition)
```

```bash
mpiexec -n 4 python example.py   # or just: python example.py (one rank)
```

The [tutorial](/mpimarkers/tutorials/moving-markers/) runs this on four ranks and checks that no
marker or id is lost. See the [API reference](/mpimarkers/api/mpimarkers/) for every function.
