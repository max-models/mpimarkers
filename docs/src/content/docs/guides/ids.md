---
title: Ids
description: Globally unique int64 marker ids with a rank stride, independent of the decomposition.
sidebar:
  order: 2
---

Every marker has an int64 id that never changes, also when the marker migrates. `IdAllocator(rank,
size)` gives rank `r` of `n` the ids `r, r + n, r + 2n, ...`, so ids are unique across ranks without
communication.

```python
from mpiparticles import IdAllocator, ParticleStore

store = ParticleStore(2, 3, id_allocator=IdAllocator(rank=1, size=4))
```

The rank and size are arguments, not read from the decomposition. When every rank holds its own copy
of the grid (a decomposition on `COMM_SELF`, each rank then loads its own share of the markers), the
decomposition has one rank per process, but the ids must still differ from rank to rank: pass the rank
and size of the world communicator, or use `IdAllocator.from_comm(world)`. The default allocator is the
serial one (rank 0 of 1), which is only right on a single rank.

- `allocate(count)` returns `count` new ids as an int64 array.
- `advance_past(ids)` moves the counter above given ids. `ParticleStore.add(..., ids=...)` calls it,
  so arriving markers cannot collide with new ones.
- `next_id` is the next id to hand out; it is saved by `ParticleStore.snapshot()` and restored by `restore()`.

Ids are sent as int64 during migration and sorted as int64, so they are exact above 2^53.
`clear()` empties a store without resetting the counter: ids are never reused.
