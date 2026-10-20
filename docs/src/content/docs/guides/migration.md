---
title: Decomposition and migration
description: The Decomposition protocol, the BoxDecomposition reference, and what migrate sends between ranks.
sidebar:
  order: 3
---

## The Decomposition protocol

`migrate` needs to know which rank owns a position. Anything with these members will do:

| Member                 | Meaning                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `comm`, `rank`, `size` | the communicator whose ranks own the subdomains                  |
| `owners(positions)`    | owning rank of every column of an `(ndim, n)` array, an int array |
| `owned_mask(positions)`| bool array: which columns this rank owns                         |

Both lookups take backend arrays (NumPy or CuPy) and must follow the same rule:
`owned_mask(x)[i]` is true exactly when `owners(x)[i] == rank`. Every position, also one outside the
domain, needs an owner. `owned_mask` is the cheap test run on every marker; `owners` is only evaluated
for the markers that failed it.

## BoxDecomposition

`BoxDecomposition(comm, lower, upper, proc_sizes, periodic=False)` splits a box into equal blocks,
one per rank, with ranks in row-major order (`numpy.unravel_index(rank, proc_sizes)`, the last axis
fastest). The lookup is vectorised (`searchsorted`). It is a reference implementation for tests and
examples; a code with its own decomposition writes a small adapter.

Out-of-range rule:

- A non-periodic axis clips: a position below `lower` belongs to the first block, and a position at or
  above `upper` to the last block (so `value == upper` is in the last block).
- A periodic axis first wraps the position into `[lower, upper)`; `upper` itself wraps to `lower`.
- A block edge belongs to the block above it.

## migrate

```python
sent, received = migrate(store, decomposition)
```

`migrate` is collective: every rank of `decomposition.comm` calls it, also a rank with nothing to
send. It

1. selects the leavers (alive markers not owned by this rank) with `owned_mask`;
2. looks up `owners` for the leavers only;
3. sends only the leavers with `mpiarray.migrate`, as two arrays: a float matrix with a
   `(position, velocity, weight)` row per marker, and the ids as a separate int64 array;
4. marks the leavers dead locally (the next `compact` drops them) and appends the arrivals,
   keeping their ids.

It returns the number of markers sent and received by this rank. With one rank it returns `(0, 0)`
and makes no MPI call. Because new markers are appended, the storage may grow; take views again
afterwards.

`keep_owned(store, decomposition)` is not collective: it drops the markers this rank does not own,
for markers that every rank loaded from the same global description.

## Global helpers

- `global_count(store, comm)` is the number of alive markers over all ranks (collective).
- `gather_columns(array, comm, root=0)` concatenates the `(ncomp, n)` or `(n,)` arrays of all ranks along
  the marker axis into a host NumPy array on `root` (`root=None` gives it to every rank), for checks
  and output.

Both do nothing but local work when `comm` is `None` or has one rank.
