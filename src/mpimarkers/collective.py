"""Global quantities over the markers of all ranks."""

from __future__ import annotations

from typing import Any

import cunumpy as xp
import maybempi
import numpy as np

from mpimarkers.store import ParticleStore


def global_count(store: ParticleStore, comm: Any = None) -> int:
    """Return the number of alive markers over all ranks of ``comm``.

    Collective on ``comm``; with ``comm=None`` or one rank it is the local count
    and no MPI call is made.
    """
    count = store.num_alive
    if comm is None or comm.Get_size() == 1:
        return count
    return int(comm.allreduce(count, op=maybempi.get_mpi().SUM))


def gather_columns(array: Any, comm: Any = None, root: int | None = 0) -> Any:
    """Concatenate component-major ``(ncomp, n)`` or ``(n,)`` arrays over the ranks.

    Each rank passes its own columns (for example ``store.active_positions()``
    or ``store.active_ids()``); the marker axis is the last one and ``n`` may
    differ between ranks. The result is a NumPy array on the host, ordered by
    rank. Collective on ``comm``; with ``comm=None`` or one rank the array is
    just copied to the host.

    Parameters
    ----------
    array : array
        This rank's columns.
    comm : communicator, optional
        The communicator; default: serial.
    root : int or None, optional
        The rank that receives the result (the others get ``None``), or
        ``None`` to give every rank the result.
    """
    local = np.asarray(xp.to_numpy(array))
    if comm is None or comm.Get_size() == 1:
        return local
    pieces = comm.allgather(local) if root is None else comm.gather(local, root=root)
    if pieces is None:
        return None
    return np.concatenate(pieces, axis=-1)
