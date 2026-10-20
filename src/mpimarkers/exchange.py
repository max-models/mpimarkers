"""Moving markers to the rank that owns them."""

from __future__ import annotations

import cunumpy as xp
import mpiarray as mpa

from mpimarkers._types import Array
from mpimarkers.decomposition import Decomposition
from mpimarkers.store import ParticleStore


def _leavers(store: ParticleStore, decomposition: Decomposition) -> Array:
    """Return the indices of the alive markers this rank does not own."""
    end = store.length
    if end == 0:
        return xp.zeros(0, dtype=xp.int64)
    owned = decomposition.owned_mask(store.positions[:, :end])
    return xp.nonzero(store.alive[:end] & ~owned)[0]


def migrate(store: ParticleStore, decomposition: Decomposition) -> tuple[int, int]:
    """Send the markers that left this rank's subdomain to their owners.

    Collective: every rank of ``decomposition.comm`` must call it, also a rank
    with nothing to send (it still takes part in the exchange). Only the leavers
    travel: the cheap ``owned_mask`` test finds them, ``owners`` is evaluated
    for them alone, and they are sent with :func:`mpiarray.migrate` as two
    arrays, one float matrix with a row ``(position, velocity, weight)`` per
    marker and the ids as a separate int64 array, so ids are never cast to float
    and stay exact above ``2**53``. The leavers are marked dead here (the next
    :meth:`~ParticleStore.compact` drops them) and the arrivals are appended
    with their ids kept.

    With one rank (a serial run) nothing is exchanged and no MPI call is made.

    Parameters
    ----------
    store : ParticleStore
        This rank's markers.
    decomposition : Decomposition
        Which rank owns which position.

    Returns
    -------
    tuple of int
        The number of markers sent away and received by this rank.
    """
    if decomposition.size == 1:
        return 0, 0
    leaving = _leavers(store, decomposition)
    count = int(leaving.size)
    ndim, nvel = store.ndim, store.nvel
    if count:
        destinations = xp.asarray(
            decomposition.owners(store.positions[:, leaving]), dtype=xp.int64
        )
        packed = xp.ascontiguousarray(
            xp.concatenate(
                [
                    store.positions[:, leaving],
                    store.velocities[:, leaving],
                    store.weights[leaving][None, :],
                ],
                axis=0,
            ).T
        )
        leaving_ids = store.ids[leaving]
    else:
        destinations = xp.zeros(0, dtype=xp.int64)
        packed = xp.zeros((0, ndim + nvel + 1), dtype=store.dtype)
        leaving_ids = xp.zeros(0, dtype=xp.int64)
    arrived, arrived_ids = mpa.migrate(
        destinations, packed, leaving_ids, comm=decomposition.comm
    )
    if count:
        mask = xp.zeros(store.length, dtype=bool)
        mask[leaving] = True
        store.deactivate(mask)
    received = int(arrived_ids.shape[0])
    if received:
        columns = arrived.T
        store.add(
            columns[:ndim],
            columns[ndim : ndim + nvel],
            columns[ndim + nvel],
            ids=arrived_ids,
        )
    return count, received


def keep_owned(store: ParticleStore, decomposition: Decomposition) -> int:
    """Drop the markers this rank does not own (not collective).

    For markers every rank loaded from the same global description: each rank
    keeps its share. Removes the foreign markers and, when there were any, the
    dead rows too (a compaction). Returns the number dropped.
    """
    leaving = _leavers(store, decomposition)
    count = int(leaving.size)
    if count:
        mask = xp.zeros(store.length, dtype=bool)
        mask[leaving] = True
        store.deactivate(mask)
        store.compact()
    return count
