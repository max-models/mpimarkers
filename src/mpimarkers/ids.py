"""Globally unique marker ids across ranks."""

from __future__ import annotations

from typing import Any

import cunumpy as xp

from mpimarkers._types import Array


class IdAllocator:
    """Hand out int64 ids that are unique across all ranks.

    Rank ``r`` of ``size`` ranks owns the ids ``r, r + size, r + 2 * size, ...``
    (a rank stride), so no communication is needed to keep ids unique, and a
    marker keeps its id when it migrates to another rank.

    The rank and size are arguments, independent of the communicator of a
    decomposition: when every rank holds its own copy of the grid (a
    decomposition on ``COMM_SELF``) the markers still need ids that differ from
    rank to rank, so pass the rank and size of the *world* communicator.

    Parameters
    ----------
    rank : int, optional
        This rank's position in the id sequence, ``0 <= rank < size``.
    size : int, optional
        The number of ranks that allocate ids, the stride of the sequence.
    next_id : int, optional
        The first id to hand out; must be ``rank`` modulo ``size``. Default:
        ``rank``.

    Examples
    --------
    >>> ids = IdAllocator(rank=1, size=4)
    >>> [int(i) for i in ids.allocate(3)]
    [1, 5, 9]
    >>> ids.next_id
    13
    """

    def __init__(self, rank: int = 0, size: int = 1, *, next_id: int | None = None):
        """Create an allocator for ``rank`` of ``size``."""
        if size < 1 or not 0 <= rank < size:
            raise ValueError(f"rank {rank} is not in 0..{size - 1}")
        self._rank = int(rank)
        self._size = int(size)
        self._next_id = self._rank
        if next_id is not None:
            self.next_id = next_id

    @classmethod
    def from_comm(cls, comm: Any) -> IdAllocator:
        """Return the allocator of ``comm``'s rank and size."""
        return cls(comm.Get_rank(), comm.Get_size())

    @property
    def rank(self) -> int:
        """Return the rank whose ids this allocator hands out."""
        return self._rank

    @property
    def size(self) -> int:
        """Return the stride between this rank's ids."""
        return self._size

    @property
    def next_id(self) -> int:
        """Return the next id that will be handed out (assign to restore a state)."""
        return self._next_id

    @next_id.setter
    def next_id(self, value: int) -> None:
        value = int(value)
        if value < 0 or (value - self._rank) % self._size:
            raise ValueError(
                f"next_id {value} is not {self._rank} modulo {self._size}, so it is "
                "not in this rank's sequence"
            )
        self._next_id = value

    def allocate(self, count: int) -> Array:
        """Return ``count`` new ids as an int64 array (on the backend's device)."""
        if count < 0:
            raise ValueError("count must be >= 0")
        start = self._next_id
        self._next_id += self._size * count
        if self._next_id > 2**63 - 1:
            raise OverflowError("the int64 ids of this rank are used up")
        return start + self._size * xp.arange(count, dtype=xp.int64)

    def advance_past(self, ids: Array) -> None:
        """Make sure no later id collides with ``ids`` (ids of arriving markers).

        The next id becomes the first one in this rank's sequence above every
        given id; it never decreases.
        """
        ids = xp.asarray(ids)
        if ids.size == 0:
            return
        lower_bound = int(xp.max(ids)) + 1
        steps = max(0, -(-(lower_bound - self._rank) // self._size))
        self._next_id = max(self._next_id, self._rank + steps * self._size)
