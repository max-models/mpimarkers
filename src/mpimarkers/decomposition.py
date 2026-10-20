"""Which rank owns a position: the decomposition protocol and a box reference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import cunumpy as xp
import maybempi
import numpy as np

from mpimarkers._types import Array


@runtime_checkable
class Decomposition(Protocol):
    """Splits space between ranks; what :func:`~mpimarkers.migrate` needs.

    Positions are component-major, ``(ndim, n)`` like the store's, and arrays
    live on the store's backend (NumPy or CuPy). Both methods must follow the
    same rule: ``owned_mask(x)[i]`` is true exactly when ``owners(x)[i] ==
    rank``. ``owned_mask`` is the cheap test applied to every marker; ``owners``
    is only evaluated for the markers that failed it. Every position, also one
    outside the domain, must have an owner.
    """

    @property
    def comm(self) -> Any:
        """Return the communicator whose ranks own the subdomains."""
        ...

    @property
    def rank(self) -> int:
        """Return this rank."""
        ...

    @property
    def size(self) -> int:
        """Return the number of ranks."""
        ...

    def owners(self, positions: Array) -> Array:
        """Return the owning rank of each position, an int array of shape ``(n,)``."""
        ...

    def owned_mask(self, positions: Array) -> Array:
        """Return which positions this rank owns, a bool array of shape ``(n,)``."""
        ...


class BoxDecomposition:
    """A box split into equal blocks, one per rank (a reference decomposition).

    The block of rank ``r`` has the block coordinates ``numpy.unravel_index(r,
    proc_sizes)``: ranks are row-major, the last axis varies fastest. Owner
    lookup is vectorised (``searchsorted`` on the block edges).

    **Out-of-range positions.** For a non-periodic axis a position below
    ``lower`` belongs to the first block and a position at or above ``upper``
    to the last block (so ``value == upper`` is in the last block, and markers
    that left through a wall still have an owner). For a periodic axis the
    position is first wrapped into ``[lower, upper)``; the value ``upper``
    itself wraps to ``lower``. NaN positions are owned by the last block.

    Parameters
    ----------
    comm : communicator, optional
        MPI communicator with ``prod(proc_sizes)`` ranks; default
        ``MPI.COMM_WORLD`` (a serial stand-in without an MPI launcher).
    lower, upper : sequence of float
        Corners of the box, one entry per axis.
    proc_sizes : sequence of int
        Number of blocks along each axis.
    periodic : bool or sequence of bool, optional
        Whether each axis (or all axes) wraps around.

    Examples
    --------
    >>> deco = BoxDecomposition(None, [0.0, 0.0], [1.0, 1.0], [1, 1])
    >>> deco.owners(xp.asarray([[0.2, 3.0], [0.5, 0.5]])).tolist()
    [0, 0]
    """

    def __init__(
        self,
        comm: Any,
        lower: Sequence[float],
        upper: Sequence[float],
        proc_sizes: Sequence[int],
        periodic: bool | Sequence[bool] = False,
    ) -> None:
        """Create the decomposition of the box among the ranks of ``comm``."""
        self._comm = maybempi.get_mpi().COMM_WORLD if comm is None else comm
        self._lower = tuple(float(v) for v in lower)
        self._upper = tuple(float(v) for v in upper)
        self._proc_sizes = tuple(int(n) for n in proc_sizes)
        ndim = len(self._proc_sizes)
        if len(self._lower) != ndim or len(self._upper) != ndim:
            raise ValueError("lower, upper and proc_sizes need one entry per axis")
        if any(n < 1 for n in self._proc_sizes):
            raise ValueError("proc_sizes must be positive")
        if any(u <= lo for lo, u in zip(self._lower, self._upper, strict=True)):
            raise ValueError("upper must be above lower on every axis")
        if isinstance(periodic, bool):
            self._periodic = (periodic,) * ndim
        else:
            self._periodic = tuple(bool(p) for p in periodic)
            if len(self._periodic) != ndim:
                raise ValueError("periodic needs one entry per axis")
        size = int(np.prod(self._proc_sizes))
        if size != self._comm.Get_size():
            raise ValueError(
                f"proc_sizes {self._proc_sizes} need {size} ranks, "
                f"the communicator has {self._comm.Get_size()}"
            )
        self._edges = [
            np.linspace(lo, hi, n + 1)
            for lo, hi, n in zip(
                self._lower, self._upper, self._proc_sizes, strict=True
            )
        ]
        self._coords = tuple(
            int(c) for c in np.unravel_index(self.rank, self._proc_sizes)
        )

    @property
    def comm(self) -> Any:
        """Return the communicator."""
        return self._comm

    @property
    def rank(self) -> int:
        """Return this rank."""
        return int(self._comm.Get_rank())

    @property
    def size(self) -> int:
        """Return the number of ranks."""
        return int(self._comm.Get_size())

    @property
    def ndim(self) -> int:
        """Return the number of axes."""
        return len(self._proc_sizes)

    @property
    def proc_sizes(self) -> tuple[int, ...]:
        """Return the number of blocks along each axis."""
        return self._proc_sizes

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Return whether each axis wraps around."""
        return self._periodic

    def bounds(self, rank: int | None = None) -> tuple[tuple[float, ...], ...]:
        """Return ``(lower, upper)`` corners of the block of ``rank`` (default: own)."""
        coords = (
            self._coords if rank is None else np.unravel_index(rank, self._proc_sizes)
        )
        low = tuple(float(self._edges[d][int(c)]) for d, c in enumerate(coords))
        high = tuple(float(self._edges[d][int(c) + 1]) for d, c in enumerate(coords))
        return low, high

    def _axis(self, positions: Array, dim: int) -> Array:
        """Return the coordinates along ``dim``, wrapped if the axis is periodic."""
        x = positions[dim]
        if self._periodic[dim]:
            lo, hi = self._lower[dim], self._upper[dim]
            x = lo + (x - lo) % (hi - lo)
        return x

    def owners(self, positions: Array) -> Array:
        """Return the owning rank of every position, int64 of shape ``(n,)``."""
        positions = xp.asarray(positions)
        owners = xp.zeros(positions.shape[1], dtype=xp.int64)
        for dim, count in enumerate(self._proc_sizes):
            if count == 1:
                continue
            edges = xp.asarray(self._edges[dim][1:-1])
            block = xp.searchsorted(edges, self._axis(positions, dim), side="right")
            stride = int(np.prod(self._proc_sizes[dim + 1 :], dtype=np.int64))
            owners += block.astype(xp.int64) * stride
        return owners

    def owned_mask(self, positions: Array) -> Array:
        """Return which positions this rank owns (two comparisons per axis)."""
        positions = xp.asarray(positions)
        owned = xp.ones(positions.shape[1], dtype=bool)
        for dim, count in enumerate(self._proc_sizes):
            if count == 1:
                continue
            block = self._coords[dim]
            x = self._axis(positions, dim)
            edges = self._edges[dim]
            # the same comparisons as searchsorted(side="right") on the inner
            # edges; the first and last block are open towards the outside
            if block == count - 1:
                owned &= (x >= edges[block]) | xp.isnan(x)  # NaN: last block
            else:
                if block > 0:
                    owned &= x >= edges[block]
                owned &= x < edges[block + 1]
        return owned
