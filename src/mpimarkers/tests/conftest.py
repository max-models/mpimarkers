"""Shared test helpers."""

from __future__ import annotations

import cunumpy as xp
import pytest

from mpimarkers import ParticleStore


class FakeComm:
    """A communicator that only knows its rank and size (no communication)."""

    def __init__(self, rank: int = 0, size: int = 1) -> None:
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size


@pytest.fixture
def fake_comm():
    return FakeComm


def filled_store(count: int, ndim: int = 2, nvel: int = 3, **kwargs) -> ParticleStore:
    """Return a store of ``count`` markers: position i, velocity -i, weight i + 1."""
    store = ParticleStore(ndim, nvel, **kwargs)
    index = xp.arange(count, dtype=float)
    store.add(
        xp.stack([index + 0.5 * d for d in range(ndim)]),
        xp.stack([-index - d for d in range(nvel)]),
        index + 1.0,
    )
    return store
