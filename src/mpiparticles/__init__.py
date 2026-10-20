"""Distributed marker (particle) container for MPI codes.

A :class:`ParticleStore` holds the markers of one rank in component-major
arrays (CPU or GPU through ``cunumpy``) with stable int64 ids; a
:class:`Decomposition` says which rank owns which position; :func:`migrate`
moves the markers that left a rank's subdomain to their owners. Without an MPI
launcher everything runs serially, with no MPI calls.

Example
-------
>>> import cunumpy as xp
>>> from mpiparticles import BoxDecomposition, ParticleStore, migrate
>>> store = ParticleStore(ndim=1, nvel=3)
>>> _ = store.add(xp.asarray([[0.25, 0.75]]), xp.zeros((3, 2)))
>>> deco = BoxDecomposition(None, [0.0], [1.0], [1])
>>> migrate(store, deco)
(0, 0)
"""

from importlib.metadata import PackageNotFoundError, version

from mpiparticles.collective import gather_columns, global_count
from mpiparticles.decomposition import BoxDecomposition, Decomposition
from mpiparticles.exchange import keep_owned, migrate
from mpiparticles.ids import IdAllocator
from mpiparticles.store import MarkerView, ParticleStore, StoreSnapshot

try:
    __version__ = version("mpiparticles")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BoxDecomposition",
    "Decomposition",
    "IdAllocator",
    "MarkerView",
    "ParticleStore",
    "StoreSnapshot",
    "gather_columns",
    "global_count",
    "keep_owned",
    "migrate",
]
