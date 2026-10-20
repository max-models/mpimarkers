import cunumpy as xp
import numpy as np
import pytest

from mpimarkers import BoxDecomposition, Decomposition
from mpimarkers.tests.conftest import FakeComm


def host(array):
    return np.asarray(xp.to_numpy(array))


def box(rank, proc_sizes, periodic=False, lower=(0.0, 0.0), upper=(1.0, 2.0)):
    size = int(np.prod(proc_sizes))
    return BoxDecomposition(FakeComm(rank, size), lower, upper, proc_sizes, periodic)


def test_protocol():
    assert isinstance(box(0, (1, 1)), Decomposition)


def test_ranks_are_row_major():
    deco = box(0, (2, 3))
    # block (i, j) is rank 3 * i + j; the box is [0, 1] x [0, 2]
    x = xp.asarray(
        [[0.25, 0.25, 0.25, 0.75, 0.75, 0.75], [0.1, 0.9, 1.9, 0.1, 0.9, 1.9]]
    )
    assert host(deco.owners(x)).tolist() == [0, 1, 2, 3, 4, 5]
    assert deco.bounds(4) == ((0.5, 2 / 3), (1.0, 4 / 3))


def test_out_of_range_rule_clips_and_upper_is_last():
    deco = box(0, (2, 2))
    x = xp.asarray([[-5.0, 1.0, 7.0, 0.5, 0.4999], [-1.0, 2.0, 9.0, 1.0, 0.0]])
    # block edges at 0.5 (x) and 1.0 (y): a value on an edge belongs to the upper block
    assert host(deco.owners(x)).tolist() == [0, 3, 3, 3, 0]


def test_periodic_wraps():
    deco = box(0, (2, 1), periodic=(True, False), upper=(1.0, 1.0))
    x = xp.asarray([[-0.25, 1.25, 1.0, 2.75], [0.0, 0.0, 0.0, 0.0]])
    # wrapped: 0.75, 0.25, 0.0 (upper wraps to lower), 0.75
    assert host(deco.owners(x)).tolist() == [1, 0, 0, 1]


@pytest.mark.parametrize("proc_sizes", [(1, 1), (4, 1), (2, 3), (3, 2)])
@pytest.mark.parametrize("periodic", [False, (True, False), True])
def test_owned_mask_agrees_with_owners_on_every_rank(proc_sizes, periodic):
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.5, 2.5, size=(2, 400))
    pts[:, :6] = [[0, 0.5, 1, 0.25, 2 / 3, 1 / 3], [0, 1, 2, 1.0, 0.5, 4 / 3]]
    pts[0, 6] = np.nan
    x = xp.asarray(pts)
    size = int(np.prod(proc_sizes))
    owners = None
    for rank in range(size):
        deco = box(rank, proc_sizes, periodic)
        owners = host(deco.owners(x))
        assert np.array_equal(host(deco.owned_mask(x)), owners == rank)
    assert owners is not None and owners.min() >= 0 and owners.max() < size


def test_every_position_has_exactly_one_owner():
    rng = np.random.default_rng(2)
    x = xp.asarray(rng.uniform(-1, 3, size=(2, 300)))
    total = sum(host(box(r, (2, 2)).owned_mask(x)).astype(int) for r in range(4))
    assert (total == 1).all()


def test_validation():
    comm = FakeComm(0, 4)
    with pytest.raises(ValueError):
        BoxDecomposition(comm, [0, 0], [1, 1], [2, 1])  # 2 blocks, 4 ranks
    with pytest.raises(ValueError):
        BoxDecomposition(comm, [0], [1, 1], [2, 2])
    with pytest.raises(ValueError):
        BoxDecomposition(comm, [0, 0], [0, 1], [2, 2])
    with pytest.raises(ValueError):
        BoxDecomposition(comm, [0, 0], [1, 1], [2, 2], periodic=[True])


def test_default_comm_is_serial_world():
    deco = BoxDecomposition(None, [0.0], [1.0], [1])
    assert (deco.rank, deco.size) == (0, 1)
