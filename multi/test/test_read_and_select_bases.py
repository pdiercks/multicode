import pathlib
import numpy as np
from multi.io import read_bases
from multi.misc import select_modes

DATA = pathlib.Path(__file__).parent / "data"


def prepare_test_data(n):
    phi = np.ones((8, n))
    bottom = np.ones((6, n)) * 2
    right = np.ones((7, n)) * 3
    top = np.ones((10, n)) * 4
    left = np.ones((3, n)) * 5
    np.savez(DATA / "basis_0.npz", phi=phi, b=bottom, t=top)
    np.savez(DATA / "basis_1.npz", r=right, l=left)
    np.savez(DATA / "basis_all.npz", phi=phi, b=bottom, r=right, t=top, l=left)


def test_read():
    n = 10
    prepare_test_data(n)
    # FIXME prepare list of (filepath, string) of length 5 in each case
    basis, modes = read_bases(
            [(DATA / "basis_all.npz", "phi"),
(DATA / "basis_all.npz", "b"),
(DATA / "basis_all.npz", "r"),
(DATA / "basis_all.npz", "t"),
(DATA / "basis_all.npz", "l")], return_num_modes=True
    )
    sigma = 8 + 6 + 7 + 10 + 3
    assert basis.shape == (sigma, n)
    assert np.sum(modes) == sigma - 8

    basis, modes = read_bases(
            [(DATA / "basis_all.npz", "phi"),
(DATA / "basis_all.npz", "b"),
(DATA / "basis_all.npz", "r"),
(DATA / "basis_all.npz", "t"),
(DATA / "basis_all.npz", "l")], modes_per_edge=3, return_num_modes=True
    )
    sigma = 8 + 3 * 4
    assert basis.shape == (sigma, n)
    assert np.sum(modes) == sigma - 8

    basis, modes = read_bases([
        (DATA / "basis_0.npz", "phi"),
        (DATA / "basis_0.npz", "b"),
        (DATA / "basis_0.npz", "t"),
        (DATA / "basis_1.npz", "r"),
        (DATA / "basis_1.npz", "l")],
        modes_per_edge=8,
        return_num_modes=True,
    )
    sigma = 8 + 3 + 6 + 7 + 8
    assert basis.shape == (sigma, n)
    assert np.sum(modes) == sigma - 8


def test_select():
    n = 10
    basis, modes = read_bases([
        (DATA / "basis_0.npz", "phi"),
        (DATA / "basis_0.npz", "b"),
        (DATA / "basis_0.npz", "t"),
        (DATA / "basis_1.npz", "r"),
        (DATA / "basis_1.npz", "l")],
        modes_per_edge=10,
        return_num_modes=True,
    )
    sigma = 8 + 3 + 6 + 7 + 10
    assert basis.shape == (sigma, n)
    assert np.sum(modes) == sigma - 8

    B = select_modes(basis, 10, modes)
    assert B.shape == basis.shape
    assert np.allclose(B[np.arange(8)], np.ones((8, n)))
    assert np.allclose(B[np.arange(8, 14)], np.ones((6, n)) * 2)
    assert np.allclose(B[np.arange(14, 21)], np.ones((7, n)) * 3)
    assert np.allclose(B[np.arange(21, 31)], np.ones((10, n)) * 4)
    assert np.allclose(B[np.arange(31, 34)], np.ones((3, n)) * 5)

    B = select_modes(basis, 4, modes)
    assert B.shape == (8 + 4 * 4 - 1, n)
    assert np.allclose(B[np.arange(8)], np.ones((8, n)))
    assert np.allclose(B[np.arange(8, 12)], np.ones((4, n)) * 2)
    assert np.allclose(B[np.arange(12, 16)], np.ones((4, n)) * 3)
    assert np.allclose(B[np.arange(16, 20)], np.ones((4, n)) * 4)
    assert np.allclose(B[np.arange(20, 23)], np.ones((3, n)) * 5)


if __name__ == "__main__":
    test_read()
    test_select()
