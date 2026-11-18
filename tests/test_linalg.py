import numpy as np

from src import linalg


def test_sym_to_vec_preserves_frobenius_inner_product():
    rng = np.random.default_rng(0)
    m = 5
    A = rng.standard_normal((m, m))
    A = A + A.T
    B = rng.standard_normal((m, m))
    B = B + B.T
    va, vb = linalg.sym_to_vec(A), linalg.sym_to_vec(B)
    assert np.isclose(va @ vb, np.trace(A @ B))


def test_vec_to_sym_is_inverse_of_sym_to_vec():
    rng = np.random.default_rng(1)
    m = 4
    A = rng.standard_normal((m, m))
    A = A + A.T
    v = linalg.sym_to_vec(A)
    A_rec = linalg.vec_to_sym(v, m)
    assert np.allclose(A, A_rec)


def test_outer_vec_matches_sym_to_vec_of_outer_product():
    rng = np.random.default_rng(2)
    phi = rng.standard_normal(4)
    v1 = linalg.outer_vec(phi)
    v2 = linalg.sym_to_vec(np.outer(phi, phi))
    assert np.allclose(v1, v2)


def test_basis_change_matrix_is_orthogonal_and_rotates_correctly():
    rng = np.random.default_rng(3)
    m = 4
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    R = linalg.basis_change_matrix(Q)
    assert np.allclose(R @ R.T, np.eye(linalg.vech_dim(m)), atol=1e-8)

    A = rng.standard_normal((m, m))
    A = A + A.T
    rotated = Q.T @ A @ Q
    assert np.allclose(R @ linalg.sym_to_vec(A), linalg.sym_to_vec(rotated), atol=1e-8)


def test_vech_dim():
    assert linalg.vech_dim(4) == 10
    assert linalg.vech_dim(1) == 1
