"""Vectorization of symmetric matrices with a Frobenius-preserving scaling
("scaled vech"), and the basis change induced by an orthogonal eigenbasis.

Custom implementation: a standard linear-algebra device (e.g. Seber, 2008,
"A Matrix Handbook for Statisticians"), specialized here to keep the inner
product <A, B> = trace(A B) of the paper's Hessian and score-covariance
operators (Theorem 4.3, Remark B.1) exactly represented in a flat vector
space of dimension m(m+1)/2.
"""

from __future__ import annotations

import numpy as np

_SQRT2 = np.sqrt(2.0)


def vech_dim(m: int) -> int:
    return m * (m + 1) // 2


def sym_to_vec(A: np.ndarray) -> np.ndarray:
    """Scaled upper-triangular vectorization of a symmetric m x m matrix so
    that the Euclidean inner product of two vecs equals trace(A B)."""
    m = A.shape[-1]
    idx_i, idx_j = np.triu_indices(m)
    out = A[..., idx_i, idx_j].copy()
    off_diag = idx_i != idx_j
    out[..., off_diag] *= _SQRT2
    return out


def vec_to_sym(v: np.ndarray, m: int) -> np.ndarray:
    """Inverse of sym_to_vec."""
    idx_i, idx_j = np.triu_indices(m)
    off_diag = idx_i != idx_j
    A = np.zeros(v.shape[:-1] + (m, m))
    w = v.copy()
    w[..., off_diag] /= _SQRT2
    A[..., idx_i, idx_j] = w
    A[..., idx_j, idx_i] = w
    return A


def outer_vec(phi: np.ndarray) -> np.ndarray:
    """vec(phi phi^T) for a batch of vectors phi of shape (..., m)."""
    C = phi[..., :, None] * phi[..., None, :]
    return sym_to_vec(C)


def basis_change_matrix(Q: np.ndarray) -> np.ndarray:
    """Orthogonal M x M matrix R such that R @ sym_to_vec(A) = sym_to_vec(Q^T A Q)
    for every symmetric A, where M = m(m+1)/2 and Q is an m x m orthogonal matrix.
    """
    m = Q.shape[0]
    M = vech_dim(m)
    R = np.zeros((M, M))
    idx_i, idx_j = np.triu_indices(m)
    for k in range(M):
        E = np.zeros((m, m))
        a, b = idx_i[k], idx_j[k]
        if a == b:
            E[a, b] = 1.0
        else:
            E[a, b] = 1.0 / _SQRT2
            E[b, a] = 1.0 / _SQRT2
        rotated = Q.T @ E @ Q
        R[:, k] = sym_to_vec(rotated)
    return R
