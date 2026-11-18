"""Cosine dictionary of Section 6.1: phi_k(z) = cos(omega_k * z + varsigma_k).

Custom implementation (paper Section 6.1 / 6.2): this finite trigonometric
dictionary is defined directly by the paper for the simulation study, so no
general-purpose feature-map library applies.
"""

from __future__ import annotations

import numpy as np

from src import config


def cosine_dictionary(m: int, include_constant: bool = False):
    """Build the cosine dictionary phi(z) in R^m (paper Section 6.1 / 6.2).

    Parameters
    ----------
    m : int
        Number of cosine functions, must be 4 or 8.
    include_constant : bool
        If True, prepend a constant function 1 to the dictionary (used in
        Sections 6.4 and Appendix C.1 as {1, phi_1, ..., phi_4}).

    Returns
    -------
    phi : callable
        phi(z) -> ndarray of shape (..., dict_size) evaluated elementwise on z.
    dict_size : int
        Total dictionary size (m, or m+1 with the constant).
    """
    if m == 4:
        omega = config.OMEGA_M4
        varsigma = config.VARSIGMA_M4
    elif m == 8:
        omega = np.concatenate([config.OMEGA_M4, config.OMEGA_M8_EXTRA])
        varsigma = np.concatenate([config.VARSIGMA_M4, config.VARSIGMA_M8_EXTRA])
    else:
        raise ValueError("m must be 4 or 8 for the cosine dictionary of Section 6.1/6.2")

    def phi(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z = z[..., None]
        features = np.cos(omega * z + varsigma)
        if include_constant:
            ones = np.ones(z.shape)
            features = np.concatenate([ones, features], axis=-1)
        return features

    dict_size = m + 1 if include_constant else m
    return phi, dict_size


def linear_kernel_dictionary(d: int):
    """phi(z) = (1, z_1, ..., z_d), the exact feature map of the linear kernel
    k(z, z') = 1 + <z, z'>/d (Section 7.1, dictionary 'lin'; Appendix C.3,
    linear-kernel grid m=d+1)."""
    def phi(z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        ones = np.ones((z.shape[0], 1))
        out = np.concatenate([ones, z / np.sqrt(d)], axis=1)
        return out if z.ndim > 1 else out[0]
    return phi, d + 1


def rank_angle_dictionary(d: int, half_angle: bool = False):
    """Rank-angle dictionary 'arc' (half_angle=False) or 'half' (True) of
    Section 7.1: theta_j = arccos(z_j) in [0, pi], features
    1, z_j/sqrt(d), sqrt(1-z_j^2)/sqrt(d) for 'arc' (cos/sin theta_j split),
    or cos((theta_j-theta_j')/2) style half-angle features for 'half'.
    m = 2d + 1 in both cases (implementation notes of exp_4).
    """
    def phi(z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        z = np.clip(z, -1.0, 1.0)
        theta = np.arccos(z)
        ones = np.ones((z.shape[0], 1))
        if not half_angle:
            cos_part = z / np.sqrt(d)
            sin_part = np.sqrt(np.clip(1 - z ** 2, 0.0, None)) / np.sqrt(d)
        else:
            cos_part = np.cos(theta / 2) / np.sqrt(d)
            sin_part = np.sin(theta / 2) / np.sqrt(d)
        out = np.concatenate([ones, cos_part, sin_part], axis=1)
        return out
    return phi, 2 * d + 1


def pivoted_cholesky_landmarks(X: np.ndarray, kernel_fn, m: int, rng: np.random.Generator | None = None):
    """Select m pivoted-Cholesky landmark rows of X for a Nystrom feature map
    (Filipovic and Schneider's pivoted Cholesky section selection, referenced
    in Section 7.1's grid-dictionary construction).

    Custom implementation (paper Section 7.1): greedy pivoted Cholesky /
    Nystrom landmark selection, the standard construction for low-rank kernel
    feature maps (Harbrecht, Peters and Schneider, 2012); no surveyed library
    (scikit-learn's Nystroem samples landmarks uniformly at random rather than
    by pivoted-Cholesky residual maximization) implements this exact selection
    rule.

    Returns the Nystrom feature map phi(z) = L^{-1} k(landmarks, z), with
    dictionary size m, such that phi(z) phi(z')^T approximates kernel_fn(z, z').
    """
    n = X.shape[0]
    K_full = kernel_fn(X, X)
    residual = K_full.copy()
    pivots: list[int] = []
    for k in range(m):
        residual_diag = np.clip(np.diag(residual), 0.0, None)
        i = int(np.argmax(residual_diag))
        piv = residual_diag[i]
        if piv < 1e-14:
            break
        pivots.append(i)
        col = residual[:, i] / np.sqrt(piv)
        residual -= np.outer(col, col)

    landmarks = X[pivots]
    m = len(pivots)
    K_ll = kernel_fn(landmarks, landmarks)
    K_ll = 0.5 * (K_ll + K_ll.T) + 1e-10 * np.eye(m)
    L = np.linalg.cholesky(K_ll)
    L_inv = np.linalg.inv(L)

    def phi(z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        K_zl = kernel_fn(z, landmarks)
        return K_zl @ L_inv.T

    return phi, m, landmarks


def rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-sq / (2 * bandwidth ** 2))
