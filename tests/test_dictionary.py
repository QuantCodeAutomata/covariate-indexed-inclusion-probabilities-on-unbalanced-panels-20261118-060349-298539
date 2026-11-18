import numpy as np

from src import dictionary


def test_cosine_dictionary_shape_m4():
    phi, size = dictionary.cosine_dictionary(4)
    assert size == 4
    z = np.linspace(-1, 1, 7)
    out = phi(z)
    assert out.shape == (7, 4)
    assert np.all(np.isfinite(out))


def test_cosine_dictionary_m8_includes_m4_frequencies():
    phi4, _ = dictionary.cosine_dictionary(4)
    phi8, size8 = dictionary.cosine_dictionary(8)
    assert size8 == 8
    z = np.array([0.3])
    out4 = phi4(z)
    out8 = phi8(z)
    assert np.allclose(out4, out8[:, :4])


def test_cosine_dictionary_with_constant():
    phi, size = dictionary.cosine_dictionary(4, include_constant=True)
    assert size == 5
    out = phi(np.array([0.0, 0.5]))
    assert np.allclose(out[:, 0], 1.0)


def test_cosine_dictionary_rejects_bad_m():
    import pytest
    with pytest.raises(ValueError):
        dictionary.cosine_dictionary(5)


def test_linear_kernel_dictionary_matches_kernel():
    d = 6
    phi, size = dictionary.linear_kernel_dictionary(d)
    assert size == d + 1
    rng = np.random.default_rng(0)
    z1 = rng.uniform(-1, 1, d)
    z2 = rng.uniform(-1, 1, d)
    inner = phi(z1[None, :])[0] @ phi(z2[None, :])[0]
    expected = 1 + (z1 @ z2) / d
    assert np.isclose(inner, expected)


def test_rank_angle_dictionary_dimension_and_norm():
    d = 5
    phi, size = dictionary.rank_angle_dictionary(d)
    assert size == 2 * d + 1
    rng = np.random.default_rng(1)
    z = rng.uniform(-1, 1, (3, d))
    out = phi(z)
    assert out.shape == (3, size)
    assert np.all(np.isfinite(out))


def test_pivoted_cholesky_landmarks_recovers_kernel_on_landmarks():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (30, 3))
    phi, m, landmarks = dictionary.pivoted_cholesky_landmarks(
        X, lambda a, b: dictionary.rbf_kernel(a, b, 0.5), m=8, rng=rng
    )
    assert landmarks.shape[0] == m
    approx = phi(landmarks) @ phi(landmarks).T
    exact = dictionary.rbf_kernel(landmarks, landmarks, 0.5)
    assert np.allclose(approx, exact, atol=1e-6)
