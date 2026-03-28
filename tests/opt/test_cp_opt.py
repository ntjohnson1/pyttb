# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import logging

import numpy as np
import pytest

import pyttb as ttb
from pyttb.cp_opt import cp_opt, get_initial_guess
from pyttb.opt.fg_setup import FGHandlesOPT, setup_opt
from pyttb.opt.optimizers import LBFGSB

# ---------------------------------------------------------------------------
# Unit tests for get_initial_guess
# ---------------------------------------------------------------------------


def test_initial_guesses():
    rank = 2
    data = ttb.tenones((2, 2, 2))

    M0 = get_initial_guess(data, rank, "random")
    assert M0.full().shape == data.shape
    assert np.all(M0.weights == 1)
    assert all(np.all(fm >= 0.0) and np.all(fm <= 1.0) for fm in M0.factor_matrices)

    M0 = get_initial_guess(data, rank, "random_normal")
    assert M0.full().shape == data.shape
    assert np.all(M0.weights == 1)

    M1 = get_initial_guess(data, rank, M0)
    assert M1.isequal(M0)
    assert M1 is M0

    M1 = get_initial_guess(data, rank, M0.factor_matrices)
    assert M1.isequal(M0)


def test_initial_guess_nvecs():
    """nvecs init should produce a valid ktensor with the right shape."""
    rank = 2
    data = ttb.tenones((3, 4, 2))
    M0 = get_initial_guess(data, rank, "nvecs")
    assert M0.full().shape == data.shape
    assert M0.ncomponents == rank


def test_initial_guess_nonunit_weights_warns(caplog):
    """ktensor init with non-unit weights should log a warning and renormalize."""
    rank = 2
    data = ttb.tenones((3, 4))
    np.random.seed(0)
    M0 = ttb.ktensor.from_function(
        lambda s: np.random.uniform(0.5, 1.5, size=s), data.shape, rank
    )
    M0.weights = np.array([2.0, 3.0])  # non-unit weights

    with caplog.at_level(logging.WARNING):
        M1 = get_initial_guess(data, rank, M0)

    assert "unit weight" in caplog.text.lower()
    assert M1 is M0  # returned same object after in-place normalization


def test_initial_guess_invalid_raises():
    """An unrecognised init string should raise ValueError."""
    rank = 2
    data = ttb.tenones((2, 2))
    with pytest.raises(ValueError, match="Unsupported initialization"):
        get_initial_guess(data, rank, "bad_init")


@pytest.mark.parametrize("init", ["random", "random_normal"])
def test_initial_guess_state_int_reproducible(init):
    """Integer seed should produce identical factor matrices across two calls."""
    rank = 2
    data = ttb.tenones((3, 4, 2))
    M1 = get_initial_guess(data, rank, init, state=42)
    M2 = get_initial_guess(data, rank, init, state=42)
    for fm1, fm2 in zip(M1.factor_matrices, M2.factor_matrices):
        assert np.array_equal(fm1, fm2), f"Factor matrices differ for init={init!r}"


@pytest.mark.parametrize("init", ["random", "random_normal"])
def test_initial_guess_state_generator_reproducible(init):
    """Passing identical Generators should produce identical factor matrices."""
    rank = 2
    data = ttb.tenones((3, 4, 2))
    M1 = get_initial_guess(data, rank, init, state=np.random.default_rng(7))
    M2 = get_initial_guess(data, rank, init, state=np.random.default_rng(7))
    for fm1, fm2 in zip(M1.factor_matrices, M2.factor_matrices):
        assert np.array_equal(fm1, fm2), f"Factor matrices differ for init={init!r}"


@pytest.mark.parametrize("init", ["random", "random_normal"])
def test_initial_guess_different_seeds_differ(init):
    """Different seeds should (almost certainly) produce different factor matrices."""
    rank = 2
    data = ttb.tenones((4, 5, 3))
    M1 = get_initial_guess(data, rank, init, state=0)
    M2 = get_initial_guess(data, rank, init, state=1)
    assert not all(
        np.array_equal(fm1, fm2)
        for fm1, fm2 in zip(M1.factor_matrices, M2.factor_matrices)
    )


@pytest.mark.parametrize("init", ["random", "random_normal"])
def test_initial_guess_state_isolates_from_global_rng(init):
    """The same seed should produce identical results regardless of global RNG state.

    This is the key behavioural guarantee: explicit state makes initialization
    deterministic even when the caller has advanced np.random elsewhere.
    """
    rank = 2
    data = ttb.tenones((3, 4, 2))

    np.random.seed(0)
    M1 = get_initial_guess(data, rank, init, state=42)

    np.random.seed(999)  # advance global state to a completely different position
    M2 = get_initial_guess(data, rank, init, state=42)

    for fm1, fm2 in zip(M1.factor_matrices, M2.factor_matrices):
        assert np.array_equal(fm1, fm2), (
            f"state=42 gave different results when global RNG state differed ({init})"
        )


# ---------------------------------------------------------------------------
# Unit tests for FGHandlesOPT
# ---------------------------------------------------------------------------


class TestFGHandlesOPT:
    """Unit tests for FGHandlesOPT function and gradient handles."""

    def test_known_values(self):
        """Check F and G against manually computed values on a tiny matrix.

        Z = [[1,2],[3,4]], rank-1, A[0]=A[1]=[[1],[1]]
        full(K) = [[1,1],[1,1]], ||Z||^2 = 30, scale=1
        F = (30 - 2*10 + 4) / 1 = 14  (= ||Z-M||^2)
        G[0] = 2*(M-Z) unfolded * A[1] = [[-2],[-10]]
        G[1] = 2*(M-Z)^T unfolded * A[0] = [[-4],[-8]]
        """
        Z_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        A = [np.ones((2, 1)), np.ones((2, 1))]
        M = ttb.ktensor(A)
        data = ttb.tensor(Z_arr)
        Xnormsqr = float(np.sum(Z_arr**2))  # 30
        scale = 1.0
        fgh = FGHandlesOPT(scale, Xnormsqr)

        F = fgh.function_handle(M, data)
        assert np.isclose(F, 14.0), f"Expected F=14, got {F}"

        G = fgh.gradient_handle(M, data)
        assert np.allclose(G[0], [[-2.0], [-10.0]]), f"G[0] wrong: {G[0]}"
        assert np.allclose(G[1], [[-4.0], [-8.0]]), f"G[1] wrong: {G[1]}"

    def test_function_at_minimum(self):
        """When model == data (exact low-rank), F should be near zero."""
        shape = (3, 4, 2)
        rank = 2
        np.random.seed(0)
        M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0, 1, size=s), shape, rank
        )
        data = M.full()
        Xnormsqr = data.norm() ** 2
        scale = Xnormsqr
        fgh = FGHandlesOPT(scale, Xnormsqr)

        F = fgh.function_handle(M, data)
        assert np.isclose(F, 0.0, atol=1e-12), f"Expected F≈0 but got {F}"

    def test_gradient_at_minimum(self):
        """When model == data (exact low-rank), gradient should be near zero."""
        shape = (3, 4, 2)
        rank = 2
        np.random.seed(1)
        M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0, 1, size=s), shape, rank
        )
        data = M.full()
        Xnormsqr = data.norm() ** 2
        scale = Xnormsqr
        fgh = FGHandlesOPT(scale, Xnormsqr)

        fgh.function_handle(M, data)  # function must be called first (caching)
        G = fgh.gradient_handle(M, data)

        assert len(G) == len(shape)
        for k, Gk in enumerate(G):
            assert Gk.shape == (shape[k], rank)
            assert np.allclose(Gk, 0.0, atol=1e-10), f"G[{k}] not near zero: {Gk}"

    def test_finite_difference_gradient(self):
        """Finite difference check: analytic gradient should match numerical."""
        shape = (4, 3, 2)
        rank = 2
        np.random.seed(42)
        M = ttb.ktensor.from_function(
            lambda s: np.random.normal(0, 1, size=s), shape, rank
        )
        np.random.seed(7)
        data = ttb.tensor(np.random.normal(0, 1, shape))
        Xnormsqr = data.norm() ** 2
        scale = Xnormsqr
        eps = 1e-5

        fgh = FGHandlesOPT(scale, Xnormsqr)
        fgh.function_handle(M, data)
        G = fgh.gradient_handle(M, data)

        k, i, j = 0, 1, 0
        original = M.factor_matrices[k][i, j]

        M_plus = M.copy()
        M_plus.factor_matrices[k][i, j] = original + eps
        fgh_plus = FGHandlesOPT(scale, Xnormsqr)
        F_plus = fgh_plus.function_handle(M_plus, data)

        M_minus = M.copy()
        M_minus.factor_matrices[k][i, j] = original - eps
        fgh_minus = FGHandlesOPT(scale, Xnormsqr)
        F_minus = fgh_minus.function_handle(M_minus, data)

        fd_grad = (F_plus - F_minus) / (2 * eps)
        assert np.isclose(G[k][i, j], fd_grad, rtol=1e-4), (
            f"FD gradient {fd_grad:.6e} ≠ analytic gradient {G[k][i, j]:.6e}"
        )

    def test_caching_state(self):
        """Verify the caching counter resets correctly across multiple calls."""
        shape = (3, 3, 3)
        rank = 2
        np.random.seed(5)
        M = ttb.ktensor.from_function(
            lambda s: np.random.normal(0, 1, size=s), shape, rank
        )
        np.random.seed(6)
        data = ttb.tensor(np.random.normal(0, 1, shape))
        Xnormsqr = data.norm() ** 2
        fgh = FGHandlesOPT(Xnormsqr, Xnormsqr)

        for _ in range(2):
            assert fgh._local_iter == 0
            F = fgh.function_handle(M, data)
            assert fgh._local_iter == 1
            G = fgh.gradient_handle(M, data)
            assert fgh._local_iter == 0
            assert F is not None
            assert G is not None

    def test_scale_divides_function_and_gradient(self):
        """Doubling scale should halve F and all gradient entries."""
        shape = (3, 3)
        rank = 1
        np.random.seed(0)
        M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0, 1, size=s), shape, rank
        )
        np.random.seed(1)
        data = ttb.tensor(np.random.uniform(0, 1, shape))
        Xnormsqr = data.norm() ** 2

        fgh1 = FGHandlesOPT(1.0, Xnormsqr)
        F1 = fgh1.function_handle(M, data)
        fgh1_g = FGHandlesOPT(1.0, Xnormsqr)
        fgh1_g.function_handle(M, data)
        G1 = fgh1_g.gradient_handle(M, data)

        fgh2 = FGHandlesOPT(2.0, Xnormsqr)
        F2 = fgh2.function_handle(M, data)
        fgh2_g = FGHandlesOPT(2.0, Xnormsqr)
        fgh2_g.function_handle(M, data)
        G2 = fgh2_g.gradient_handle(M, data)

        assert np.isclose(F1 / F2, 2.0), f"Expected F1/F2=2, got {F1 / F2}"
        assert np.isclose(G1[0][0, 0] / G2[0][0, 0], 2.0)


# ---------------------------------------------------------------------------
# Unit tests for setup()
# ---------------------------------------------------------------------------


def test_setup_opt_returns_handles():
    """setup_opt() should return callable handles and -inf lower bound."""
    fh, gh, lb = setup_opt(5.0, 10.0)
    assert callable(fh)
    assert callable(gh)
    assert lb == -np.inf


def test_fg_setup_smoke():
    rank = 2
    data = ttb.tenones((2, 2, 2))

    M0 = get_initial_guess(data, rank, "random")
    scale = M0.full().norm() ** 2
    fgh = FGHandlesOPT(scale, scale)
    f = fgh.function_handle(M0, M0.full())
    g = fgh.gradient_handle(M0, M0.full())
    assert np.abs(f) < 5 * np.finfo(f.dtype).eps
    assert (np.abs(g_i) < 5 * np.finfo(g_i.dtype).eps for g_i in g)


# ---------------------------------------------------------------------------
# Integration tests for cp_opt
# ---------------------------------------------------------------------------


def test_cp_opt_smoke():
    rank = 2
    data = ttb.tenones((2, 2, 2))
    optimizer = LBFGSB()

    M0 = get_initial_guess(data, rank, "random")
    model, _, info = cp_opt(M0.full(), rank, optimizer, M0)
    assert np.linalg.norm(M0.full().data - model.full().data) < (
        5 * np.finfo(model.full().data.dtype).eps
    )


def test_cp_opt_rank_mismatch_raises():
    """cp_opt should raise ValueError if init has wrong number of components."""
    rank = 2
    data = ttb.tenones((3, 3))
    optimizer = LBFGSB()
    wrong_rank_init = get_initial_guess(data, 3, "random")  # rank 3, not 2
    with pytest.raises(ValueError, match="Initial guess has"):
        cp_opt(data, rank, optimizer, init=wrong_rank_init)


def test_cp_opt_explicit_scale_and_xnormsqr():
    """Explicit scale and Xnormsqr should be accepted and produce a result."""
    rank = 2
    np.random.seed(1)
    data = ttb.tensor(np.random.uniform(0, 1, (3, 3)))
    optimizer = LBFGSB()
    Xnormsqr = data.norm() ** 2
    result, _, _ = cp_opt(data, rank, optimizer, scale=Xnormsqr, Xnormsqr=Xnormsqr)
    assert isinstance(result, ttb.ktensor)
    assert result.ncomponents == rank


def test_cp_opt_xnormsqr_zero():
    """When data is all zeros (Xnormsqr==0), scale defaults to 1 without error."""
    rank = 1
    data = ttb.tensor(np.zeros((2, 2, 2)))
    optimizer = LBFGSB()
    result, _, _ = cp_opt(data, rank, optimizer)
    assert isinstance(result, ttb.ktensor)


def test_cp_opt_nvecs_init():
    """nvecs initialization should converge to a valid solution."""
    rank = 2
    np.random.seed(0)
    true_M = ttb.ktensor.from_function(
        lambda s: np.random.uniform(0.5, 1.5, size=s), (4, 3, 2), rank
    )
    data = true_M.full()
    optimizer = LBFGSB()
    result, M0, _ = cp_opt(data, rank, optimizer, init="nvecs")
    assert isinstance(result, ttb.ktensor)
    assert M0.ncomponents == rank


def test_cp_opt_sptensor():
    """cp_opt should work with a sparse tensor input."""
    rank = 2
    np.random.seed(2)
    dense = ttb.tensor(np.random.uniform(0, 1, (4, 3, 2)))
    sparse = dense.to_sptensor()
    optimizer = LBFGSB()
    result, _, _ = cp_opt(sparse, rank, optimizer)
    assert isinstance(result, ttb.ktensor)
    assert result.ncomponents == rank


def test_cp_opt_state_int_reproducible():
    """cp_opt with the same integer seed should produce the same initial guess."""
    rank = 2
    data = ttb.tenones((3, 4, 2))
    optimizer = LBFGSB()
    _, M0_a, _ = cp_opt(data, rank, optimizer, state=99)
    _, M0_b, _ = cp_opt(data, rank, optimizer, state=99)
    assert M0_a.isequal(M0_b)


def test_cp_opt_state_generator_reproducible():
    """cp_opt with identically seeded Generators should produce the same M0."""
    rank = 2
    data = ttb.tenones((3, 4, 2))
    optimizer = LBFGSB()
    _, M0_a, _ = cp_opt(data, rank, optimizer, state=np.random.default_rng(5))
    _, M0_b, _ = cp_opt(data, rank, optimizer, state=np.random.default_rng(5))
    assert M0_a.isequal(M0_b)
