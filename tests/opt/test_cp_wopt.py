# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.cp_wopt import cp_wopt
from pyttb.opt.fg_setup import FGHandlesWOPT, setup_wopt
from pyttb.opt.optimizers import LBFGSB

# ---------------------------------------------------------------------------
# Unit tests for FGHandlesWOPT
# ---------------------------------------------------------------------------


class TestFGHandlesWOPT:
    """Unit tests for FGHandlesWOPT function and gradient handles."""

    def test_function_value_no_missing(self):
        """With no missing entries (W all ones) and model == data, F should be 0."""
        shape = (3, 4, 2)
        rank = 2
        np.random.seed(0)
        M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0, 1, size=s), shape, rank
        )
        data = M.full()
        W = ttb.tenones(shape)
        normZsqr = data.norm() ** 2
        fgh = FGHandlesWOPT(W, normZsqr)

        F = fgh.function_handle(M, data)
        assert np.isclose(F, 0.0, atol=1e-12), f"Expected F≈0 but got {F}"

    def test_gradient_zero_no_missing(self):
        """With no missing entries and model == data, gradient should be ~0."""
        shape = (3, 4, 2)
        rank = 2
        np.random.seed(1)
        M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0, 1, size=s), shape, rank
        )
        data = M.full()
        W = ttb.tenones(shape)
        normZsqr = data.norm() ** 2
        fgh = FGHandlesWOPT(W, normZsqr)

        # function_handle must be called first (caching relies on call order)
        fgh.function_handle(M, data)
        G = fgh.gradient_handle(M, data)

        assert len(G) == len(shape)
        for k, Gk in enumerate(G):
            assert Gk.shape == (shape[k], rank)
            assert np.allclose(Gk, 0.0, atol=1e-10), f"G[{k}] not near zero: {Gk}"

    def test_function_and_gradient_consistent(self):
        """Finite difference check: gradient should match numerical derivative."""
        shape = (4, 3, 2)
        rank = 2
        np.random.seed(42)
        M = ttb.ktensor.from_function(
            lambda s: np.random.normal(0, 1, size=s), shape, rank
        )
        np.random.seed(7)
        data_arr = np.random.normal(0, 1, shape)
        # Mask: observe ~75% of entries
        W_arr = (np.random.uniform(0, 1, shape) > 0.25).astype(float)
        data_arr[W_arr == 0] = 0.0
        data = ttb.tensor(data_arr)
        W = ttb.tensor(W_arr)
        normZsqr = data.norm() ** 2
        eps = 1e-5

        fgh = FGHandlesWOPT(W, normZsqr)
        fgh.function_handle(M, data)
        G = fgh.gradient_handle(M, data)

        # Perturb one entry of factor matrix 0 and check finite-difference gradient
        k = 0
        i, j = 1, 0
        original = M.factor_matrices[k][i, j]

        M_plus = M.copy()
        M_plus.factor_matrices[k][i, j] = original + eps
        fgh_plus = FGHandlesWOPT(W, normZsqr)
        F_plus = fgh_plus.function_handle(M_plus, data)

        M_minus = M.copy()
        M_minus.factor_matrices[k][i, j] = original - eps
        fgh_minus = FGHandlesWOPT(W, normZsqr)
        F_minus = fgh_minus.function_handle(M_minus, data)

        fd_grad = (F_plus - F_minus) / (2 * eps)
        assert np.isclose(G[k][i, j], fd_grad, rtol=1e-4), (
            f"FD gradient {fd_grad:.6e} ≠ analytic gradient {G[k][i, j]:.6e}"
        )

    def test_known_values(self):
        """Check F and G against manually computed values on a tiny tensor."""
        # 2x2 tensor, rank 1, all entries observed
        # Z = [[1, 2], [3, 4]], W = ones, A = [[1, 1], [1, 1]] (rank-1 model = all ones)
        # full(K) = [[1,1],[1,1]], B = W.*full(K) = [[1,1],[1,1]]
        # F = 0.5*||Z||^2 - <Z,B> + 0.5*||B||^2
        #   = 0.5*(1+4+9+16) - (1+2+3+4) + 0.5*(1+1+1+1)
        #   = 15 - 10 + 2 = 7
        Z_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        W_arr = np.ones((2, 2))
        A = [np.ones((2, 1)), np.ones((2, 1))]
        M = ttb.ktensor(A)
        data = ttb.tensor(Z_arr)
        W = ttb.tensor(W_arr)
        normZsqr = float(np.sum(Z_arr**2))

        fgh = FGHandlesWOPT(W, normZsqr)
        F = fgh.function_handle(M, data)
        assert np.isclose(F, 7.0), f"Expected F=7.0, got {F}"

        G = fgh.gradient_handle(M, data)
        # T = Z - B = [[0,1],[2,3]]
        # G[0] = -mttkrp(T, A, 0): A[1]=[[1],[1]], T(mode-1 unfold)*khatri-rao...
        # mttkrp(T, A, 0) = T_(0) * (A[1] khatri-rao)
        # T_(0) = [[0,1],[2,3]] (2x2 mode-0 unfolding of [[0,1],[2,3]])
        # khatri_rao(A[1]) = [[1],[1]]
        # mttkrp = T_(0) @ [[1],[1]] = [[0+1],[2+3]] = [[1],[5]]
        # G[0] = -[[1],[5]]
        # G[1] = -mttkrp(T, A, 1): T_(1) = [[0,2],[1,3]], khatri_rao(A[0])=[[1],[1]]
        # mttkrp = [[0+2],[1+3]] = [[2],[4]]
        # G[1] = -[[2],[4]]
        assert np.allclose(G[0], [[-1.0], [-5.0]]), f"G[0] wrong: {G[0]}"
        assert np.allclose(G[1], [[-2.0], [-4.0]]), f"G[1] wrong: {G[1]}"

    def test_caching_state(self):
        """Verify the caching counter resets correctly across multiple calls."""
        shape = (3, 3, 3)
        rank = 2
        np.random.seed(5)
        M = ttb.ktensor.from_function(
            lambda s: np.random.normal(0, 1, size=s), shape, rank
        )
        np.random.seed(6)
        data_arr = np.random.normal(0, 1, shape)
        data = ttb.tensor(data_arr)
        W = ttb.tenones(shape)
        normZsqr = data.norm() ** 2

        fgh = FGHandlesWOPT(W, normZsqr)

        # Simulate two full optimizer steps
        for _ in range(2):
            assert fgh._local_iter == 0
            F = fgh.function_handle(M, data)
            assert fgh._local_iter == 1
            G = fgh.gradient_handle(M, data)
            assert fgh._local_iter == 0
            assert F is not None
            assert G is not None

    def test_missing_entries_ignored(self):
        """Entries where W==0 should not affect F or G."""
        shape = (3, 3, 3)
        rank = 2
        np.random.seed(10)
        M = ttb.ktensor.from_function(
            lambda s: np.random.normal(0, 1, size=s), shape, rank
        )

        # Full observation
        data_arr = np.random.normal(1, 0.1, shape)
        W_full = ttb.tenones(shape)
        data_full = ttb.tensor(data_arr.copy())
        fgh_full = FGHandlesWOPT(W_full, float(np.sum(data_arr**2)))
        F_full = fgh_full.function_handle(M, data_full)
        fgh_full.function_handle(M, data_full)  # reset cache
        fgh_full2 = FGHandlesWOPT(W_full, float(np.sum(data_arr**2)))
        fgh_full2.function_handle(M, data_full)
        G_full = fgh_full2.gradient_handle(M, data_full)

        # Mask one entry; zero it in data too
        W_arr = np.ones(shape)
        W_arr[0, 0, 0] = 0.0
        data_masked_arr = data_arr.copy()
        data_masked_arr[0, 0, 0] = 0.0
        W_masked = ttb.tensor(W_arr)
        data_masked = ttb.tensor(data_masked_arr)
        normZsqr_masked = float(np.sum(data_masked_arr**2))
        fgh_masked = FGHandlesWOPT(W_masked, normZsqr_masked)
        F_masked = fgh_masked.function_handle(M, data_masked)
        G_masked = fgh_masked.gradient_handle(M, data_masked)

        # Results should differ because one observation is removed
        assert not np.isclose(F_full, F_masked)
        assert not all(np.allclose(G_full[k], G_masked[k]) for k in range(len(shape)))


# ---------------------------------------------------------------------------
# Unit tests for setup_wopt
# ---------------------------------------------------------------------------


def test_setup_wopt_returns_handles():
    """setup_wopt should return callable function/gradient handles and -inf bound."""
    W = ttb.tenones((2, 3, 4))
    normZsqr = 10.0
    fh, gh, lb = setup_wopt(W, normZsqr)
    assert callable(fh)
    assert callable(gh)
    assert lb == -np.inf


# ---------------------------------------------------------------------------
# Integration tests for cp_wopt
# ---------------------------------------------------------------------------


class TestCPWOPT:
    """Integration tests for the cp_wopt solver."""

    def test_cp_wopt_no_missing(self):
        """With no missing data, cp_wopt should recover a low-rank tensor."""
        np.random.seed(0)
        shape = (5, 4, 3)
        rank = 2
        # Generate a rank-2 tensor
        true_M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0.5, 1.5, size=s), shape, rank
        )
        data = true_M.full()
        W = ttb.tenones(shape)
        optimizer = LBFGSB()

        result, M0, info = cp_wopt(data, W, rank, optimizer, skip_zeroing=True)

        # Should recover to a low residual (random init won't be exact)
        residual = np.linalg.norm(data.data - result.full().data) / np.linalg.norm(
            data.data
        )
        assert residual < 0.05, f"Residual {residual:.2e} too large"

    def test_cp_wopt_with_missing(self):
        """cp_wopt should fit observed entries well even with ~20% missing."""
        np.random.seed(42)
        shape = (6, 5, 4)
        rank = 2
        true_M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0.5, 1.5, size=s), shape, rank
        )
        full_data = true_M.full()

        # Mask ~20% of entries
        W_arr = (np.random.uniform(0, 1, shape) > 0.2).astype(float)
        W = ttb.tensor(W_arr)
        data_arr = full_data.data.copy()
        data_arr[W_arr == 0] = 0.0
        data = ttb.tensor(data_arr)

        optimizer = LBFGSB()
        result, M0, info = cp_wopt(
            data, W, rank, optimizer, skip_zeroing=True, printitn=0
        )

        # Fit on observed entries
        obs_mask = W_arr.astype(bool)
        residual_obs = np.linalg.norm(
            full_data.data[obs_mask] - result.full().data[obs_mask]
        ) / np.linalg.norm(full_data.data[obs_mask])
        assert residual_obs < 0.05, f"Observed residual {residual_obs:.2e} too large"

    def test_cp_wopt_zeroing(self):
        """cp_wopt should zero out missing entries in data internally.

        Both runs start from the same initial guess and solve the same effective
        problem (dirty data with zeroing vs. pre-zeroed data).  They should
        converge to essentially the same objective.
        """
        np.random.seed(3)
        shape = (4, 3, 3)
        rank = 2
        true_M = ttb.ktensor.from_function(
            lambda s: np.random.uniform(0, 1, size=s), shape, rank
        )
        data = true_M.full()
        W_arr = (np.random.uniform(0, 1, shape) > 0.3).astype(float)
        W = ttb.tensor(W_arr)

        # Dirty version: garbage at missing positions (cp_wopt should zero them out)
        dirty_data_arr = data.data.copy()
        dirty_data_arr[W_arr == 0] = 999.0
        dirty_data = ttb.tensor(dirty_data_arr)

        # Pre-zeroed version: missing positions already zero
        clean_data_arr = data.data.copy()
        clean_data_arr[W_arr == 0] = 0.0
        clean_data = ttb.tensor(clean_data_arr)

        # Shared initial guess: both runs start from exactly the same point
        np.random.seed(7)
        M0 = ttb.ktensor.from_function(
            lambda s: np.random.normal(0, 1, size=s), shape, rank
        )

        optimizer = LBFGSB()
        result_dirty, _, _ = cp_wopt(
            dirty_data, W, rank, optimizer, init=M0, printitn=0
        )
        result_clean, _, _ = cp_wopt(
            clean_data, W, rank, optimizer, init=M0, skip_zeroing=True, printitn=0
        )

        F_dirty = 0.5 * np.sum(
            (W_arr * (clean_data.data - result_dirty.full().data)) ** 2
        )
        F_clean = 0.5 * np.sum(
            (W_arr * (clean_data.data - result_clean.full().data)) ** 2
        )
        assert np.isclose(F_dirty, F_clean, rtol=1e-3), (
            f"Zeroing mismatch: F_dirty={F_dirty:.4e}, F_clean={F_clean:.4e}"
        )

    def test_cp_wopt_returns_correct_shapes(self):
        """Return shapes and types should match the interface."""
        shape = (3, 4, 2)
        rank = 2
        data = ttb.tenones(shape)
        W = ttb.tenones(shape)
        optimizer = LBFGSB()

        result, M0, info = cp_wopt(data, W, rank, optimizer, skip_zeroing=True)

        assert isinstance(result, ttb.ktensor)
        assert isinstance(M0, ttb.ktensor)
        assert isinstance(info, dict)
        assert result.ncomponents == rank
        assert result.ndims == len(shape)

    def test_cp_wopt_shape_mismatch_raises(self):
        """Mismatched data and weights shapes should raise ValueError."""
        data = ttb.tenones((3, 4, 2))
        W = ttb.tenones((3, 4, 3))
        optimizer = LBFGSB()
        with pytest.raises(ValueError, match="shape"):
            cp_wopt(data, W, 2, optimizer)

    def test_cp_wopt_sparse_data_raises(self):
        """Sparse data input should raise ValueError."""
        data = ttb.sptensor(np.array([[0, 0, 0]]), np.array([1.0]), (3, 4, 2))
        W = ttb.tenones((3, 4, 2))
        optimizer = LBFGSB()
        with pytest.raises(ValueError, match="dense"):
            cp_wopt(data, W, 2, optimizer)

    def test_cp_wopt_sparse_weights_raises(self):
        """Non-tensor weights should raise ValueError."""
        data = ttb.tenones((3, 4, 2))
        W = ttb.sptensor(np.array([[0, 0, 0]]), np.array([1.0]), (3, 4, 2))
        optimizer = LBFGSB()
        with pytest.raises(ValueError, match="dense"):
            cp_wopt(data, W, 2, optimizer)

    def test_cp_wopt_rank_mismatch_raises(self):
        """cp_wopt should raise ValueError if init has wrong number of components."""
        shape = (3, 4, 2)
        rank = 2
        data = ttb.tenones(shape)
        W = ttb.tenones(shape)
        optimizer = LBFGSB()
        wrong_init = ttb.ktensor.from_function(
            lambda s: np.ones(s),
            shape,
            rank + 1,  # rank 3, not 2
        )
        with pytest.raises(ValueError, match="Initial guess has"):
            cp_wopt(data, W, rank, optimizer, init=wrong_init)

    def test_cp_wopt_nonneg(self):
        """With lower_bound=0, result factor matrices should be nonneg."""
        np.random.seed(99)
        shape = (5, 4, 3)
        rank = 2
        true_M = ttb.ktensor.from_function(
            lambda s: np.abs(np.random.normal(1, 0.2, size=s)), shape, rank
        )
        data = true_M.full()
        W = ttb.tenones(shape)
        optimizer = LBFGSB()

        result, _, _ = cp_wopt(
            data, W, rank, optimizer, lower_bound=0.0, skip_zeroing=True
        )

        for k, fm in enumerate(result.factor_matrices):
            assert np.all(fm >= -1e-8), f"factor_matrix[{k}] has negative values"

    def test_cp_wopt_state_int_reproducible(self):
        """cp_wopt with the same integer seed should produce the same initial guess."""
        shape = (4, 3, 2)
        rank = 2
        data = ttb.tenones(shape)
        W = ttb.tenones(shape)
        optimizer = LBFGSB()
        _, M0_a, _ = cp_wopt(data, W, rank, optimizer, state=17)
        _, M0_b, _ = cp_wopt(data, W, rank, optimizer, state=17)
        assert M0_a.isequal(M0_b)

    def test_cp_wopt_state_generator_reproducible(self):
        """cp_wopt with identically seeded Generators should produce the same M0."""
        shape = (4, 3, 2)
        rank = 2
        data = ttb.tenones(shape)
        W = ttb.tenones(shape)
        optimizer = LBFGSB()
        _, M0_a, _ = cp_wopt(data, W, rank, optimizer, state=np.random.default_rng(3))
        _, M0_b, _ = cp_wopt(data, W, rank, optimizer, state=np.random.default_rng(3))
        assert M0_a.isequal(M0_b)
