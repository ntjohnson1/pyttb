"""Fit a weighted CP decomposition via optimization (for missing data)."""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

import pyttb as ttb
from pyttb.cp_opt import get_initial_guess
from pyttb.opt.fg_setup import setup_wopt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyttb.opt.optimizers import LBFGSB


def cp_wopt(  # noqa: PLR0913
    data: ttb.tensor,
    weights: ttb.tensor,
    rank: int,
    optimizer: LBFGSB,
    init: Literal["random"]
    | Literal["random_normal"]
    | Literal["nvecs"]
    | ttb.ktensor
    | Sequence[np.ndarray] = "random_normal",
    state: np.random.Generator | int | None = None,
    lower_bound: float = -np.inf,
    skip_zeroing: bool = False,
    printitn: int = 1,
) -> tuple[ttb.ktensor, ttb.ktensor, dict]:
    """Fits a weighted CP decomposition to handle missing data.

    The objective being optimized is F(K) = 0.5 * || W .* (Z - K) ||^2
    where W is a binary indicator tensor (0=missing, 1=observed).

    Parameters
    ----------
    data:
        Tensor Z to decompose. Missing entries should be zero (or will be
        zeroed by this function unless ``skip_zeroing=True``).
    weights:
        Binary indicator tensor W. Entry is 0 for missing, 1 for observed.
    rank:
        Rank of desired CP decomposition.
    optimizer:
        Optimizer class for solving the decomposition problem.
    init:
        Initial solution to the problem.
    state:
        Random number generator or integer seed for reproducible random
        initialization. See :func:`cp_opt` for details.
    lower_bound:
        Lower bound on factor matrix entries (e.g., 0.0 for nonnegative).
    skip_zeroing:
        If True, skip zeroing out missing entries in ``data``. Set this
        when the missing entries are already zero (avoids the copy).
    printitn:
        Controls verbosity of printing throughout the solve.

    Returns
    -------
        Solution, Initial Guess, Dictionary of meta data

    References
    ----------
    E. Acar, D. M. Dunlavy, T. G. Kolda and M. Morup, Scalable Tensor
    Factorizations for Incomplete Data, Chemometrics and Intelligent
    Laboratory Systems, 106(1):41-56, 2011.
    """
    if not isinstance(data, ttb.tensor):
        raise ValueError("data must be a dense tensor.")
    if not isinstance(weights, ttb.tensor):
        raise ValueError("weights must be a dense tensor.")
    if data.shape != weights.shape:
        raise ValueError(
            f"data shape {data.shape} must match weights shape {weights.shape}."
        )

    # Zero out missing entries so they don't contribute to the objective
    if not skip_zeroing:
        data = data.copy()
        data.data[weights.data == 0] = 0.0

    M0 = get_initial_guess(data, rank, init, state)
    if M0.ncomponents != rank:
        raise ValueError(
            f"Initial guess has {M0.ncomponents} components but expected {rank}"
        )

    normZsqr = data.norm() ** 2

    if printitn > 0:
        logging.info("\nCP-WOPT Weighted CP Optimization")

    function_handle, gradient_handle, lb = setup_wopt(weights, normZsqr)
    # Allow caller to tighten the bound (e.g., nonneg factorization)
    lb = max(lb, lower_bound)

    result, info = optimizer.solve(M0, data, function_handle, gradient_handle, lb)
    result.arrange()
    result = result.fixsigns()
    return result, M0, info
