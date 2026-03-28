"""Fit a CP decomposition via optimization."""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

import pyttb as ttb
from pyttb.opt.fg_setup import setup_opt

if TYPE_CHECKING:
    from pyttb.opt.optimizers import LBFGSB


def cp_opt(  # noqa: PLR0913
    data: ttb.tensor | ttb.sptensor,
    rank: int,
    optimizer: LBFGSB,
    init: Literal["random"]
    | Literal["random_normal"]
    | Literal["nvecs"]
    | ttb.ktensor
    | Sequence[np.ndarray] = "random_normal",
    state: np.random.Generator | int | None = None,
    scale: float | None = None,
    Xnormsqr: float | None = None,
    printitn: int = 1,
) -> tuple[ttb.ktensor, ttb.ktensor, dict]:
    """Fits a CP decomposition with user-specified optimizer.

    The objective being optimized is F(M) = || X - M ||^2 / || X ||^2.

    Parameters
    ----------
    data:
        Tensor to decompose.
    rank:
        Rank of desired CP decomposition.
    optimizer:
        Optimizer class for solving the decomposition problem defined.
    init:
        Initial solution to the problem.
    state:
        Random number generator or integer seed for reproducible random
        initialization. Accepts a :class:`numpy.random.Generator` or an
        integer seed (passed to :func:`numpy.random.default_rng`). When
        ``None`` (default), the global :mod:`numpy.random` state is used.
        Ignored when ``init`` is not ``"random"`` or ``"random_normal"``.
    scale:
        Scale the denominator of the optimization problem.
        F(M) = ||X-M||^2 / scale. If converging prematurely try setting the scale to
        S = ||X||^2 / C is less than O(1e10).
    printitn:
        Controls verbosity of printing throughout the solve.

    Returns
    -------
        Solution, Initial Guess, Dictionary of meta data
    """
    M0 = get_initial_guess(data, rank, init, state)
    if M0.ncomponents != rank:
        raise ValueError(f"Initial guess has {M0.ncomponents} but expected {rank}")

    if Xnormsqr is None:
        Xnormsqr = data.norm() ** 2

    if scale is None:
        scale = 1
        if Xnormsqr > 0.0:
            scale = Xnormsqr

    # Optimization stage
    if printitn > 0:
        logging.info("\nCP-OPT Direct Optimization")
    function_handle, gradient_handle, lower_bound = setup_opt(scale, Xnormsqr)
    result, info = optimizer.solve(
        M0,
        data,
        function_handle,
        gradient_handle,
        lower_bound,
    )
    result.arrange()
    result = result.fixsigns()
    return result, M0, info


def _resolve_rng(
    state: np.random.Generator | int | None,
) -> np.random.Generator:
    """Return a numpy random interface from *state*.

    * ``None``  → the legacy ``np.random`` module (preserves global-seed behaviour)
    * ``int``   → ``np.random.default_rng(state)``
    * Generator → returned unchanged
    """
    if state is None:
        return np.random  # type: ignore[return-value]
    if isinstance(state, int):
        return np.random.default_rng(state)
    return state


def get_initial_guess(
    data: ttb.tensor | ttb.sptensor,
    rank: int,
    init: Literal["random"]
    | Literal["random_normal"]
    | Literal["nvecs"]
    | ttb.ktensor
    | Sequence[np.ndarray] = "random_normal",
    state: np.random.Generator | int | None = None,
) -> ttb.ktensor:
    """Get initial guess for cp_opt.

    Parameters
    ----------
    data:
        Tensor whose shape determines the factor matrix sizes.
    rank:
        Number of components.
    init:
        Initialization strategy. See :func:`cp_opt` for details.
    state:
        Random number generator or integer seed. See :func:`cp_opt` for details.

    Returns
    -------
        Normalized ktensor.
    """
    if isinstance(init, Sequence) and not isinstance(init, str):
        return ttb.ktensor(list(init))
    if isinstance(init, ttb.ktensor):
        if not np.all(init.weights == 1):
            # cp_opt normalizes column-wise (matching MATLAB's normalize(M0,1))
            # rather than jointly across all modes as in gcp_opt normalize("all")
            logging.warning("Initial guess doesn't have unit weights; renormalizing")
            init.normalize(1)
        return init
    if init == "nvecs":
        U0 = []
        for k in range(data.ndims):
            U0.append(data.nvecs(k, rank))
        return ttb.ktensor(U0, copy=False)
    rng = _resolve_rng(state)
    if init == "random":
        return ttb.ktensor.from_function(
            lambda s: rng.uniform(0, 1, size=s), data.shape, rank
        )
    if init == "random_normal":
        return ttb.ktensor.from_function(
            lambda s: rng.normal(0, 1, size=s), data.shape, rank
        )
    raise ValueError(f"Unsupported initialization type {init}")
