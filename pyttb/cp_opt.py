"""Fit a CP decomposition via optimization"""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import logging
import time
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

import pyttb as ttb
from pyttb.gcp.optimizers import LBFGSB, StochasticSolver


def cp_opt(
    data: Union[ttb.tensor, ttb.sptensor],
    rank: int,
    optimizer: Union[StochasticSolver, LBFGSB],
    init: Union[
        Literal["random"],
        Literal["random_normal"],
        Literal["nvecs"],
        ttb.ktensor,
        List[np.ndarray],
    ] = "random_normal",
    state=None,
    scale: Optional[float] = None,
    Xnormsqr: Optional[float] = None,
    printitn: int = 1,
) -> Tuple[ttb.ktensor, ttb.ktensor, Dict]:
    """Fits a CP decomposition with user-specified optimizer.
    The objective being optimized is F(M) = || X - M ||^2 / || X ||^2

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
        Random generator to control reproducibility.
    scale:
        Scale the denominator of the optimization problem.
        F(M) = ||X-M||^2 / scale. If converging prematurely try setting the scale to
        S = ||X||^2 / C is less than O(1e10).

    printitn:
        Controls verbosity of printing throughout the solve

    Returns
    -------
        Solution, Initial Guess, Dictionary of meta data
    """
    start = time.monotonic()
    # Skip to line 93 in cp_opt.m
    shape = data.shape
    ndims = data.ndims
    M0 = _get_initial_guess(data, rank, init)
    if M0.ncomponents != rank:
        raise ValueError(f"Initial guess has {M0.ncomponents} but expected {rank}")

    if Xnormsqr is None:
        Xnormsqr = data.norm()**2

    if scale is None:
        scale = 1
        if Xnormsqr > 0.0:
            scale = Xnormsqr

    setup_time = time.monotonic() - start

    # Optimization stage
    start = time.monotonic()
    if printitn > 0:
        logging.info("\nCP-OPT Direct Optimization")
    function_handle, gradient_handle = setup()


def _get_initial_guess(
    data: Union[ttb.tensor, ttb.sptensor],
    rank: int,
    init: Union[
        Literal["random"],
        Literal["random_normal"],
        Literal["nvecs"],
        ttb.ktensor,
        List[np.ndarray],
    ] = "random_normal",
) -> ttb.ktensor:
    """Get initial guess for cp_opt

    Returns
    -------
        Normalized ktensor.
    """
    # TODO might be nice to merge with gcp_opt
    if isinstance(init, list):
        return ttb.ktensor(init)
    if isinstance(init, ttb.ktensor):
        if not np.all(init.weights == 1):
            # FIXME: This doesn't match gcp_opt normalization
            logging.warning("Initial guess doesn't have unit weights; renormalizing")
            init.normalize(1)
        return init
    if init == "nvecs":
        U0 = []
        for k in range(data.ndims):
            U0.append(data.nvecs(k, rank))
        return ttb.ktensor(U0, copy=False)
    if init == "random":
        # TODO tie into shared generator/seed
        rand = partial(np.random.uniform, 0, 1)
        return ttb.ktensor.from_function(rand, data.shape, rank)
    if init == "random_normal":
        randn = partial(np.random.normal, 0, 1)
        return ttb.ktensor.from_function(randn, data.shape, rank)
    raise ValueError(f"Unsupported initialization type {init}")
