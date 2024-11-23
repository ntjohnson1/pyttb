"""Prepare Function and Gradient Handles for CP OPT"""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

from itertools import chain
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import pyttb as ttb

function_type = Callable[[ttb.ktensor, Union[ttb.tensor, ttb.sptensor]], float]
gradient_type = Callable[
    [ttb.ktensor, Union[ttb.tensor, ttb.sptensor]], List[np.ndarray]
]
fg_return = Tuple[function_type, gradient_type, float]


class FGHandles:
    def __init__(self, scale: float, Xnormsqr: float):
        self._scale = scale
        self._Xnormsqr = Xnormsqr
        self._global_iter: int = 0
        self._local_iter: int = 0
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, List[np.ndarray]]] = None

    def _core(
        self, model: ttb.ktensor, data: Union[ttb.tensor, ttb.sptensor]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        if self._local_iter == 1:
            self._local_iter = 0
            self._global_iter += 1
            assert self._cache is not None
            ret_val = self._cache
            self._cache = None
            return ret_val
        self._local_iter += 1
        Upsilon = []
        for k in range(data.ndims):
            Upsilon.append(
                model.factor_matrices[k].transpose().dot(model.factor_matrices[k])
            )
        # For gradient
        Gamma = []
        for k in range(data.ndims):
            Gamma.append(np.ones((model.ncomponents, model.ncomponents)))
            for ell in chain(range(0, k), range(k + 1, data.ndims)):
                Gamma[-1] *= Upsilon[ell]
        W = Gamma[0] * Upsilon[0]

        U = data.mttkrp(model.factor_matrices, 0)
        self._cache = (U, W, Gamma)
        return U, W, Gamma

    def gradient_handle(
        self, model: ttb.ktensor, data: Union[ttb.tensor, ttb.sptensor]
    ):
        U, _, Gamma = self._core(model, data)
        # Calculate gradient
        G = []
        # FIXME: Since we are setting up the gradient handle we
        # can clean up this loop
        print(
            f"{U=}\n{model.factor_matrices[0]=}\n{Gamma[0]=}\n"
            f"{model.factor_matrices[0].dot(Gamma[0])}"
        )
        G.append(-U + model.factor_matrices[0].dot(Gamma[0]))
        print(f"{G[0]=}")
        for k in range(1, data.ndims):
            U = data.mttkrp(model.factor_matrices, k)
            G.append(-U + model.factor_matrices[k].dot(Gamma[k]))
        G = [factor * (2 / self._scale) for factor in G]
        # G = [factor.flatten() for factor in G]
        return G

    def function_handle(
        self, model: ttb.ktensor, data: Union[ttb.tensor, ttb.sptensor]
    ):
        U, W, _ = self._core(model, data)
        V = model.factor_matrices[0] * U
        F2 = np.sum(V)
        # Calculate F
        # F1 = ||X||^2
        F1 = self._Xnormsqr

        # F3 = ||M||^2
        F3 = np.sum(W)

        F = (F1 - 2 * F2 + F3) / self._scale

        return F


def setup(
    scale: float,
    Xnormsqr: float,
) -> fg_return:
    """Collects the function and gradient handles for GCP

    Parameters
    ----------
    scale:
        Scale the denominator of the optimization problem.
        F(M) = ||X-M||^2 / scale.
    Xnormsqr:
        Norm squared of the data. ||X||^2

    Returns
    -------
        Function handle, gradient handle, and lower bound.
    """
    lower_bound = -np.inf
    fgh = FGHandles(scale, Xnormsqr)
    # TODO this works if we operate on ktensors and (sp)tensors
    #  need to update to work on vector valued quantities or specify this
    #  is the delta from gcp opt
    return fgh.function_handle, fgh.gradient_handle, lower_bound
