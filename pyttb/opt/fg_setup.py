"""Prepare Function and Gradient Handles for CP OPT"""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

from functools import partial
from itertools import chain
from typing import Callable, Optional, Tuple, Union

import numpy as np

import pyttb as ttb
from pyttb.gcp.fg_setup import fg_return, function_type
from pyttb.gcp import handles
from pyttb.gcp.handles import Objectives

class FGHandles:
    def __init__(self, scale: float, XnormSqr: float):
        self._scale = scale
        self._XnormSqr = XnormSqr

    def calculate(self, model: ttb.ktensor, data: Union[ttb.tensor, ttb.sptensor]):
        Upsilon = []
        for k in range(data.ndims):
            Upsilon.append(model.factor_matrices[k])
        # For gradient
        Gamma = []
        for k in range(data.ndims):
            Gamma.append(np.ones_like(data, shape=(model.ncomponents, model.ncomponents)))
            for ell in chain(range(0,k), range(k+1, data.ndims)):
                Gamma[-1] *= Upsilon[ell]
        W = Gamma[0] * Upsilon[0]

        U = data.mttkrp(model.factor_matrices,0)
        V = model.factor_matrics[0] * U
        F2 = np.sum(V)

        # Calculate gradient
        G = []
        # FIXME: Since we are setting up the gradient handle we
        # can clean up this loop
        G.append(-U + model.factor_matrics[0]*Gamma[0])
        for k in range(1, data.ndims):
            U = data.mttkrp(model.factor_matrics, k)
            G.append(-U + model.factor_matrics[k]*Gamma[k])
        G = [factor * (2/self._scale) for factor in G]
        G = [factor.flatten() for factor in G]

        # Calculate F
        # F1 = ||X||^2
        F1 = self._XnormSqr

        # F3 = ||M||^2
        F3 = np.sum(W)

        F = (F1 - 2*F2 + F3) / self._scale

        return F, G


def setup(  # noqa: PLR0912,PLR0915
    scale: float,
    XnormSqr: float
) -> fg_return:
    lower_bound = -np.inf
    function_handle = handles.gaussian
    gradient_handle = handles.gaussian_grad
