"""Prepare Function and Gradient Handles for CP OPT."""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
from __future__ import annotations

import abc
from abc import ABC
from collections.abc import Callable
from itertools import chain

import numpy as np

import pyttb as ttb
from pyttb.ktensor import ktensor
from pyttb.sptensor import sptensor
from pyttb.tensor import tensor

function_type = Callable[[ktensor, tensor | sptensor], float]
gradient_type = Callable[[ktensor, tensor | sptensor], list[np.ndarray]]
fg_return = tuple[function_type, gradient_type, float]


class FGHandlesBase(ABC):
    """Base class to support the various OPT function and gradient definitions."""

    @abc.abstractmethod
    def gradient_handle(self, model: ttb.ktensor, data: ttb.tensor | ttb.sptensor):
        """Calculate the gradient value.

        Parameters
        ----------
        model:
            Current decomposition.
        data:
            Source tensor to decompose.
        """

    @abc.abstractmethod
    def function_handle(self, model: ttb.ktensor, data: ttb.tensor | ttb.sptensor):
        """Calculate the function value.

        Parameters
        ----------
        model:
            Current decomposition.
        data:
            Source tensor to decompose.
        """


class FGHandlesOPT(FGHandlesBase):
    """Function and gradient handles for CP OPT."""

    def __init__(self, scale: float, Xnormsqr: float):
        """Prepare function and gradient handles.

        Parameters
        ----------
        scale:
            Scale the denominator of the optimization problem.
            F(M) = ||model-data||^2 / scale.
        Xnormsqr:
            Norm squared of the data. ||data||^2
        """
        self._scale = scale
        self._Xnormsqr = Xnormsqr
        self._local_iter: int = 0
        self._cache: tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None = None

    def _core(
        self, model: ttb.ktensor, data: ttb.tensor | ttb.sptensor
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        if self._local_iter == 1:
            self._local_iter = 0
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
            for ell in chain(range(k), range(k + 1, data.ndims)):
                Gamma[-1] *= Upsilon[ell]
        W = Gamma[0] * Upsilon[0]

        U = data.mttkrp(model.factor_matrices, 0)
        self._cache = (U, W, Gamma)
        return U, W, Gamma

    def gradient_handle(self, model: ttb.ktensor, data: ttb.tensor | ttb.sptensor):
        """Calculate the gradient value.

        Parameters
        ----------
        model:
            Current decomposition.
        data:
            Source tensor to decompose.
        """
        U, _, Gamma = self._core(model, data)
        # Calculate gradient
        G = []
        G.append(-U + model.factor_matrices[0].dot(Gamma[0]))
        for k in range(1, data.ndims):
            U = data.mttkrp(model.factor_matrices, k)
            G.append(-U + model.factor_matrices[k].dot(Gamma[k]))
        G = [factor * (2 / self._scale) for factor in G]
        return G

    def function_handle(self, model: ttb.ktensor, data: ttb.tensor | ttb.sptensor):
        """Calculate the function value.

        Parameters
        ----------
        model:
            Current decomposition.
        data:
            Source tensor to decompose.
        """
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


class FGHandlesWOPT(FGHandlesBase):
    """Function and gradient handles for CP WOPT.

    Optimizes F(K) = 0.5 * || W .* (Z - K) ||^2
    where W is a binary indicator tensor (0=missing, 1=observed).
    Z is assumed to have zeros at missing (W==0) entries.
    """

    def __init__(self, indicator: ttb.tensor, normZsqr: float):
        """Prepare function and gradient handles.

        Parameters
        ----------
        indicator:
            Binary weight tensor W (0=missing, 1=observed).
        normZsqr:
            Norm squared of the data tensor Z. ||Z||^2
        """
        self.W = indicator
        self.normZsqr = normZsqr
        self._local_iter: int = 0
        self._cache: np.ndarray | None = None

    def _core(
        self,
        model: ttb.ktensor,
        data: ttb.tensor | ttb.sptensor,  # noqa: ARG002
    ) -> np.ndarray:
        """Compute (and cache) B = W .* full(model).

        The expensive ktensor-to-dense reconstruction is cached so
        function_handle and gradient_handle can share it within one
        optimizer step.
        """
        if self._local_iter == 1:
            self._local_iter = 0
            assert self._cache is not None
            B_data = self._cache
            self._cache = None
            return B_data
        self._local_iter += 1
        B_data = self.W.data * model.full().data
        self._cache = B_data
        return B_data

    def function_handle(self, model: ttb.ktensor, data: ttb.tensor | ttb.sptensor):
        """Calculate the function value F = 0.5 * ||W*(Z-M)||^2.

        Parameters
        ----------
        model:
            Current decomposition.
        data:
            Dense source tensor Z (missing entries should be zeroed out).
        """
        if not isinstance(data, ttb.tensor):
            raise ValueError("CP-WOPT requires a dense tensor")
        B_data = self._core(model, data)
        Z_data = data.data
        # F = 0.5*||Z||^2 - <Z,B> + 0.5*||B||^2
        return 0.5 * self.normZsqr - np.sum(Z_data * B_data) + 0.5 * np.sum(B_data**2)

    def gradient_handle(self, model: ttb.ktensor, data: ttb.tensor | ttb.sptensor):
        """Calculate the gradient of F = 0.5 * ||W*(Z-M)||^2.

        Parameters
        ----------
        model:
            Current decomposition.
        data:
            Dense source tensor Z (missing entries should be zeroed out).
        """
        if not isinstance(data, ttb.tensor):
            raise ValueError("CP-WOPT requires a dense tensor")
        B_data = self._core(model, data)
        Z_data = data.data
        T = ttb.tensor(Z_data - B_data, copy=False)
        G = []
        for k in range(data.ndims):
            G.append(-T.mttkrp(model.factor_matrices, k))
        return G


def setup_opt(
    scale: float,
    Xnormsqr: float,
) -> fg_return:
    """Collect the function and gradient handles for CP Opt.

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
    # cp_opt handles operate on ktensors and (sp)tensors directly, unlike gcp
    # handles which use flat numpy vectors. This is an intentional design choice.
    lower_bound = -np.inf
    fgh = FGHandlesOPT(scale, Xnormsqr)
    return fgh.function_handle, fgh.gradient_handle, lower_bound


def setup_wopt(
    indicator: ttb.tensor,
    normZsqr: float,
) -> fg_return:
    """Collect the function and gradient handles for CP WOPT.

    Parameters
    ----------
    indicator:
        Binary weight tensor W (0=missing, 1=observed).
    normZsqr:
        Norm squared of the data tensor Z. ||Z||^2

    Returns
    -------
        Function handle, gradient handle, and lower bound.
    """
    lower_bound = -np.inf
    fgh = FGHandlesWOPT(indicator, normZsqr)
    return fgh.function_handle, fgh.gradient_handle, lower_bound
