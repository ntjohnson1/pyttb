"""Optimizer Implementations for CP OPT"""

# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Union

import numpy as np

import pyttb as ttb
from pyttb.gcp.optimizers import LBFGSB_Base
from pyttb.opt.fg_setup import function_type, gradient_type


def evaluate(
    model, data, function_handle, gradient_handle
) -> Tuple[float, List[np.ndarray]]:
    F = function_handle(model, data)
    G = gradient_handle(model, data)
    return F, G


# If we use more scipy optimizers in the future we should generalize this
class LBFGSB(LBFGSB_Base):
    """Simple wrapper around scipy lbfgsb

    NOTE: If used for publications please see scipy documentation for adding citation
    for the implementation.
    """

    def _get_lbfgsb_func_grad(
        self,
        model: ttb.ktensor,
        data: Union[ttb.tensor, ttb.sptensor],
        function_handle: function_type,
        gradient_handle: gradient_type,
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        def lbfgsb_func_grad(vector: np.ndarray) -> Tuple[float, np.ndarray]:
            model.update(np.arange(model.ndims), vector)
            func_val, grads = evaluate(
                model,
                data,
                function_handle,
                gradient_handle,
            )
            return func_val, ttb.ktensor(grads, copy=False).tovec(False)

        return lbfgsb_func_grad

    def solve(
        self,
        initial_model: ttb.ktensor,
        data: Union[ttb.tensor, ttb.sptensor],
        function_handle: function_type,
        gradient_handle: gradient_type,
        lower_bound: float = -np.inf,
    ) -> Tuple[ttb.ktensor, Dict]:
        """Solves the defined optimization problem"""

        model = initial_model.copy()

        lbfgsb_func_grad = self._get_lbfgsb_func_grad(
            model,
            data,
            function_handle,
            gradient_handle,
        )

        x0 = model.tovec(False)
        model, lbfgsb_info = self._run_solver(
            x0, model, lbfgsb_func_grad, lower_bound, data.shape
        )

        # TODO big print output
        return model, lbfgsb_info
