# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import numpy as np

import pyttb as ttb
from pyttb.cp_opt import _get_initial_guess


def test_initial_guesses():
    rank = 2
    data = ttb.tenones((2, 2, 2))

    M0 = _get_initial_guess(data, rank, "random")
    assert M0.full().shape == data.shape
    assert np.all(M0.weights == 1)
    assert all(np.all(fm >= 0.0) and np.all(fm <= 1.0) for fm in M0.factor_matrices)

    M0 = _get_initial_guess(data, rank, "random_normal")
    assert M0.full().shape == data.shape
    assert np.all(M0.weights == 1)

    M1 = _get_initial_guess(data, rank, M0)
    assert M1.isequal(M0)
    assert M1 is M0

    M1 = _get_initial_guess(data, rank, M0.factor_matrices)
    assert M1.isequal(M0)
