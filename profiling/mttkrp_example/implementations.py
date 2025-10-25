from __future__ import annotations

import numpy as np


def old_mttkrp(Ul, Y, Ur, szl, szr, szn, R):
    """Existing implementation"""
    Y = np.reshape(Y, (-1, szr))
    Y = Y @ Ul
    Y = np.reshape(Y, (szl, szn, R))
    Ur = Ur.reshape((szl, 1, R))
    V = np.zeros((szn, R))
    for r in range(R):
        V[:, [r]] = Y[:, :, r].T @ Ur[:, :, r]
    return V


def new_mttkrp(Ul, Y, Ur, szl, szr, szn, R):
    """Proposed implementation using einsum"""
    Y = np.reshape(Y, (-1, szr))
    Y = Y.reshape(-1, szr) @ Ul
    Y = np.reshape(Y, (szl, szn, R))
    V = np.einsum("ijk, ik -> jk", Y, Ur)
    return V
