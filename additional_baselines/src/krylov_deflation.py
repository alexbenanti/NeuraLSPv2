import time
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def orthonormalize_columns(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    Q, _ = np.linalg.qr(Q)
    return Q


def _validated_preconditioner_apply(
    apply_preconditioner: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(apply_preconditioner(x), dtype=np.float64).reshape(-1)
    if y.shape != x.shape:
        raise RuntimeError(f"Preconditioner returned shape {y.shape}, expected {x.shape}.")
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("Preconditioner returned non-finite values.")
    return y


def solve_with_prebuilt_preconditioner(
    A,
    b,
    apply_preconditioner: Callable[[np.ndarray], np.ndarray],
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
) -> Tuple[float, int]:
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)

    if maxiter is None:
        maxiter = 5 * A_csr.shape[0]

    M = spla.LinearOperator(
        shape=A_csr.shape,
        matvec=lambda x: _validated_preconditioner_apply(apply_preconditioner, x),
        dtype=np.float64,
    )

    iters = [0]

    def callback(_xk):
        iters[0] += 1

    t0 = time.perf_counter()
    try:
        _, info = spla.cg(
            A_csr,
            b_vec,
            M=M,
            rtol=tol,
            atol=0.0,
            maxiter=maxiter,
            callback=callback,
        )
    except TypeError:
        _, info = spla.cg(
            A_csr,
            b_vec,
            M=M,
            tol=tol,
            maxiter=maxiter,
            callback=callback,
        )
    t_solve = time.perf_counter() - t0

    if info < 0:
        raise RuntimeError("CG failed with illegal input or breakdown.")
    if info > 0 and iters[0] == 0:
        iters[0] = int(info)

    return t_solve, int(iters[0])


def solve_with_deflated_cg(
    A,
    b,
    Q,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
) -> Tuple[float, float, int]:
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A_csr.shape[0]

    if maxiter is None:
        maxiter = 5 * n

    t0 = time.perf_counter()
    Q = orthonormalize_columns(Q)
    AQ = A_csr @ Q
    E = Q.T @ AQ
    E_reg = E + 1e-10 * np.eye(E.shape[0], dtype=np.float64)
    E_inv = np.linalg.inv(E_reg)

    def projector(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x - AQ @ (E_inv @ (Q.T @ x))

    coarse = Q @ (E_inv @ (Q.T @ b))
    rhs = projector(b)

    A_def = spla.LinearOperator(
        shape=A_csr.shape,
        matvec=lambda x: projector(A_csr @ np.asarray(x, dtype=np.float64)),
        dtype=np.float64,
    )
    t_setup = time.perf_counter() - t0

    iters = [0]

    def callback(_xk):
        iters[0] += 1

    t1 = time.perf_counter()
    try:
        y, info = spla.cg(
            A_def,
            rhs,
            rtol=tol,
            atol=0.0,
            maxiter=maxiter,
            callback=callback,
        )
    except TypeError:
        y, info = spla.cg(
            A_def,
            rhs,
            tol=tol,
            maxiter=maxiter,
            callback=callback,
        )
    t_solve = time.perf_counter() - t1

    if info < 0:
        raise RuntimeError("Deflated CG failed with illegal input or breakdown.")
    if info > 0 and iters[0] == 0:
        iters[0] = int(info)

    x = coarse + y - Q @ (E_inv @ (Q.T @ (A_csr @ y)))
    _ = x
    return t_setup, t_solve, int(iters[0])