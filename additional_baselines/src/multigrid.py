import time
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, gmres, LinearOperator

class TwoGridPreconditioner:
    def __init__(self, A_csr, P_dense, omega=0.66, nu_pre=5, nu_post=5):
        self.A = A_csr
        self.P = P_dense
        self.omega = omega
        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.N = A_csr.shape[0]
        
        self.Dinv = 1.0 / (A_csr.diagonal() + 1e-12)
        
        # Ensure P is (N, r) even if r=1
        if self.P.ndim == 1:
            self.P = self.P[:, None]
            
        AP = A_csr @ P_dense
        self.Ac = P_dense.T @ AP
        
        #Force Ac to be 2D (r, r) even if r=1 
        self.Ac = np.atleast_2d(self.Ac)

        try:
            self.Ac_inv = np.linalg.inv(self.Ac + 1e-10 * np.eye(self.Ac.shape[0]))
            self.solve_coarse = lambda r: self.Ac_inv @ r
        except np.linalg.LinAlgError:
            self.solve_coarse = lambda r: np.linalg.solve(self.Ac + 1e-8*np.eye(self.Ac.shape[0]), r)

    def smooth(self, x, b, nu: int):
        for _ in range(int(nu)):
            r = b - self.A @ x
            x += self.omega * (self.Dinv * r)
        return x

    def __call__(self, r):
        z = np.zeros_like(r)
        z = self.smooth(z, r, self.nu_pre)
        res = r - self.A @ z
        
        # Restriction
        # rc will be (r,) or (r, 1)
        rc = self.P.T @ res
        
        
        if rc.ndim == 1:
            rc = rc[:, None]
            
        ec = self.solve_coarse(rc)
        
        # Flatten back to 1D for prolongation
        ec = ec.flatten()
        
        z += self.P @ ec
        z = self.smooth(z, r, self.nu_post)
        return z


def pcg_solve(A, b, M_apply, tol=1e-8, max_iter=500):
    t_start = time.perf_counter()
    x = np.zeros_like(b)
    r = b - A @ x
    res_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(b) + 1e-12
    
    history = [res_norm / b_norm]
    time_history = [time.perf_counter() - t_start]
    
    if history[-1] < tol: return x, history, time_history

    z = M_apply(r)
    p = z.copy()
    rz = np.dot(r, z)
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = rz / (np.dot(p, Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        
        res_norm = np.linalg.norm(r)
        rel_res = res_norm / b_norm
        
        history.append(rel_res)
        time_history.append(time.perf_counter() - t_start)
        
        if rel_res < tol: break
            
        z = M_apply(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
        
    return x, history, time_history

