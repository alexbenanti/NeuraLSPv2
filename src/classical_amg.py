import numpy as np
import scipy.sparse as sp

def classical_msa_inference(A_csr, S_smooth, num_aggregates=None):
    """
    Pure Python Smoothed Aggregation Baseline.
    
    Logic:
    1. Aggregation: Random Voronoi partitioning (Vectorized for speed).
    2. Tentative P: Injection from aggregates.
    3. Smoothed P: P <- (I - w D^-1 A) P_tent.
    """
    n = A_csr.shape[0]
    
    if num_aggregates is None: 
        num_aggregates = max(1, n // 4)
    
    # Ensure num_aggregates doesn't exceed n
    num_aggregates = min(num_aggregates, n)
        
    rng = np.random.default_rng()
    seeds = rng.choice(n, size=num_aggregates, replace=False)
    
    # Vectorized Voronoi Aggregation
    cols = np.arange(num_aggregates)
    data = np.ones(num_aggregates)
    # SeedMat: Columns are indicator vectors for each seed
    SeedMat = sp.coo_matrix((data, (seeds, cols)), shape=(n, num_aggregates)).tocsr()
    
    # 
    # This step simulates "flooding" from seeds to neighbors to form aggregates
    A_abs = abs(A_csr)
    A_abs.setdiag(1.0) 
    Influence = A_abs @ SeedMat
    
    # Use dense for argmax (fast enough for N=1024)
    dense_inf = Influence.toarray()
    
    # FIX: Force assignments to be at least 1D array
    assignments = np.argmax(dense_inf, axis=1)
    assignments = np.atleast_1d(assignments) 
    
    # Construct Sparse Tentative P
    rows = np.arange(n)
    P_tent = sp.coo_matrix((np.ones(n), (rows, assignments)), shape=(n, num_aggregates)).tocsr()
    
    # Smoother (Jacobi)
    omega = 0.66
    D_inv = sp.diags(1.0 / (A_csr.diagonal() + 1e-12))
    Smoother_Op = sp.eye(n) - omega * (D_inv @ A_csr)
    P_smooth = Smoother_Op @ P_tent
    
    # FIX: Convert np.matrix (from todense) to np.array to prevent broadcasting errors
    return np.asarray(P_smooth.todense())