import numpy as np
import scipy.sparse as sp
import scipy.ndimage as ndi
from scipy.spatial import Delaunay

from typing import Optional

#First step, we have to make the unit square mesh 

def unit_square_mesh(N: int):
    """
    This function generates the triangular mesh on the unit square [0,1]^2

    Args:
        N: this is the number of subdivisions along a single axis (total number of nodes = (N+1)^2)
    
        
    Returns: 
        nodes: (n_nodes, 2) array of vertex coordinates 
        triangles: (n_elems, 3) array of triangle indicies 
    """

    xs = np.linspace(0.0, 1.0, N+1)
    ys = np.linspace(0.0, 1.0, N+1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)

    def idx(i,j):
        return i*(N+1) + j
    
    triangles = []
    for i in range(N):
        for j in range(N):
            #split the square cell into two triangles 
            n00, n10 = idx(i,j), idx(i+1, j)
            n01, n11 = idx(i,j+1), idx(i+1, j+1)

            #First Triangle 

            triangles.append([n00,n10,n11])

            #Second Triangle

            triangles.append([n00,n11,n01])

    
    return nodes, np.array(triangles, dtype = np.int64)

def filter_degenerate_triangles(nodes: np.ndarray, tris: np.ndarray, det_eps: float) -> np.ndarray:
    """
    Remove triangles with near-zero signed area (|detJ| < det_eps).
    This prevents crashes in FEM assembly when Delaunay produces degenerate boundary simplices.
    """
    xy = nodes[tris]  # (T, 3, 2)

    x1 = xy[:, 0, 0]; y1 = xy[:, 0, 1]
    x2 = xy[:, 1, 0]; y2 = xy[:, 1, 1]
    x3 = xy[:, 2, 0]; y3 = xy[:, 2, 1]

    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    mask = np.abs(detJ) >= det_eps

    

    return tris[mask]

def delaunay_jitter_mesh(N: int, jitter: float = 0.35, seed: Optional[int] = None):
    try:
        from scipy.spatial import Delaunay
    except ImportError as e:
        raise ImportError("scipy is required: pip install scipy") from e

    rng = np.random.default_rng(seed)

    xs = np.linspace(0.0, 1.0, N + 1)
    ys = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Jitter ONLY interior points, and do NOT clip (clipping can create duplicates).
    h = 1.0 / N
    eps = 1e-12
    is_boundary = (
        (nodes[:, 0] < eps) | (nodes[:, 0] > 1.0 - eps) |
        (nodes[:, 1] < eps) | (nodes[:, 1] > 1.0 - eps)
    )

    # Safe jitter: cap by distance to boundary so no point crosses onto boundary
    d = np.minimum.reduce([nodes[:,0], 1.0 - nodes[:,0], nodes[:,1], 1.0 - nodes[:,1]])
    max_step = np.minimum(jitter * h, 0.49 * d)
    max_step[is_boundary] = 0.0

    noise = (rng.random(nodes.shape) - 0.5) * 2.0 * max_step[:, None]
    nodes = nodes + noise

    tri = Delaunay(nodes, qhull_options="QJ Qt")
    tris = tri.simplices.astype(np.int64)

    # Filter degenerate triangles (critical!)
    det_eps = 1e-14  # works well for N up to a few hundred
    tris = filter_degenerate_triangles(nodes, tris, det_eps=det_eps)

    if tris.shape[0] == 0:
        raise RuntimeError("Delaunay produced no valid triangles after filtering. Reduce jitter or change point set.")

    return nodes, tris


def get_triangle_gradients_and_area(xy):
    """
    This function computes the gradients of the linear basis functions on a triangle
    """

    x1,y1 = xy[0]
    x2,y2 = xy[1]
    x3,y3 = xy[2]

    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    if abs(detJ) < 1e-14:
        raise ValueError("Degenerate triangle detected.")

    area = 0.5 * abs(detJ)

    b1 = y2-y3; b2 = y3-y1; b3 = y1-y2
    c1 = x3-x2; c2 = x1-x3; c3 = x2-x1

    gradients = np.array([[b1,c1],[b2,c2],[b3,c3]]) / detJ
    return gradients, area


#Next step, we write the FEM assembly routines 


def assemble_diffusion(nodes, triangles, kappa=None):
    """
    Here, we assemble the stiffness matrix A for -div(kappa*grad u)
    """

    n_nodes = nodes.shape[0]
    n_triangles = triangles.shape[0]
    rows, cols, data = [], [], []

    for t in range(n_triangles):
        tri_indices = triangles[t]
        tri_coords = nodes[tri_indices]
        gradients, area = get_triangle_gradients_and_area(tri_coords)

        if kappa is None:
            G = np.eye(2)
        elif np.isscalar(kappa):
            G = kappa*np.eye(2)
        elif kappa.ndim == 1:
            G = kappa[t] * np.eye(2)
        else:
            G = kappa[t]
    
        K_local = area*(gradients @ G @ gradients.T)

        for i_loc in range(3):
            for j_loc in range(3):
                rows.append(tri_indices[i_loc])
                cols.append(tri_indices[j_loc])
                data.append(K_local[i_loc, j_loc])

    A = sp.csr_matrix((data, (rows,cols)), shape = (n_nodes, n_nodes))
    A = 0.5* (A+A.T)
    return A
    

def assemble_mass(nodes, tris):
    """
    Assembles the Mass Matrix M for (u, v).
    Needed for Screened Poisson.
    """
    n_nodes = nodes.shape[0]
    rows, cols, data = [], [], []
    base_mass = np.array([[2, 1, 1],
                          [1, 2, 1],
                          [1, 1, 2]])

    for t in range(tris.shape[0]):
        tri_indices = tris[t]
        _, area = get_triangle_gradients_and_area(nodes[tri_indices])
        M_local = (area / 12.0) * base_mass
        for i_loc in range(3):
            for j_loc in range(3):
                rows.append(tri_indices[i_loc])
                cols.append(tri_indices[j_loc])
                data.append(M_local[i_loc, j_loc])

    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

def assemble_advection(nodes, tris, velocity_field):
    """
    Assembles the Advection Matrix C for v . grad(u).
    This term is Non-Symmetric.
    """
    n_nodes = nodes.shape[0]
    rows, cols, data = [], [], []
    
    for t in range(tris.shape[0]):
        tri_indices = tris[t]
        # Centroid of triangle
        centroid = np.mean(nodes[tri_indices], axis=0)
        
        # Velocity at centroid (2,)
        v = velocity_field(centroid) 
        
        # Gradients (3, 2)
        grads, area = get_triangle_gradients_and_area(nodes[tri_indices])
        
        # Local Advection: Integral( phi_i * (v . grad phi_j) )
        # Using 1-point quadrature at centroid
        # shape functions phi_i at centroid are all 1/3
        
        # C_local_ij = Area * (1/3) * (v . grad_phi_j)
        # This is a simplified mass-lumped-ish advection
        
        for i in range(3):
            for j in range(3):
                val = area * (1.0/3.0) * np.dot(v, grads[j])
                rows.append(tri_indices[i])
                cols.append(tri_indices[j])
                data.append(val)
                
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))



def apply_dirichlet_bc(A, nodes, mask_fn=None):
    """
    Enforces u=0 on boundaries.
    """
    if mask_fn is None:
        eps = 1e-8
        def mask_fn(x, y):
            return (x < eps) | (x > 1.0 - eps) | (y < eps) | (y > 1.0 - eps)

    x, y = nodes[:, 0], nodes[:, 1]
    boundary_mask = mask_fn(x, y)
    boundary_indices = np.where(boundary_mask)[0]
    
    A = A.tolil()
    A[boundary_indices, :] = 0.0
    A[:, boundary_indices] = 0.0
    A[boundary_indices, boundary_indices] = 1.0
    return A.tocsr()

#Finally, we implement the smoother and data generation 

def smooth_test_vectors(A, num_vectors=20, sweeps=10, omega=0.66):
    """
    Generates 'smoothed' random vectors S using Jacobi smoothing.
    """
    n = A.shape[0]
    rng = np.random.default_rng()
    X = rng.standard_normal((n, num_vectors))
    
    diag_inv = 1.0 / (A.diagonal() + 1e-12)
    Dinv = sp.diags(diag_inv)
    
    for _ in range(sweeps):
        AX = A @ X
        X = X - omega * (Dinv @ AX)
        # Normalize to prevent numerical vanish
        norms = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
        X = X / norms
        
    return X.astype(np.float32)

def generate_pde_data(N, pde_type="diffusion", mesh_type='unstructured_delaunay'):
    """
    Generates problem matrices A and training data S.
    Supported types:
        "diffusion"
        "anisotropic"
        "screened_poisson"
        "heat_equation"   -> backward-Euler step matrix (M + dt K)
        "wave_equation"   -> implicit wave step matrix (M + c^2 dt^2 K)
    """
    if mesh_type == 'structured':
        nodes, tris = unit_square_mesh(N)
    elif mesh_type == 'unstructured_delaunay':
        nodes, tris = delaunay_jitter_mesh(N)
    else:
        raise ValueError('Invalid Mesh Type')

    n_elems = tris.shape[0]

    if pde_type == "diffusion":
        # High-contrast diffusion
        kappa = np.exp(np.random.randn(n_elems) * 1.5)
        A = assemble_diffusion(nodes, tris, kappa)

    elif pde_type == "anisotropic":
        # Random anisotropic tensors
        kappa = np.zeros((n_elems, 2, 2))
        for i in range(n_elems):
            theta = np.random.rand() * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            D = np.diag([1000.0, 1.0])
            kappa[i] = R @ D @ R.T
        A = assemble_diffusion(nodes, tris, kappa)

    elif pde_type == "screened_poisson":
        # -Delta u + alpha * u = f (SPD!)
        log_kappa = np.random.randn(n_elems) * 1.0
        kappa = np.exp(log_kappa)
        K = assemble_diffusion(nodes, tris, kappa)
        M = assemble_mass(nodes, tris)

        log_alpha = np.random.uniform(0, 2.0)
        alpha = 10.0**log_alpha
        A = K + alpha * M

    elif pde_type == "heat_equation":
        # To match 'screened_poisson' conditioning:
        log_kappa = np.random.randn(n_elems) * 1.0  # Matched variance
        kappa = np.exp(log_kappa)
        K = assemble_diffusion(nodes, tris, kappa)
        M = assemble_mass(nodes, tris)

        # Alpha was 10^[0, 2]. Therefore, dt = 1/alpha is 10^[-2, 0]
        log_dt = np.random.uniform(-2.0, 0.0) 
        dt = 10.0**log_dt
        A = M + dt * K

    elif pde_type == "wave_equation":
        # To match pure 'diffusion' conditioning:
        log_kappa = np.random.randn(n_elems) * 1.5  # Matched variance
        kappa = np.exp(log_kappa)
        K = assemble_diffusion(nodes, tris, kappa)
        M = assemble_mass(nodes, tris)

        h = 1.0 / N
        
        # We need (c * dt)^2 to be large so that K dominates.
        # A factor of 10^3 to 10^5 is usually enough to wash out M.
        c = 10.0**np.random.uniform(2, 3)    
        dt = np.random.uniform(h, 2.0 * h)
        
        A = M + (c * dt)**2 * K

    else:
        raise ValueError(f"Unsupported PDE type: {pde_type}")

    # Apply BCs
    A = apply_dirichlet_bc(A, nodes)
    return A





