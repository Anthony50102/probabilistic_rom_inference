import jax
import jax.numpy as jnp
from typing import Dict

# @jax.jit
def rbf_kernel(x1: jnp.ndarray,
               x2: jnp.ndarray,
               lengthscale: float,
               variance: float) -> jnp.ndarray:
    """
    RBF (Gaussian) kernel function.
    
    Args:
        x1: Input array of shape [N, D]
        x2: Input array of shape [M, D]
        lengthscale: Length scale parameter
        variance: Variance parameter
    
    Returns:
        Kernel matrix K of shape [N, M] with
        K_ij = variance * exp(-0.5 * ||x1[i]-x2[j]||^2 / lengthscale^2)
    """
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :])**2, axis=-1)
    return variance * jnp.exp(-0.5 * sqdist / (lengthscale**2))

# @jax.jit
def c_dash(params: Dict[str, float], t1: float, t2: float) -> float:
    """
    d/dt2 of the RBF kernel
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t1: First input point
        t2: Second input point
        
    Returns:
        Derivative of kernel with respect to t2
    """
    l2 = params['lengthscale']**2
    return (t1 - t2) / l2 * k(params, t1, t2)

# @jax.jit
def dash_c(params: Dict[str, float], t1: float, t2: float) -> float:
    """
    d/dt1 of the RBF kernel
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t1: First input point
        t2: Second input point
        
    Returns:
        Derivative of kernel with respect to t1
    """
    l2 = params['lengthscale']**2
    return -(t1 - t2) / l2 * k(params, t1, t2)

# @jax.jit
def c_double_dash(params: Dict[str, float], t1: float, t2: float) -> float:
    """
    d^2/dt1/dt2 of the RBF kernel
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t1: First input point
        t2: Second input point
        
    Returns:
        Second mixed derivative of kernel with respect to t1 and t2
    """
    l2 = params['lengthscale']**2
    diff = t1 - t2
    return (1 / l2 - diff**2 / l2**2) * k(params, t1, t2)


# @jax.jit
def flatten_time(t: jnp.ndarray) -> jnp.ndarray:
    """
    Return t with shape (n,) no matter if (n,), (n,1) or (1,n) was given.
    
    Args:
        t: Input time array
        
    Returns:
        Flattened time array with shape (n,)
    """
    return jnp.ravel(t)

# @jax.jit
def rbf_kernel_no_nugget(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Full n×n RBF kernel matrix K_ij = variance * exp(-(t_i-t_j)^2 / (2*ell^2)).
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Kernel matrix of shape [n, n]
    """
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = params["lengthscale"] ** 2
    return params["variance"] * jnp.exp(-diff**2 / (2.0 * ell2))

# @jax.jit
def get_c_phi(params: Dict[str, float], t: jnp.ndarray, nugget: float = 1e-4) -> jnp.ndarray:
    """
    Kernel matrix plus nugget on the diagonal.
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        nugget: Small value added to diagonal for numerical stability
        
    Returns:
        Kernel matrix of shape [n, n] with nugget on diagonal
    """
    kmat = rbf_kernel_no_nugget(params, t)
    return kmat + nugget * jnp.eye(kmat.shape[0])

# @jax.jit
def get_c_phi_dash(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Derivative with respect to the second time argument (dt2).
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Derivative kernel matrix of shape [n, n]
    """
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = params["lengthscale"] ** 2
    return (diff / ell2) * rbf_kernel_no_nugget(params, t)

# @jax.jit
def get_dash_c_phi(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Derivative with respect to the first time argument (dt1).
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Derivative kernel matrix of shape [n, n]
    """
    return -get_c_phi_dash(params, t)

# @jax.jit
def get_c_phi_double_dash(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Second mixed derivative with respect to both time arguments.
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Second derivative kernel matrix of shape [n, n]
    """
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = params["lengthscale"] ** 2
    return (1.0 / ell2 - diff**2 / ell2**2) * rbf_kernel_no_nugget(params, t)

def get_As(t, params):
    CDashs = get_c_phi_dash(params, t)
    DashCs = get_dash_c_phi(params, t) 
    CPhis = get_c_phi(params, t)
    CDoubleDashs = get_c_phi_double_dash(params, t)
    A = []
    # for i in jnp.arange(len(CDashs)):
    A.append(
        CDoubleDashs - jnp.dot(
            DashCs,
            jnp.linalg.solve(CPhis, CDashs)))
    return A

def get_Ds(t, params):
    """
    each entry represents a state
    
    Parameters
    ----------
    DashCs:         list of matrices of shape nTime x nTime
    CInvs:          list of matrices of shape nTime x nTime

    Returns
    ----------
    D:  list of matrices of shape nTime x nTime
        each entry represents one state
    """
    DashCs = get_dash_c_phi(params, t)
    CPhis = get_c_phi(params, t)
    D = []
    def getProdWithD(x,):
        """
        defines a function to get a product of a vector with the matrix D
        """
        return jnp.dot(DashCs,
                    jnp.linalg.solve(CPhis, x)
                    )
    D.append(getProdWithD)
    return D