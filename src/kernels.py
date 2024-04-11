import numpy as np

def exp(x, alpha):
    """Element-wise exponential non-linearity.

    The kernel_exp function is defined as k(x) = exp(alpha * (x - 1)).

    Args:
        x (ndarray): Input array.
        alpha (float): Scaling factor.

    Returns:
        ndarray: Output array with the same shape as input x.
    """
    return np.exp(alpha * (x - 1.))

def poly(x, alpha=2):
    """Element-wise polynomial non-linearity.

    The polynomial function computes the element-wise exponentiation of the input array to the power of alpha.

    Args:
        x (ndarray): Input array.
        alpha (float, optional): Exponentiation factor. Defaults to 2.

    Returns:
        ndarray: Output array containing each element of x raised to the power of alpha.
    """
    return np.power(x, alpha)

kernels = {
    "exp": exp,
    "poly": poly
}