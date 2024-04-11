# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport floor

def correlate2d_cython(np.ndarray[np.float64_t, ndim=2] image, np.ndarray[np.float64_t, ndim=2] kernel):
    cdef int image_height = image.shape[0]
    cdef int image_width = image.shape[1]
    cdef int kernel_height = kernel.shape[0]
    cdef int kernel_width = kernel.shape[1]
    cdef int result_height = image_height - kernel_height + 1
    cdef int result_width = image_width - kernel_width + 1
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((result_height, result_width), dtype=np.float64)
    cdef int i, j, m, n

    for i in range(result_height):
        for j in range(result_width):
            for m in range(kernel_height):
                for n in range(kernel_width):
                    result[i, j] += image[i + m, j + n] * kernel[m, n]

    return result