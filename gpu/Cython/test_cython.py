import numpy as np
from scipy import signal
import timeit
from correlate2d_cython import correlate2d_cython
import cupy as cp

# Creating a test image and kernel
image = np.random.rand(500, 500)
kernel = np.random.rand(10, 10)

# Correlation calculation with our Cython implementation
start_time = timeit.default_timer()
result_cython = correlate2d_cython(image, kernel)
cython_time = timeit.default_timer() - start_time

# Correlation calculation with scipy.signal.correlate2d
start_time = timeit.default_timer()
result_scipy = signal.correlate2d(image, kernel, mode='valid')
scipy_time = timeit.default_timer() - start_time

print("Execution time Cython :", cython_time)
print("Execution time scipy.signal.correlate2d :", scipy_time)