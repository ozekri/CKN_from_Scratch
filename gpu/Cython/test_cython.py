import numpy as np
from scipy import signal
import timeit
from correlate2d_cython import correlate2d_cython
from utils import torch_correlate2d
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

# Correlation calculation with torch.nn.functional.conv2d
start_time = timeit.default_timer()
result_torch = torch_correlate2d(image, kernel, padding='valid')
torch_time = timeit.default_timer() - start_time


# Correlation calculation with torch.nn.functional.conv2d
image_gpu = cp.asarray(image)
kernel_gpu = cp.asarray(kernel)
start_time = timeit.default_timer()
result_cupy = signal.correlate2d(image_gpu.get(),kernel_gpu.get(), mode='valid')
cupy_time = timeit.default_timer() - start_time


print("Execution time Cython :", cython_time)
print("Execution time scipy.signal.correlate2d :", scipy_time)
print("Execution time torch.nn.functional.conv2d :", torch_time)
print("Execution time Cupy :", torch_time)
