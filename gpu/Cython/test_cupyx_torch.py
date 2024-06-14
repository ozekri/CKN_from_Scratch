import timeit
import numpy as np
import cupy as cp
from scipy import signal
from utils import torch_correlate2d
from cupyx.scipy.signal import correlate2d as cupyx_correlate2d

# Creating a test image and kernel
image = np.random.rand(5000, 5000)
kernel = np.random.rand(10, 10)

# Correlation calculation with scipy.signal.correlate2d
start_time = timeit.default_timer()
result_scipy = signal.correlate2d(image, kernel, mode='valid')
scipy_time = timeit.default_timer() - start_time
print("Execution time with scipy.signal.correlate2d :", scipy_time)

# Correlation calculation with torch.nn.functional.conv2d
start_time = timeit.default_timer()
result_torch = torch_correlate2d(image, kernel, padding='valid')
torch_time = timeit.default_timer() - start_time
print("Execution time with torch.nn.functional.conv2d :", torch_time)

# Correlation calculation with cupyx.scipy.signal.correlate2d
image_gpu = cp.asarray(image)
kernel_gpu = cp.asarray(kernel)
start_time = timeit.default_timer()
result_cupyx = cupyx_correlate2d(image_gpu, kernel_gpu, mode='valid')
cupyx_time = timeit.default_timer() - start_time
print("Execution time with cupyx.scipy.signal.correlate2d :", cupyx_time)