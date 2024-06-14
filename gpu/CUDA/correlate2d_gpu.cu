/*!
Cuda implementation of scipy.signal.correlate2d
See <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html> for
the origial implementation.
*/


extern "C" {
    __global__ void correlate2d_gpu_kernel(
        float* result,
        float* image,
        float* kernel,
        int image_width,
        int image_height,
        int kernel_width,
        int kernel_height) {
        /*!
        results: Contains the discrete linear cross-correlation of image with kernel.
        image: First input.
        kernel: Second input.
        image_width: Width of first input.
        image_height: Height of first input.
        kernel_width: Width of second input.
        kernel_height: Height of second input.
        */
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < image_width - kernel_width + 1 && j < image_height - kernel_height + 1) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel_width; ki++) {
                for (int kj = 0; kj < kernel_height; kj++) {
                    sum += kernel[ki * kernel_width + kj] * image[(i + ki) * image_width + (j + kj)];
                }
            }
            result[i * (image_height - kernel_height + 1) + j] = sum;
        }
    }

    void correlate2d_gpu(
        float* result,
        float* image,
        float* kernel,
        int image_width,
        int image_height,
        int kernel_width,
        int kernel_height) {
        /*!
        results: Contains the discrete linear cross-correlation of image with kernel.
        image: First input.
        kernel: Second input.
        image_width: Width of first input.
        image_height: Height of first input.
        kernel_width: Width of second input.
        kernel_height: Height of second input.
        */
        float* d_result;
        float* d_image;
        float* d_kernel;

        cudaMalloc((void**)&d_result, (image_width - kernel_width + 1) * (image_height - kernel_height + 1) * sizeof(float));
        cudaMalloc((void**)&d_image, image_width * image_height * sizeof(float));
        cudaMalloc((void**)&d_kernel, kernel_width * kernel_height * sizeof(float));

        cudaMemcpy(d_image, image, image_width * image_height * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel, kernel_width * kernel_height * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((image_width - kernel_width + 1 + blockSize.x - 1) / blockSize.x, (image_height - kernel_height + 1 + blockSize.y - 1) / blockSize.y);

        correlate2d_gpu_kernel<<<gridSize, blockSize>>>(d_result, d_image, d_kernel, image_width, image_height, kernel_width, kernel_height);

        cudaMemcpy(result, d_result, (image_width - kernel_width + 1) * (image_height - kernel_height + 1) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_result);
        cudaFree(d_image);
        cudaFree(d_kernel);
    }
}