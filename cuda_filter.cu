/*
 * cuda_filter.cu
 * ----------------
 * CUDA (GPU) implementation of grayscale conversion and Sobel edge
 * detection. Same algorithms/formulas as serial_filter.c and
 * openmp_filter.c, for direct comparability:
 *   - Grayscale: gray = 0.299*R + 0.587*G + 0.114*B
 *   - Sobel: same Gx/Gy 3x3 kernels, magnitude = sqrt(Gx^2 + Gy^2), clamp 255
 *
 * Execution model:
 *   Each CUDA thread handles exactly ONE pixel. For a 512x512 image, that's
 *   262,144 threads; for 4000x4000, 16,000,000 threads -- all conceptually
 *   running "at once" on the GPU's many small cores.
 *
 *   Threads are organized into a 2D grid of 2D blocks (16x16 threads per
 *   block is a common, GPU-friendly choice). Each thread computes its own
 *   (row, col) from its block/thread indices, so it knows exactly which
 *   pixel it owns -- no coordination with other threads needed, matching
 *   the same "embarrassingly parallel" design already used in OpenMP.
 *
 * Compile (on a machine/Colab with CUDA toolkit + GPU):
 *   nvcc -O2 -o cuda_filter cuda_filter.cu
 * Run:
 *   ./cuda_filter <input_image> <gray_out.png> <edges_out.png>
 *
 * NOTE: Requires stb_image.h and stb_image_write.h in the same directory
 * (same files already used by serial_filter.c / openmp_filter.c).
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Error-checking macro: wrap CUDA API calls so failures are caught
   immediately with a clear message, instead of silently corrupting
   results later. */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/*
 * Grayscale kernel: one thread per pixel.
 * Each thread reads its own pixel's R,G,B from the input buffer and
 * writes one grayscale byte to the output buffer. No thread touches any
 * other thread's data -- identical independence guarantee as the OpenMP
 * version, just expressed as GPU threads instead of CPU threads.
 */
__global__ void grayscale_kernel(const unsigned char *input, unsigned char *output,
                                  int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return; /* guard: grid may overshoot image edges */

    int idx = (row * width + col) * channels;
    unsigned char r = input[idx + 0];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    output[row * width + col] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

/*
 * Sobel edge detection kernel: one thread per pixel.
 * Each thread reads its own 3x3 neighborhood directly from the (read-only)
 * grayscale input buffer in GPU global memory. Border pixels are set to 0
 * (same convention as the CPU versions).
 */
__global__ void sobel_kernel(const unsigned char *input, unsigned char *output,
                              int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    /* Border pixels: no full 3x3 neighborhood available -> zero, same as CPU versions */
    if (row == 0 || row == height - 1 || col == 0 || col == width - 1) {
        output[row * width + col] = 0;
        return;
    }

    /* Sobel kernels -- identical values to serial_filter.c / openmp_filter.c */
    int gx =
        -1 * input[(row - 1) * width + (col - 1)] + 1 * input[(row - 1) * width + (col + 1)] +
        -2 * input[(row    ) * width + (col - 1)] + 2 * input[(row    ) * width + (col + 1)] +
        -1 * input[(row + 1) * width + (col - 1)] + 1 * input[(row + 1) * width + (col + 1)];

    int gy =
        -1 * input[(row - 1) * width + (col - 1)] + -2 * input[(row - 1) * width + col] + -1 * input[(row - 1) * width + (col + 1)] +
         1 * input[(row + 1) * width + (col - 1)] +  2 * input[(row + 1) * width + col] +  1 * input[(row + 1) * width + (col + 1)];

    float magnitude = sqrtf((float)(gx * gx + gy * gy));
    if (magnitude > 255.0f) magnitude = 255.0f;

    output[row * width + col] = (unsigned char)magnitude;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_image> <grayscale_out.png> <edges_out.png>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *gray_out_path = argv[2];
    const char *edges_out_path = argv[3];

    int width, height, channels;
    unsigned char *h_img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!h_img) {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        return EXIT_FAILURE;
    }
    channels = 3;

    size_t img_size = (size_t)width * height * channels;
    size_t gray_size = (size_t)width * height;

    unsigned char *h_gray = (unsigned char *) malloc(gray_size);
    unsigned char *h_edges = (unsigned char *) malloc(gray_size);

    /* Device (GPU) memory pointers */
    unsigned char *d_img, *d_gray, *d_edges;
    CUDA_CHECK(cudaMalloc((void **)&d_img, img_size));
    CUDA_CHECK(cudaMalloc((void **)&d_gray, gray_size));
    CUDA_CHECK(cudaMalloc((void **)&d_edges, gray_size));

    /* Timing: use CUDA events for accurate GPU-side timing, including
       memory transfer overhead (Host->Device and Device->Host), since
       that transfer cost is a real, reportable part of GPU throughput. */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    /* Copy input image Host -> Device */
    CUDA_CHECK(cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice));

    /* Launch configuration: 16x16 threads per block is a common choice
       that balances occupancy and simplicity for image kernels. Grid size
       is computed so every pixel is covered even if width/height aren't
       exact multiples of 16 (the `if (row >= height || col >= width) return;`
       guard in each kernel handles the overshoot safely). */
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    grayscale_kernel<<<gridDim, blockDim>>>(d_img, d_gray, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    sobel_kernel<<<gridDim, blockDim>>>(d_gray, d_edges, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results Device -> Host */
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_edges, d_edges, gray_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double elapsed_s = elapsed_ms / 1000.0;

    stbi_write_png(gray_out_path, width, height, 1, h_gray, width);
    stbi_write_png(edges_out_path, width, height, 1, h_edges, width);

    printf("=== CUDA Image Filter Pipeline ===\n");
    printf("Image size          : %d x %d (%d channels)\n", width, height, channels);
    printf("Block size          : %d x %d threads\n", blockDim.x, blockDim.y);
    printf("Grid size           : %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("Total time (s)      : %.6f  (includes H2D/D2H transfer)\n", elapsed_s);
    printf("Throughput (img/s)  : %.4f\n", 1.0 / elapsed_s);
    printf("Saved: %s, %s\n", gray_out_path, edges_out_path);

    /* Cleanup */
    cudaFree(d_img);
    cudaFree(d_gray);
    cudaFree(d_edges);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    stbi_image_free(h_img);
    free(h_gray);
    free(h_edges);

    return EXIT_SUCCESS;
}
