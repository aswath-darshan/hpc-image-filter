/*
 * cuda_filter_streams.cu
 * -----------------------
 * Advanced CUDA Implementation: Asynchronous Pipelining with CUDA Streams & Pinned Memory.
 *
 * HPC Concepts Demonstrated:
 *   1. Pipelining & Concurrency: Overlapping Host-to-Device (H2D) PCIe data transfer,
 *      Kernel computation on Streaming Multiprocessors (SMs), and Device-to-Host (D2H) transfer.
 *   2. Pinned (Page-Locked) Host Memory: Allocated via cudaMallocHost() to enable true
 *      Direct Memory Access (DMA) asynchronous transfers via cudaMemcpyAsync().
 *   3. 2D Domain Decomposition with Stencil Halo Exchange: Splitting 2D image domains
 *      into horizontal slabs while managing 1-pixel boundary halo regions for 3x3 Sobel convolutions.
 *   4. Latency Hiding & PCIe Bandwidth Saturation: Overcoming the PCIe transfer bottleneck
 *      inherent in discrete GPU computing (Amdahl's Law for heterogeneous systems).
 *
 * Execution Model Comparison:
 *   - Standard CUDA (cuda_filter.cu):
 *       [ H2D Transfer (All) ] -> [ Kernel Compute (All) ] -> [ D2H Transfer (All) ]
 *       (PCIe bus is idle during compute; GPU SMs are idle during transfers)
 *
 *   - Pipelined CUDA Streams (cuda_filter_streams.cu):
 *       Stream 0: [ H2D_0 ] -> [ Kernel_0 ] -> [ D2H_0 ]
 *       Stream 1:              [ H2D_1    ] -> [ Kernel_1 ] -> [ D2H_1 ]
 *       Stream 2:                              [ H2D_2    ] -> [ Kernel_2 ] -> [ D2H_2 ]
 *       Stream 3:                                            [ H2D_3    ] -> [ Kernel_3 ] -> [ D2H_3 ]
 *       (Compute Engine and Copy Engines run concurrently, hiding transfer latency!)
 *
 * Compile:
 *   nvcc -O2 -o cuda_filter_streams cuda_filter_streams.cu
 *
 * Run:
 *   ./cuda_filter_streams <input_image> <gray_out.png> <edges_out.png> [num_streams] [serial_baseline_s]
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define DEFAULT_NUM_STREAMS 4
#define MAX_STREAMS 16

/* Error checking macro */
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
 * Grayscale Kernel (Point Operator):
 * Converts RGB to Grayscale for an arbitrary horizontal slab (height = chunk_h).
 * No halo required because pixel transformation is strictly point-wise independent.
 */
__global__ void grayscale_chunk_kernel(const unsigned char *input, unsigned char *output,
                                       int width, int chunk_h, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= chunk_h || col >= width) return;

    int idx = (row * width + col) * channels;
    unsigned char r = input[idx + 0];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    output[row * width + col] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

/*
 * Sobel Edge Detection Kernel with Stencil Halo Support:
 * Computes 3x3 Sobel gradient for an output chunk of height `out_h`.
 *
 * Halo Management:
 *   - If the chunk is NOT the first chunk (r_start > 0), input has a 1-row upper halo,
 *     so the active row in the chunk input buffer starts at relative row = 1.
 *   - If the chunk is NOT the last chunk (r_start + out_h < full_height), input has a 1-row lower halo.
 *   - Boundary pixels of the GLOBAL full image are clamped to 0.
 */
__global__ void sobel_chunk_kernel(const unsigned char *gray_in, unsigned char *edges_out,
                                   int width, int full_height, int r_start, int out_h,
                                   int is_first_chunk) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_out_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_out_row >= out_h || col >= width) return;

    int global_row = r_start + local_out_row;

    /* Global image boundary conditions (edges are zeroed) */
    if (global_row == 0 || global_row == full_height - 1 || col == 0 || col == width - 1) {
        edges_out[local_out_row * width + col] = 0;
        return;
    }

    /* Relative row index inside local gray_in chunk */
    int r_in = local_out_row + (is_first_chunk ? 0 : 1);
    int r_up = r_in - 1;
    int r_mid = r_in;
    int r_dn = r_in + 1;

    /* Sobel gradient calculation */
    int gx =
        -1 * gray_in[r_up * width + (col - 1)] + 1 * gray_in[r_up * width + (col + 1)] +
        -2 * gray_in[r_mid * width + (col - 1)] + 2 * gray_in[r_mid * width + (col + 1)] +
        -1 * gray_in[r_dn * width + (col - 1)] + 1 * gray_in[r_dn * width + (col + 1)];

    int gy =
        -1 * gray_in[r_up * width + (col - 1)] + -2 * gray_in[r_up * width + col] + -1 * gray_in[r_up * width + (col + 1)] +
         1 * gray_in[r_dn * width + (col - 1)] +  2 * gray_in[r_dn * width + col] +  1 * gray_in[r_dn * width + (col + 1)];

    float magnitude = sqrtf((float)(gx * gx + gy * gy));
    if (magnitude > 255.0f) magnitude = 255.0f;

    edges_out[local_out_row * width + col] = (unsigned char)magnitude;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_image> <grayscale_out.png> <edges_out.png> [num_streams] [serial_baseline_s]\n", argv[0]);
        fprintf(stderr, "  num_streams        : number of concurrent CUDA streams (default: %d, max: %d)\n", DEFAULT_NUM_STREAMS, MAX_STREAMS);
        fprintf(stderr, "  serial_baseline_s  : optional baseline time from serial_filter for Speedup/Efficiency\n");
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *gray_out_path = argv[2];
    const char *edges_out_path = argv[3];

    int num_streams = DEFAULT_NUM_STREAMS;
    double serial_baseline_time = 0.0;
    int has_baseline = 0;

    if (argc >= 5) {
        /* Check if argv[4] is integer stream count or floating baseline */
        if (strchr(argv[4], '.') != NULL) {
            serial_baseline_time = atof(argv[4]);
            has_baseline = 1;
        } else {
            num_streams = atoi(argv[4]);
            if (num_streams < 1) num_streams = 1;
            if (num_streams > MAX_STREAMS) num_streams = MAX_STREAMS;
        }
    }
    if (argc >= 6) {
        serial_baseline_time = atof(argv[5]);
        has_baseline = 1;
    }

    /* 1. Load image using stb_image */
    int width, height, channels;
    unsigned char *raw_img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!raw_img) {
        fprintf(stderr, "Failed to load input image: %s\n", input_path);
        return EXIT_FAILURE;
    }
    channels = 3;

    size_t img_size = (size_t)width * height * channels;
    size_t gray_size = (size_t)width * height;

    /*
     * 2. Pinned (Page-Locked) Host Memory:
     * Regular malloc() memory is pageable. The OS virtual memory manager can move it,
     * so standard cudaMemcpy copies through a staging buffer.
     * cudaMallocHost() pins the host memory, allowing GPU DMA engines to transfer
     * data asynchronously in parallel with kernel execution without CPU intervention!
     */
    unsigned char *h_img_pinned, *h_gray_pinned, *h_edges_pinned;
    CUDA_CHECK(cudaMallocHost((void **)&h_img_pinned, img_size));
    CUDA_CHECK(cudaMallocHost((void **)&h_gray_pinned, gray_size));
    CUDA_CHECK(cudaMallocHost((void **)&h_edges_pinned, gray_size));

    /* Copy raw decoded image to pinned buffer */
    memcpy(h_img_pinned, raw_img, img_size);
    stbi_image_free(raw_img);

    /* Get Device Information */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    /* 3. Create CUDA Streams */
    cudaStream_t streams[MAX_STREAMS];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    /*
     * 4. Compute chunk dimensions and domain decomposition parameters:
     * Divide the image into `num_streams` horizontal slabs.
     */
    int r_start[MAX_STREAMS];
    int r_end[MAX_STREAMS];
    int h_out[MAX_STREAMS];
    int r_in_start[MAX_STREAMS];
    int r_in_end[MAX_STREAMS];
    int h_in[MAX_STREAMS];

    size_t in_offset_bytes[MAX_STREAMS];
    size_t in_size_bytes[MAX_STREAMS];
    size_t out_offset_bytes[MAX_STREAMS];
    size_t out_size_bytes[MAX_STREAMS];

    int chunk_nominal = height / num_streams;
    size_t max_in_bytes = 0;
    size_t max_out_bytes = 0;

    for (int i = 0; i < num_streams; i++) {
        r_start[i] = i * chunk_nominal;
        r_end[i] = (i == num_streams - 1) ? height : (i + 1) * chunk_nominal;
        h_out[i] = r_end[i] - r_start[i];

        /* Include 1-pixel upper and lower halo for 3x3 Sobel convolution */
        r_in_start[i] = (r_start[i] > 0) ? (r_start[i] - 1) : 0;
        r_in_end[i] = (r_end[i] < height) ? (r_end[i] + 1) : height;
        h_in[i] = r_in_end[i] - r_in_start[i];

        in_offset_bytes[i] = (size_t)r_in_start[i] * width * channels;
        in_size_bytes[i] = (size_t)h_in[i] * width * channels;

        out_offset_bytes[i] = (size_t)r_start[i] * width;
        out_size_bytes[i] = (size_t)h_out[i] * width;

        if (in_size_bytes[i] > max_in_bytes) max_in_bytes = in_size_bytes[i];
        if (out_size_bytes[i] > max_out_bytes) max_out_bytes = out_size_bytes[i];
    }

    /*
     * 5. Allocate per-stream Device buffers:
     * Each stream has its own independent working slab buffers in GPU global memory.
     */
    unsigned char *d_img[MAX_STREAMS];
    unsigned char *d_gray[MAX_STREAMS];
    unsigned char *d_edges[MAX_STREAMS];

    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaMalloc((void **)&d_img[i], in_size_bytes[i]));
        CUDA_CHECK(cudaMalloc((void **)&d_gray[i], (size_t)h_in[i] * width));
        CUDA_CHECK(cudaMalloc((void **)&d_edges[i], out_size_bytes[i]));
    }

    /* 6. Execution and Timing via CUDA Events */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 blockDim(16, 16);

    CUDA_CHECK(cudaEventRecord(start, 0));

    /*
     * PIPELINE DISPATCH:
     * Issue asynchronous commands into independent CUDA streams.
     * With Hyper-Q and concurrent copy engines:
     *   - Stream i+1 H2D transfer runs concurrently with Stream i Kernel Execution!
     *   - Stream i-1 D2H transfer runs concurrently with Stream i Kernel Execution!
     */
    for (int i = 0; i < num_streams; i++) {
        /* Step A: Async Host-to-Device transfer of chunk RGB slice (with halo) */
        CUDA_CHECK(cudaMemcpyAsync(
            d_img[i],
            h_img_pinned + in_offset_bytes[i],
            in_size_bytes[i],
            cudaMemcpyHostToDevice,
            streams[i]
        ));

        /* Step B: Grayscale kernel on chunk */
        dim3 grid_in((width + blockDim.x - 1) / blockDim.x,
                     (h_in[i] + blockDim.y - 1) / blockDim.y);
        grayscale_chunk_kernel<<<grid_in, blockDim, 0, streams[i]>>>(
            d_img[i], d_gray[i], width, h_in[i], channels
        );

        /* Step C: Sobel Edge Detection on chunk with halo resolution */
        dim3 grid_out((width + blockDim.x - 1) / blockDim.x,
                      (h_out[i] + blockDim.y - 1) / blockDim.y);
        sobel_chunk_kernel<<<grid_out, blockDim, 0, streams[i]>>>(
            d_gray[i], d_edges[i], width, height, r_start[i], h_out[i], (r_start[i] == 0)
        );

        /* Step D: Async Device-to-Host transfer of chunk Edges slice */
        CUDA_CHECK(cudaMemcpyAsync(
            h_edges_pinned + out_offset_bytes[i],
            d_edges[i],
            out_size_bytes[i],
            cudaMemcpyDeviceToHost,
            streams[i]
        ));

        /* Also transfer grayscale slice for saving output */
        int gray_local_offset = (r_start[i] == 0) ? 0 : width;
        CUDA_CHECK(cudaMemcpyAsync(
            h_gray_pinned + out_offset_bytes[i],
            d_gray[i] + gray_local_offset,
            out_size_bytes[i],
            cudaMemcpyDeviceToHost,
            streams[i]
        ));
    }

    /* Synchronize all streams to complete pipeline execution */
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double elapsed_s = elapsed_ms / 1000.0;

    /* 7. Save outputs */
    stbi_write_png(gray_out_path, width, height, 1, h_gray_pinned, width);
    stbi_write_png(edges_out_path, width, height, 1, h_edges_pinned, width);

    /* 8. Formatted Performance Report */
    printf("=== CUDA Streams Pipelined Image Filter ===\n");
    printf("Image dimensions    : %d x %d (%d channels)\n", width, height, channels);
    printf("GPU Device          : %s (%d Streaming Multiprocessors)\n", prop.name, prop.multiProcessorCount);
    printf("CUDA Streams        : %d concurrent streams\n", num_streams);
    printf("Memory Model        : Pinned Host (Page-Locked) + Asynchronous DMA\n");
    printf("Chunk size          : ~%d x %d pixels/stream (with 1-px halo)\n", width, chunk_nominal);
    printf("Total Pipelined Time: %.6f s (Overlapped H2D + Compute + D2H)\n", elapsed_s);
    printf("Throughput          : %.4f img/s\n", 1.0 / elapsed_s);

    if (has_baseline && serial_baseline_time > 0.0) {
        double speedup = serial_baseline_time / elapsed_s;
        double efficiency_pct = (speedup / prop.multiProcessorCount) * 100.0;
        printf("Speedup vs Serial   : %.2fx (vs serial baseline %.6fs)\n", speedup, serial_baseline_time);
        printf("Efficiency per SM   : %.4f%% (%d SMs)\n", efficiency_pct, prop.multiProcessorCount);
    } else {
        printf("Speedup             : N/A (pass serial baseline seconds as arg to compute)\n");
        printf("Efficiency          : N/A\n");
    }

    printf("Pipelining Summary  : Overlapped %d transfers and kernel launches successfully.\n", num_streams);
    printf("Saved               : %s, %s\n", gray_out_path, edges_out_path);

    /* 9. Resource Cleanup */
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_img[i]);
        cudaFree(d_gray[i]);
        cudaFree(d_edges[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFreeHost(h_img_pinned);
    cudaFreeHost(h_gray_pinned);
    cudaFreeHost(h_edges_pinned);

    return EXIT_SUCCESS;
}
