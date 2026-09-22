/*
 * openmp_filter.c
 * ----------------
 * OpenMP (shared-memory parallel) implementation of grayscale conversion
 * and Sobel edge detection. Same algorithms as serial_filter.c; both
 * pixel loops are parallelized across CPU threads with
 * #pragma omp parallel for.
 *
 * Data-sharing notes (Unit 2):
 *   - input/output image buffers are SHARED across threads. This is safe
 *     because each thread only ever WRITES to pixel indices that belong
 *     to it (no two threads write the same output pixel) and only READS
 *     from the input buffer, which never changes during the parallel
 *     region. No locks or reductions are needed here, unlike the heat
 *     diffusion project, because pixel outputs don't feed into other
 *     pixels' outputs within the same pass ("embarrassingly parallel").
 *   - Loop variables (i, j, ki, kj, gx, gy, etc.) are private to each
 *     thread automatically inside an omp for loop.
 *
 * Compile:  gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm
 * Run:      ./openmp_filter <input_image> <gray_out> <edges_out> <num_threads>
 *           e.g. ./openmp_filter input.png gray_omp.png edges_omp.png 4
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void grayscale_omp(const unsigned char *input, unsigned char *output,
                    int width, int height, int channels) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            unsigned char r = input[idx + 0];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];
            output[i * width + j] =
                (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
}

void sobel_edge_omp(const unsigned char *input, unsigned char *output,
                     int width, int height) {
    int gx_kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int gy_kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    /* Zero out borders (cheap, do serially) */
    for (int j = 0; j < width; j++) {
        output[0 * width + j] = 0;
        output[(height - 1) * width + j] = 0;
    }
    for (int i = 0; i < height; i++) {
        output[i * width + 0] = 0;
        output[i * width + (width - 1)] = 0;
    }

    /* Parallelize the interior pixel loop across rows */
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int gx = 0, gy = 0;

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    unsigned char pixel = input[(i + ki) * width + (j + kj)];
                    gx += gx_kernel[ki + 1][kj + 1] * pixel;
                    gy += gy_kernel[ki + 1][kj + 1] * pixel;
                }
            }

            double magnitude = sqrt((double)(gx * gx + gy * gy));
            if (magnitude > 255.0) magnitude = 255.0;
            output[i * width + j] = (unsigned char)magnitude;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_image> <grayscale_out.png> <edges_out.png> [num_threads] [serial_baseline_seconds]\n", argv[0]);
        fprintf(stderr, "  serial_baseline_seconds: optional. Pass the 'Total time' printed by\n");
        fprintf(stderr, "  serial_filter on the SAME image to get Speedup/Efficiency computed here.\n");
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *gray_out_path = argv[2];
    const char *edges_out_path = argv[3];
    int num_threads = (argc >= 5) ? atoi(argv[4]) : 4;
    /* Optional 5th arg: the serial baseline's total time, for computing
       Speedup/Efficiency (Unit 1 performance metrics). If not provided,
       those lines are simply skipped in the output. */
    int has_baseline = (argc >= 6);
    double serial_baseline_time = has_baseline ? atof(argv[5]) : 0.0;

    omp_set_num_threads(num_threads);

    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        return EXIT_FAILURE;
    }
    channels = 3;

    unsigned char *gray = (unsigned char *) malloc(width * height);
    unsigned char *edges = (unsigned char *) malloc(width * height);
    if (!gray || !edges) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(img);
        return EXIT_FAILURE;
    }

    double t0 = omp_get_wtime();
    grayscale_omp(img, gray, width, height, channels);
    double t1 = omp_get_wtime();
    sobel_edge_omp(gray, edges, width, height);
    double t2 = omp_get_wtime();

    double gray_time = t1 - t0;
    double edge_time = t2 - t1;
    double total_time = gray_time + edge_time;

    stbi_write_png(gray_out_path, width, height, 1, gray, width);
    stbi_write_png(edges_out_path, width, height, 1, edges, width);

    printf("=== OpenMP Image Filter Pipeline ===\n");
    printf("Image size          : %d x %d (%d channels)\n", width, height, channels);
    printf("Threads used        : %d\n", num_threads);
    printf("Grayscale time (s)  : %.6f\n", gray_time);
    printf("Sobel edge time (s) : %.6f\n", edge_time);
    printf("Total time (s)      : %.6f\n", total_time);
    printf("Throughput (img/s)  : %.4f\n", 1.0 / total_time);

    /* --- Performance metrics (Unit 1: parallel algorithm performance) ---
       Speedup    S = T_serial / T_parallel
         "How many times faster is the parallel version than serial?"
       Efficiency E = S / P   (P = number of threads/processors used)
         "How well is each additional thread actually being used?"
         E close to 1.0 (100%) = near-perfect scaling.
         E well below 1.0 = overhead (thread creation, synchronization,
         memory contention) is eating into the theoretical speedup. */
    if (has_baseline && serial_baseline_time > 0.0) {
        double speedup = serial_baseline_time / total_time;
        double efficiency_pct = (speedup / num_threads) * 100.0;
        printf("Speedup             : %.2fx  (vs serial baseline %.6fs)\n", speedup, serial_baseline_time);
        printf("Efficiency          : %.2f%%  (%d threads)\n", efficiency_pct, num_threads);
    } else {
        printf("Speedup             : N/A (pass serial_filter's total time as 5th arg to compute)\n");
        printf("Efficiency          : N/A\n");
    }

    printf("Saved: %s, %s\n", gray_out_path, edges_out_path);

    stbi_image_free(img);
    free(gray);
    free(edges);
    return EXIT_SUCCESS;
}
