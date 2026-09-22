/*
 * serial_filter.c
 * ----------------
 * Serial (single-threaded) baseline implementation of:
 *   1. Grayscale conversion  (embarrassingly parallel: each pixel independent)
 *   2. Sobel edge detection  (each pixel depends on its 3x3 neighborhood,
 *                             but pixels don't depend on each other's OUTPUT
 *                             -> still fully parallelizable)
 *
 * This is the baseline against which OpenMP and CUDA versions are
 * compared for speedup / throughput.
 *
 * Uses stb_image / stb_image_write (public domain, single-header) for
 * loading/saving PNG/JPG images -- no other dependencies needed.
 *
 * Compile:  gcc -O2 -o serial_filter serial_filter.c -lm
 * Run:      ./serial_filter <input_image> <grayscale_out> <edges_out>
 *           e.g. ./serial_filter input.png gray_serial.png edges_serial.png
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
 * Convert an RGB(A) image buffer to grayscale.
 * input:  width*height*channels bytes, interleaved (R,G,B[,A],R,G,B[,A],...)
 * output: width*height bytes, one grayscale value per pixel
 *
 * Formula: gray = 0.299*R + 0.587*G + 0.114*B
 * (weighted for human eye sensitivity -- green contributes most, blue least)
 */
void grayscale_serial(const unsigned char *input, unsigned char *output,
                       int width, int height, int channels) {
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

/*
 * Sobel edge detection on a grayscale image.
 * input:  width*height bytes, one grayscale value per pixel
 * output: width*height bytes, edge strength per pixel
 *
 * For each interior pixel, apply the horizontal (Gx) and vertical (Gy)
 * Sobel kernels to its 3x3 neighborhood, then combine:
 *   edge_strength = sqrt(Gx^2 + Gy^2), clamped to [0, 255]
 *
 * Border pixels (no full 3x3 neighborhood available) are set to 0.
 */
void sobel_edge_serial(const unsigned char *input, unsigned char *output,
                        int width, int height) {
    /* Sobel kernels */
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

    /* Zero out borders first */
    for (int j = 0; j < width; j++) {
        output[0 * width + j] = 0;
        output[(height - 1) * width + j] = 0;
    }
    for (int i = 0; i < height; i++) {
        output[i * width + 0] = 0;
        output[i * width + (width - 1)] = 0;
    }

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
        fprintf(stderr, "Usage: %s <input_image> <grayscale_out.png> <edges_out.png>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *gray_out_path = argv[2];
    const char *edges_out_path = argv[3];

    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        return EXIT_FAILURE;
    }
    channels = 3; /* we forced 3-channel (RGB) load above */

    unsigned char *gray = (unsigned char *) malloc(width * height);
    unsigned char *edges = (unsigned char *) malloc(width * height);
    if (!gray || !edges) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(img);
        return EXIT_FAILURE;
    }

    struct timespec t0, t1, t2;

    /* --- Grayscale timing --- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    grayscale_serial(img, gray, width, height, channels);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* --- Sobel edge detection timing --- */
    sobel_edge_serial(gray, edges, width, height);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    double gray_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double edge_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
    double total_time = gray_time + edge_time;

    stbi_write_png(gray_out_path, width, height, 1, gray, width);
    stbi_write_png(edges_out_path, width, height, 1, edges, width);

    printf("=== Serial Image Filter Pipeline ===\n");
    printf("Image size          : %d x %d (%d channels)\n", width, height, channels);
    printf("Grayscale time (s)  : %.6f\n", gray_time);
    printf("Sobel edge time (s) : %.6f\n", edge_time);
    printf("Total time (s)      : %.6f\n", total_time);
    printf("Throughput (img/s)  : %.4f\n", 1.0 / total_time);
    /* --- Performance metrics (Unit 1: parallel algorithm performance) ---
       Serial execution is the reference point every parallel version is
       measured against, so by definition:
         Speedup    = T_serial / T_serial = 1.00x
         Efficiency = Speedup / P (P=1 processor)  = 100%
       Printed here so the output format matches openmp_filter.c exactly,
       making it easy to line up results side by side in a report. */
    printf("Speedup             : 1.00x (baseline reference)\n");
    printf("Efficiency          : 100.00%% (1 core)\n");
    printf("Saved: %s, %s\n", gray_out_path, edges_out_path);

    stbi_image_free(img);
    free(gray);
    free(edges);
    return EXIT_SUCCESS;
}
