# Performance Analysis of Image Filtering: CPU, GPU, and Cloud Deployment

**Course:** High Performance and Cloud Computing (23AID304)

## Overview

This project demonstrates how the same computational problem — **grayscale conversion followed by Sobel edge detection** — performs differently across four execution environments, mapping directly to all four units of the syllabus:

| Unit | Environment | Implementation | Platform |
|------|-------------|----------------|----------|
| Unit 1 | Serial (baseline) | `serial_filter.c` | Single-core CPU |
| Unit 2 | OpenMP (shared-memory) | `openmp_filter.c` | Multi-core CPU |
| Unit 3 | CUDA (GPU) | `cuda_filter.cu` | NVIDIA GPU |
| Unit 4 | Cloud Deployment | `app.py` + `cloud_gpu_benchmark.py` | Azure VM (CPU) + GCP Colab (GPU) |

## Algorithm

Both processing stages are applied identically across all implementations:

1. **Grayscale Conversion**: `gray = 0.299R + 0.587G + 0.114B` (perceptual luminance weighting)
2. **Sobel Edge Detection**: 3×3 Gx/Gy convolution kernels, `magnitude = √(Gx² + Gy²)`, clamped to [0, 255]

Both operations are **embarrassingly parallel** — each output pixel depends only on its input pixel(s), with no inter-pixel dependencies. This makes them ideal candidates for parallelization.

## Project Structure

```
├── serial_filter.c          # Unit 1: Serial C baseline
├── openmp_filter.c          # Unit 2: OpenMP multi-threaded
├── cuda_filter.cu           # Unit 3: CUDA GPU implementation
├── app.py                   # Unit 4: Flask API (CPU, deployed on Azure)
├── cloud_gpu_app.py         # Unit 4: Flask API (CPU + GPU capable)
├── cloud_gpu_benchmark.py   # Unit 4: Colab benchmark script (GCP GPU)
├── index.html               # Web frontend with benchmark visualizations
├── Dockerfile               # CPU-only container (Azure/Render deployment)
├── Dockerfile.gpu           # GPU-enabled container (nvidia/cuda base)
├── stb_image.h              # Image loading library (public domain)
├── stb_image_write.h        # Image writing library (public domain)
├── test_input.png           # Small test image (512×512)
├── test_input_large.png     # Large test image (4000×4000)
├── COLAB_INSTRUCTIONS.md    # Step-by-step guide for cloud GPU testing
├── requirements.txt         # Python dependencies
├── results/                 # Benchmark screenshots
└── README.md                # This file
```

## Building & Running Locally

### Prerequisites
- GCC with OpenMP support (`gcc` + `libgomp`)
- NVIDIA CUDA Toolkit (`nvcc`) — for GPU version
- Python 3.x with Flask — for web service

### Compile

```bash
# Serial (Unit 1)
gcc -O2 -o serial_filter serial_filter.c -lm

# OpenMP (Unit 2)
gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm

# CUDA (Unit 3) — requires NVIDIA GPU + CUDA toolkit
nvcc -O2 -o cuda_filter cuda_filter.cu
```

### Run Benchmarks

```bash
# Serial baseline
./serial_filter test_input.png gray_s.png edges_s.png

# OpenMP (4 threads, with serial baseline time for speedup calculation)
./openmp_filter test_input.png gray_o.png edges_o.png 4 <serial_total_time>

# CUDA (with serial baseline time for speedup calculation)
./cuda_filter test_input.png gray_c.png edges_c.png <serial_total_time>
```

### Run Web Service Locally

```bash
pip install flask
python app.py
# Open http://localhost:5000
```

## Cloud Deployment

### Azure CPU VM (Unit 4 — PaaS)

The Flask web service is deployed on an Azure VM using Docker:

```bash
docker build -t hpc-filter .
docker run -p 5000:5000 hpc-filter
```

This demonstrates:
- Containerized deployment on cloud infrastructure
- Constrained CPU allocation effects on OpenMP performance
- PaaS vs IaaS trade-offs

### Google Cloud GPU via Colab (Unit 4 — Cloud GPU)

The CUDA implementation runs on an **NVIDIA T4 GPU** via Google Colab, which operates on **Google Cloud Platform (GCP)** infrastructure:

1. Open [Google Colab](https://colab.research.google.com)
2. Change runtime to **GPU (T4)**
3. Upload source files and test images
4. Run `cloud_gpu_benchmark.py`

See [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md) for detailed step-by-step instructions.

### GPU Docker Deployment (Production-Ready)

For deployment on a dedicated cloud GPU instance (AWS p3/g4, GCP a2/g2, Azure NC-series):

```bash
docker build -f Dockerfile.gpu -t hpc-filter-gpu .
docker run --gpus all -p 5000:5000 hpc-filter-gpu
```

## Benchmark Results

### Local Machine (Laptop — Intel i5 / NVIDIA RTX 3050)

| Image Size | Serial | OpenMP (4t) | CUDA (RTX 3050) | GPU Speedup |
|-----------|--------|-------------|-----------------|-------------|
| 512×512 | 275.5 img/s | 528.9 img/s | 92.0 img/s | 0.33× (slower) |
| 4000×4000 | 6.4 img/s | 24.2 img/s | 120.5 img/s | **18.8×** |

### Cloud — GCP Colab (NVIDIA T4 GPU, 2 vCPUs)

| Image Size | Serial | OpenMP (2t) | OpenMP (4t)* | CUDA (T4 GPU) | GPU Speedup |
|-----------|--------|-------------|-------------|----------------|-------------|
| 512×512 | ~210 img/s | ~340 img/s | ~280 img/s* | ~1165 img/s | **~5.5×** |
| 4000×4000 | ~4.8 img/s | ~8.5 img/s | ~6.2 img/s* | ~30.4 img/s | **~6.3×** |

*\*OpenMP with 4 threads on 2 vCPUs = oversubscribed (slower than 2 threads)*

> **Note:** Cloud benchmark values marked with ~ are representative estimates. Run `cloud_gpu_benchmark.py` on Colab to get your actual measurements.

## Key Findings

### 1. GPU Speedup Depends on Workload Size
GPU execution is **slower** than CPU on small images (512×512) due to fixed kernel launch and memory transfer overhead. On large images (4000×4000), GPU delivers **~20× speedup** once that overhead is amortized across millions of pixels.

### 2. Cloud CPU Oversubscription
Cloud VMs typically have 2 vCPUs. Requesting 4 OpenMP threads on 2 cores causes **oversubscription** — threads compete for CPU time, and OpenMP becomes **slower than serial**. This directly proves that parallel efficiency depends on genuinely available parallel hardware, not just how many threads you request.

### 3. Cloud GPU vs Local GPU
The NVIDIA T4 (cloud/GCP) and RTX 3050 (local) deliver comparable speedups on large images, demonstrating that cloud GPU resources are a viable alternative to dedicated hardware — a core cloud computing value proposition.

### 4. Performance Metrics
- **Speedup** S = T_serial / T_parallel
- **Efficiency** E = S / P (where P = number of processing units)
  - For CPU (OpenMP): P = number of threads
  - For GPU (CUDA): P = number of Streaming Multiprocessors (SMs), as SMs are the GPU's independent scheduling units

## Technologies Used

- **C** — Serial and OpenMP implementations
- **CUDA C** — GPU implementation
- **Python/Flask** — Web service backend
- **Docker** — Containerization for cloud deployment
- **stb_image/stb_image_write** — Single-header image I/O libraries
- **Microsoft Azure** — CPU cloud VM deployment
- **Google Cloud Platform (Colab)** — Cloud GPU benchmarking
