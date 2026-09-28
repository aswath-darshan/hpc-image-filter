# Running CUDA Image Filter on Google Colab (GCP Cloud GPU)

This guide walks you through running the complete image filter benchmark on a **free NVIDIA T4 GPU** via Google Colab, which runs on **Google Cloud Platform (GCP)** infrastructure.

---

## Prerequisites

You need these files from the project folder:
- `serial_filter.c`
- `openmp_filter.c`
- `cuda_filter.cu`
- `cuda_filter_streams.cu`
- `stb_image.h`
- `stb_image_write.h`
- `test_input.png` (512×512 test image)
- `test_input_large.png` (4000×4000 test image)
- `cloud_gpu_benchmark.py` (the benchmark script)

---

## Step-by-Step Instructions

### Step 1: Open Google Colab

1. Go to **[colab.research.google.com](https://colab.research.google.com)**
2. Sign in with your Google account
3. Click **"New Notebook"**

### Step 2: Enable GPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU**
3. Click **Save**

> ⚠️ If T4 is not available, try **A100** or **L4** — any GPU works.

### Step 3: Upload Files

In the first cell, paste and run:

```python
from google.colab import files
uploaded = files.upload()
```

When the upload button appears, select **ALL** the files listed in Prerequisites above (you can select multiple files at once).

### Step 4: Run the Benchmark

In a new cell, paste and run:

```python
!python cloud_gpu_benchmark.py
```

This will:
1. ✅ Verify GPU availability (NVIDIA T4 on GCP)
2. ✅ Compile serial, OpenMP, CUDA standard, and **CUDA Streams (pipelined)**
3. ✅ Run benchmarks on both small and large images
4. ✅ Print a formatted comparison table including stream speedup
5. ✅ Save results to `cloud_gpu_results.txt`

### Step 5: Capture Results

1. **Screenshot the output** — this shows your CUDA and CUDA Streams code running on a cloud GPU
2. Download result files:

```python
from google.colab import files
files.download('cloud_gpu_results.txt')
files.download('edges_c_small.png')
files.download('edges_c_large.png')
files.download('edges_cs_large.png')
```

---

## Alternative: Cell-by-Cell Approach

If you prefer to run step by step (better for screenshots), use separate cells:

### Cell 1: Check GPU
```python
!nvidia-smi
```

### Cell 2: Upload files
```python
from google.colab import files
uploaded = files.upload()
```

### Cell 3: Compile
```python
!gcc -O2 -o serial_filter serial_filter.c -lm
!gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm
!nvcc -O2 -o cuda_filter cuda_filter.cu
!nvcc -O2 -o cuda_filter_streams cuda_filter_streams.cu
!echo "All 4 implementations compiled successfully!"
```

### Cell 4: Run Serial baseline (512×512)
```python
!./serial_filter test_input.png gray_s.png edges_s.png
```

### Cell 5: Run OpenMP (512×512)
```python
# Pass the Total time printed by Cell 4 as the last argument:
!./openmp_filter test_input.png gray_o.png edges_o.png 2 0.0038
```

### Cell 6: Run CUDA Standard (512×512)
```python
!./cuda_filter test_input.png gray_c.png edges_c.png 0.0038
```

### Cell 7: Run CUDA Streams Pipelined (512×512)
```python
# Syntax: ./cuda_filter_streams <input> <gray> <edges> <num_streams> <serial_baseline_s>
!./cuda_filter_streams test_input.png gray_cs.png edges_cs.png 4 0.0038
```

### Cell 8: Run Serial baseline (4000×4000)
```python
!./serial_filter test_input_large.png gray_s_large.png edges_s_large.png
```

### Cell 9: Run OpenMP (4000×4000)
```python
# Pass the Total time printed by Cell 8 as the last argument:
!./openmp_filter test_input_large.png gray_o_large.png edges_o_large.png 2 0.1945
```

### Cell 10: Run CUDA Standard (4000×4000)
```python
!./cuda_filter test_input_large.png gray_c_large.png edges_c_large.png 0.1945
```

### Cell 11: Run CUDA Streams Pipelined (4000×4000)
```python
!./cuda_filter_streams test_input_large.png gray_cs_large.png edges_cs_large.png 4 0.1945
```

---

## What to Tell Your Mentor / Professor

> "To address asynchronous execution and latency hiding in heterogeneous architectures, we implemented **CUDA Streams with Pinned Host Memory (`cudaMallocHost`) and Halo Exchange Domain Decomposition**. This overlaps Host-to-Device PCIe transfers, kernel execution on Streaming Multiprocessors, and Device-to-Host transfers across concurrent streams, hiding PCIe transfer bottleneck."

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No GPU detected" | Runtime → Change runtime type → GPU |
| "nvcc not found" | Run: `!apt-get install -y nvidia-cuda-toolkit` |
| Upload fails | Try uploading fewer files at a time |
| "GPU quota exceeded" | Wait a few minutes, or try a different Google account |
| "CUDA error" | Make sure GPU runtime is selected, try Runtime → Restart runtime |
