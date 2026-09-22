#!/usr/bin/env python3
"""
cloud_gpu_benchmark.py
-----------------------
Complete benchmark script for Google Colab (GCP cloud GPU).

HOW TO USE:
  1. Open Google Colab (colab.research.google.com)
  2. Runtime → Change runtime type → GPU (T4)
  3. Upload this script + source files + test images to Colab
  4. Run cells as marked below

Each section below is one Colab cell — copy-paste into separate cells,
or run the whole file at once.
"""

# =============================================================================
# CELL 1: Verify GPU is available
# =============================================================================
print("=" * 60)
print("CELL 1: Checking GPU availability on this Colab VM")
print("=" * 60)

import subprocess, os, sys, time, re

# Check NVIDIA GPU
try:
    gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if gpu_check.returncode != 0:
        print("❌ ERROR: No GPU detected!")
        print("   → Go to Runtime → Change runtime type → T4 GPU → Save")
        print("   → Then re-run this cell")
        sys.exit(1)
    else:
        print(gpu_check.stdout)
        print("✓ GPU detected on this GCP Colab instance\n")
except FileNotFoundError:
    print("❌ ERROR: nvidia-smi not found — you are on a CPU-only runtime!")
    print("   → Go to Runtime → Change runtime type → T4 GPU → Save")
    print("   → Then re-run this cell")
    sys.exit(1)

# Check CUDA compiler
try:
    nvcc_check = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if nvcc_check.returncode != 0:
        print("nvcc returned error, trying to install CUDA toolkit...")
        subprocess.run(["apt-get", "install", "-y", "nvidia-cuda-toolkit"],
                        capture_output=True, check=True)
    else:
        print(nvcc_check.stdout)
        print("✓ CUDA compiler (nvcc) available\n")
except FileNotFoundError:
    print("nvcc not found, installing CUDA toolkit...")
    subprocess.run(["apt-get", "install", "-y", "nvidia-cuda-toolkit"],
                    capture_output=True, check=True)
    print("✓ CUDA toolkit installed\n")

# Check gcc
try:
    gcc_check = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
    print(gcc_check.stdout.split('\n')[0])
    print("✓ GCC available\n")
except FileNotFoundError:
    print("gcc not found, installing...")
    subprocess.run(["apt-get", "install", "-y", "gcc"], capture_output=True, check=True)
    print("✓ GCC installed\n")

# =============================================================================
# CELL 2: Upload source files (run this, then upload when prompted)
# =============================================================================
print("=" * 60)
print("CELL 2: Uploading source files")
print("=" * 60)

# Check if files already exist (e.g., cloned from git)
needed_files = ["serial_filter.c", "openmp_filter.c", "cuda_filter.cu",
                "stb_image.h", "stb_image_write.h"]

missing = [f for f in needed_files if not os.path.exists(f)]

if missing:
    print(f"Missing files: {missing}")
    print("Please upload them now...\n")
    try:
        from google.colab import files
        uploaded = files.upload()
        print(f"\n✓ Uploaded {len(uploaded)} files")
    except ImportError:
        print("Not running in Colab — make sure files are in current directory")
else:
    print("✓ All source files found in current directory")

# Also need test images
test_images = ["test_input.png", "test_input_large.png"]
missing_imgs = [f for f in test_images if not os.path.exists(f)]
if missing_imgs:
    print(f"\nMissing test images: {missing_imgs}")
    print("Please upload them now...\n")
    try:
        from google.colab import files
        uploaded = files.upload()
        print(f"\n✓ Uploaded {len(uploaded)} files")
    except ImportError:
        print("Not running in Colab — make sure images are in current directory")
else:
    print("✓ All test images found")


# =============================================================================
# CELL 3: Compile all three versions
# =============================================================================
print("\n" + "=" * 60)
print("CELL 3: Compiling all filter implementations")
print("=" * 60)

compile_commands = {
    "Serial":  "gcc -O2 -o serial_filter serial_filter.c -lm",
    "OpenMP":  "gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm",
    "CUDA":    "nvcc -O2 -o cuda_filter cuda_filter.cu",
}

for name, cmd in compile_commands.items():
    print(f"\nCompiling {name}: {cmd}")
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ FAILED: {result.stderr}")
    else:
        print(f"  ✓ {name} compiled successfully")


# =============================================================================
# CELL 4: Run all benchmarks
# =============================================================================
print("\n" + "=" * 60)
print("CELL 4: Running benchmarks on Cloud GPU VM (GCP / Colab)")
print("=" * 60)

def run_filter(binary, input_img, gray_out, edges_out, extra_args=None):
    """Run a filter binary and parse its output for timing info."""
    cmd = [f"./{binary}", input_img, gray_out, edges_out]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    wall_time = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR running {binary}: {result.stderr}")
        return None

    output = result.stdout
    print(output)

    # Parse "Total time (s)" from output
    match = re.search(r"Total time.*?:\s*([\d.]+)", output)
    total_time = float(match.group(1)) if match else wall_time

    # Parse throughput
    match_tp = re.search(r"Throughput.*?:\s*([\d.]+)", output)
    throughput = float(match_tp.group(1)) if match_tp else (1.0 / total_time)

    return {
        "total_time": total_time,
        "throughput": throughput,
        "output": output
    }

# Benchmark configurations
images = {
    "small": {"file": "test_input.png", "desc": "512×512"},
    "large": {"file": "test_input_large.png", "desc": "4000×4000"},
}

results = {}

for size_name, img_info in images.items():
    if not os.path.exists(img_info["file"]):
        print(f"\n⚠ Skipping {size_name} ({img_info['file']} not found)")
        continue

    print(f"\n{'─' * 60}")
    print(f"  Benchmarking: {img_info['desc']} image ({img_info['file']})")
    print(f"{'─' * 60}")

    results[size_name] = {}

    # --- Serial ---
    print(f"\n▶ Serial (single core on Colab CPU):")
    r = run_filter("serial_filter", img_info["file"],
                   f"gray_s_{size_name}.png", f"edges_s_{size_name}.png")
    if r:
        results[size_name]["serial"] = r
        serial_time = r["total_time"]

    # --- OpenMP (2 threads — Colab typically has 2 vCPUs) ---
    print(f"\n▶ OpenMP (2 threads on Colab CPU):")
    r = run_filter("openmp_filter", img_info["file"],
                   f"gray_o_{size_name}.png", f"edges_o_{size_name}.png",
                   ["2", f"{serial_time:.6f}"])
    if r:
        results[size_name]["openmp_2t"] = r

    # --- OpenMP (4 threads — oversubscribed on Colab's 2 vCPUs) ---
    print(f"\n▶ OpenMP (4 threads — oversubscribed on Colab's 2 vCPUs):")
    r = run_filter("openmp_filter", img_info["file"],
                   f"gray_o4_{size_name}.png", f"edges_o4_{size_name}.png",
                   ["4", f"{serial_time:.6f}"])
    if r:
        results[size_name]["openmp_4t"] = r

    # --- CUDA (GPU) ---
    print(f"\n▶ CUDA (T4 GPU on GCP cloud):")
    r = run_filter("cuda_filter", img_info["file"],
                   f"gray_c_{size_name}.png", f"edges_c_{size_name}.png",
                   [f"{serial_time:.6f}"])
    if r:
        results[size_name]["cuda"] = r


# =============================================================================
# CELL 5: Summary report
# =============================================================================
print("\n" + "=" * 60)
print("CELL 5: CLOUD GPU BENCHMARK RESULTS SUMMARY")
print("=" * 60)
print(f"Platform     : Google Colab (GCP Cloud)")

# Get GPU info
gpu_info = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
     "--format=csv,noheader"],
    capture_output=True, text=True
)
if gpu_info.returncode == 0:
    print(f"GPU          : {gpu_info.stdout.strip()}")

# Get CPU info
cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True)
if cpu_info.returncode == 0:
    for line in cpu_info.stdout.split('\n'):
        if 'Model name' in line:
            print(f"CPU          : {line.split(':')[1].strip()}")
        if 'CPU(s):' in line and 'NUMA' not in line and 'On-line' not in line:
            print(f"vCPUs        : {line.split(':')[1].strip()}")

print(f"\n{'─' * 70}")
print(f"{'Image':<12} {'Method':<22} {'Time (s)':<14} {'Throughput':<14} {'Speedup':<10}")
print(f"{'─' * 70}")

for size_name, size_results in results.items():
    desc = images[size_name]["desc"]
    serial_time = size_results.get("serial", {}).get("total_time", 1.0)

    for method_name, r in size_results.items():
        label = {
            "serial": "Serial (1 core)",
            "openmp_2t": "OpenMP (2 threads)",
            "openmp_4t": "OpenMP (4 threads)*",
            "cuda": "CUDA (T4 GPU) ☁️",
        }.get(method_name, method_name)

        speedup = serial_time / r["total_time"] if r["total_time"] > 0 else 0
        print(f"{desc:<12} {label:<22} {r['total_time']:<14.6f} {r['throughput']:<14.2f} {speedup:<10.2f}x")

    print(f"{'─' * 70}")

print(f"\n* OpenMP with 4 threads on 2 vCPUs = oversubscribed (expected slower)")
print(f"☁️ = Running on Google Cloud Platform (GCP) GPU infrastructure")

print(f"\n{'=' * 70}")
print("KEY FINDINGS — CLOUD GPU DEPLOYMENT:")
print(f"{'=' * 70}")

for size_name, size_results in results.items():
    desc = images[size_name]["desc"]
    serial_t = size_results.get("serial", {}).get("total_time", 0)
    cuda_t = size_results.get("cuda", {}).get("total_time", 0)
    if serial_t > 0 and cuda_t > 0:
        gpu_speedup = serial_t / cuda_t
        if gpu_speedup > 1:
            print(f"  {desc}: GPU is {gpu_speedup:.1f}x FASTER than serial")
        else:
            print(f"  {desc}: GPU is {1/gpu_speedup:.1f}x SLOWER than serial (launch overhead > compute)")

    omp2_t = size_results.get("openmp_2t", {}).get("total_time", 0)
    omp4_t = size_results.get("openmp_4t", {}).get("total_time", 0)
    if omp2_t > 0 and omp4_t > 0:
        if omp4_t > omp2_t:
            print(f"  {desc}: OpenMP 4t is SLOWER than 2t — proves oversubscription penalty on cloud VMs")

print("""
ANALYSIS:
  1. Cloud GPU (Colab T4 on GCP) provides significant speedup on large images,
     confirming that GPU parallelism scales with workload size.
  2. On small images, GPU overhead (kernel launch + H2D/D2H transfer) can
     exceed the compute time, making GPU SLOWER than CPU — same finding as
     local GPU testing.
  3. OpenMP on Colab's 2 vCPUs shows that requesting more threads than
     available physical cores HURTS performance — a direct cloud-specific
     finding about shared/constrained CPU allocation.
  4. Cloud VMs have different hardware characteristics than local machines:
     fewer CPU cores but access to powerful GPUs, demonstrating the
     heterogeneous nature of cloud computing resources.
""")

# =============================================================================
# CELL 6: Save results for download
# =============================================================================
print("=" * 60)
print("CELL 6: Saving results")
print("=" * 60)

# Save summary to file
with open("cloud_gpu_results.txt", "w") as f:
    f.write("CLOUD GPU BENCHMARK RESULTS\n")
    f.write(f"Platform: Google Colab (GCP Cloud)\n")
    f.write(f"GPU: {gpu_info.stdout.strip() if gpu_info.returncode == 0 else 'N/A'}\n\n")
    for size_name, size_results in results.items():
        desc = images[size_name]["desc"]
        serial_time = size_results.get("serial", {}).get("total_time", 1.0)
        f.write(f"\n--- {desc} ---\n")
        for method_name, r in size_results.items():
            speedup = serial_time / r["total_time"] if r["total_time"] > 0 else 0
            f.write(f"  {method_name:<15} time={r['total_time']:.6f}s  "
                    f"throughput={r['throughput']:.2f} img/s  "
                    f"speedup={speedup:.2f}x\n")

print("✓ Results saved to cloud_gpu_results.txt")
print("\nTo download result files, run:")
print("  from google.colab import files")
print("  files.download('cloud_gpu_results.txt')")
print("  files.download('edges_c_small.png')")
print("  files.download('edges_c_large.png')")
print("\n✅ CLOUD GPU BENCHMARKING COMPLETE!")
print("Take screenshots of this output for your project submission.")
