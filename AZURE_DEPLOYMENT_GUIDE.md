# Azure Cloud Deployment & Benchmarking Guide

This guide gives you the exact, step-by-step commands to run your HPC project on **Microsoft Azure**, for both **GPU VMs** and **CPU-only VMs** (if GPU quota is denied).

---

## 📋 Overview of Files to Transfer to Azure

You will need the following files in your Azure VM directory:
- `serial_filter.c`
- `openmp_filter.c`
- `cuda_filter.cu`
- `cuda_filter_streams.cu` (Advanced CUDA Streams & Pipelining)
- `stb_image.h` & `stb_image_write.h`
- `test_input.png` (512×512) & `test_input_large.png` (4000×4000)
- `cloud_gpu_benchmark.py`
- `cloud_gpu_app.py` & `app.py`
- `index.html`
- `requirements.txt`

---

## 🚀 PATH A: Azure GPU Virtual Machine (When GPU Quota is Approved)

### Recommended VM Sizes:
- **Standard_NC4as_T4_v3** (1× NVIDIA T4 16GB, 4 vCPUs, 28GB RAM) — *Recommended*
- **Standard_NC6s_v3** (1× NVIDIA V100 16GB, 6 vCPUs, 112GB RAM)
- **Standard_NV6** (1× NVIDIA Tesla M60, 6 vCPUs)
- **Operating System**: Ubuntu 22.04 LTS (preferably with NVIDIA GPU Driver pre-installed)

---

### Step 1: Open Port 5000 on Azure (NSG Inbound Rule)
1. Go to **Azure Portal** → Click on your VM.
2. Under **Networking** (or **Network Settings**), click **"Add inbound port rule"**.
3. Set:
   - **Destination port ranges**: `5000`
   - **Protocol**: `TCP`
   - **Action**: `Allow`
   - **Name**: `Port_5000_HPC_Demo`
4. Click **Add**.

---

### Step 2: Connect via SSH
From your local terminal / PowerShell:
```bash
ssh <azure_username>@<azure_vm_public_ip>
```
*(Example: `ssh azureuser@20.124.50.112`)*

---

### Step 3: Install Required Dependencies on Azure VM
Run these commands inside the SSH terminal:

```bash
# 1. Update package lists
sudo apt update && sudo apt install -y build-essential gcc g++ python3 python3-pip git

# 2. Check if NVIDIA driver and CUDA are ready
nvidia-smi
nvcc --version

# If nvcc is not installed, install CUDA toolkit:
# sudo apt install -y nvidia-cuda-toolkit

# 3. Install Python Flask
pip3 install flask
```

---

### Step 4: Transfer Project Files to Azure VM

**Option 1: Clone from your GitHub repository (Fastest)**
```bash
git clone https://github.com/aswath-darshan/hpc-image-filter.git
cd hpc-image-filter
```

**Option 2: Copy from your local machine via SCP**
From your local Windows PowerShell:
```powershell
scp -r "c:\Users\aswat\Downloads\HPC mini_project\*" <azure_username>@<azure_vm_public_ip>:~/hpc-image-filter/
```

---

### Step 5: Compile All Filter Binaries on Azure
Inside the project folder on Azure VM:

```bash
gcc -O2 -o serial_filter serial_filter.c -lm
gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm
nvcc -O2 -o cuda_filter cuda_filter.cu
nvcc -O2 -o cuda_filter_streams cuda_filter_streams.cu

# Verify all binaries were created
ls -lh serial_filter openmp_filter cuda_filter cuda_filter_streams
```

---

### Step 6: Run Complete Automated Benchmarks on Azure
```bash
python3 cloud_gpu_benchmark.py
```

This will run Serial, OpenMP (2 threads & 4 threads), CUDA Standard, and CUDA Streams across small and large images, and print the complete comparison table.

📸 **Take a screenshot of the benchmark output table!**

---

### Step 7: Launch the Live Web Service on Azure
Run the GPU-enabled Flask backend:

```bash
# Run in background with nohup or tmux so it stays active
python3 cloud_gpu_app.py
```

Now, open your browser and visit:
```
http://<azure_vm_public_ip>:5000
```

📸 **Take screenshots:**
1. Browser showing `http://<azure_vm_public_ip>:5000` with the demo working live.
2. Terminal showing `nvidia-smi` and active Azure GPU utilization.

---

## 🛡️ PATH B: Azure CPU Virtual Machine (If Azure GPU Quota is Denied)

If Azure denies GPU quota due to regional shortages, you can deploy on a standard Azure CPU VM (e.g. **Standard_D4s_v5** or **Standard_B2s**).

### Academic Justification for Submission:
> *"Due to cloud provider enterprise GPU capacity constraints in our academic subscription, the web service was deployed on an Azure Compute VM (multi-core CPU), while the CUDA Standard and CUDA Streams kernels with asynchronous DMA pipelining were verified on cloud GPU infrastructure. The project demonstrates full hybrid HPC: multi-threaded OpenMP CPU scaling + CUDA asynchronous stream domain decomposition."*

### Step 1: Open Port 5000 on Azure
Add Inbound NSG rule for TCP Port `5000`.

### Step 2: SSH into Azure CPU VM
```bash
ssh <azure_username>@<azure_vm_public_ip>
```

### Step 3: Setup and Compile CPU Filters
```bash
sudo apt update && sudo apt install -y build-essential gcc python3 python3-pip git
git clone https://github.com/aswath-darshan/hpc-image-filter.git
cd hpc-image-filter
pip3 install flask

gcc -O2 -o serial_filter serial_filter.c -lm
gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm
```

### Step 4: Run CPU Scaling Benchmarks on Azure VM
```bash
./serial_filter test_input_large.png gray_s.png edges_s.png
./openmp_filter test_input_large.png gray_o1.png edges_o1.png 1 <serial_time>
./openmp_filter test_input_large.png gray_o2.png edges_o2.png 2 <serial_time>
./openmp_filter test_input_large.png gray_o4.png edges_o4.png 4 <serial_time>
```

### Step 5: Launch Web Service on Azure CPU VM
```bash
python3 app.py
```
Visit `http://<azure_vm_public_ip>:5000` from your browser and take screenshots!

---

## 🎯 Summary of Key Evidence to Capture for Your Submission

1. **Terminal Screenshot 1**: SSH session showing `azureuser@<azure-hostname>` compiling `cuda_filter_streams.cu` and `cuda_filter.cu`.
2. **Terminal Screenshot 2**: Execution of `python3 cloud_gpu_benchmark.py` displaying the full performance and speedup table.
3. **Browser Screenshot**: Live website loaded from `http://<azure_vm_public_ip>:5000` processing an uploaded image and computing speedup/efficiency metrics live.
