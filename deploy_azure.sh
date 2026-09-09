#!/bin/bash
# deploy_azure.sh
# ----------------
# Provisions a GPU-enabled Azure VM and deploys the CUDA filter service
# onto it. Run this from YOUR OWN machine (WSL Ubuntu is fine) after you've
# completed the prerequisite steps below -- it cannot be run from inside
# this sandbox since Azure endpoints aren't reachable from here.
#
# PREREQUISITES (do these first -- see the numbered steps in chat):
#   1. Signed up for "Azure for Students" (azure.microsoft.com/free/students)
#   2. Installed Azure CLI locally:  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
#   3. Logged in:  az login   (opens a browser to authenticate)
#   4. Requested GPU quota for your chosen region (see chat instructions) --
#      this can take time to approve, so start it early.
#   5. Generated an SSH key pair if you don't have one:
#        ssh-keygen -t rsa -b 4096 -f ~/.ssh/azure_vm_key
#
# USAGE:
#   chmod +x deploy_azure.sh
#   ./deploy_azure.sh
#
# WHAT THIS SCRIPT DOES:
#   1. Creates a resource group
#   2. Creates a GPU VM (NC-series, has an NVIDIA T4 or similar)
#   3. Opens port 5000 (for the Flask API) and port 22 (SSH) in the NSG
#   4. Copies your project files to the VM
#   5. Installs CUDA toolkit + Python/Flask on the VM
#   6. Compiles cuda_filter.cu and starts filter_service.py on the VM
#
# After this completes, it prints the VM's public IP -- that's what you'll
# use to test the deployed service from your own machine.

set -e  # stop immediately if any command fails

# ---- CONFIGURATION: edit these before running ----
RESOURCE_GROUP="hpc-mini-project-rg"
LOCATION="eastus"                      # change to a region where your GPU quota was approved
VM_NAME="hpc-filter-gpu-vm"
VM_SIZE="Standard_NC4as_T4_v3"         # cheapest T4-GPU VM size on Azure
ADMIN_USERNAME="azureuser"
SSH_KEY_PATH="$HOME/.ssh/azure_vm_key.pub"
PROJECT_DIR="$(pwd)"                    # assumes you run this from inside image_pipeline/
# ---------------------------------------------------

echo "=== Step 1: Creating resource group '$RESOURCE_GROUP' in $LOCATION ==="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== Step 2: Creating GPU VM '$VM_NAME' (size: $VM_SIZE) ==="
az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --image "Ubuntu2204" \
  --size "$VM_SIZE" \
  --admin-username "$ADMIN_USERNAME" \
  --ssh-key-values "$SSH_KEY_PATH" \
  --public-ip-sku Standard

echo "=== Step 3: Opening port 5000 (Flask API) ==="
az vm open-port --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --port 5000 --priority 900

VM_IP=$(az vm show -d --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --query publicIps -o tsv)
echo "VM public IP: $VM_IP"

echo "=== Step 4: Installing NVIDIA driver + CUDA toolkit on the VM ==="
# Azure has an official VM extension that installs the NVIDIA GPU driver
az vm extension set \
  --resource-group "$RESOURCE_GROUP" \
  --vm-name "$VM_NAME" \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.9

echo "=== Step 5: Copying project files to the VM ==="
scp -i "${SSH_KEY_PATH%.pub}" -o StrictHostKeyChecking=no \
  "$PROJECT_DIR/cuda_filter.cu" \
  "$PROJECT_DIR/stb_image.h" \
  "$PROJECT_DIR/stb_image_write.h" \
  "$PROJECT_DIR/filter_service.py" \
  "$PROJECT_DIR/test_input.png" \
  "$PROJECT_DIR/test_input_large.png" \
  "$ADMIN_USERNAME@$VM_IP:~/"

echo "=== Step 6: Installing CUDA toolkit, Flask, and compiling on the VM ==="
ssh -i "${SSH_KEY_PATH%.pub}" -o StrictHostKeyChecking=no "$ADMIN_USERNAME@$VM_IP" << 'REMOTE_COMMANDS'
  sudo apt-get update
  sudo apt-get install -y nvidia-cuda-toolkit python3-pip
  pip3 install flask
  nvcc -O2 -o cuda_filter cuda_filter.cu
  nohup python3 filter_service.py > service.log 2>&1 &
  sleep 2
  echo "Service started. Health check:"
  curl -s http://localhost:5000/health
REMOTE_COMMANDS

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "VM public IP: $VM_IP"
echo "Test from your own machine with:"
echo "  curl -X POST -F \"image=@test_input.png\" http://$VM_IP:5000/process -o result_edges.png -w '\\nTotal request time: %{time_total}s\\n'"
echo ""
echo "IMPORTANT: Remember to deallocate/delete the VM when you're done"
echo "benchmarking, to avoid burning through your Azure credit:"
echo "  az vm deallocate --resource-group $RESOURCE_GROUP --name $VM_NAME"
echo "  (or fully delete: az group delete --name $RESOURCE_GROUP --yes)"
