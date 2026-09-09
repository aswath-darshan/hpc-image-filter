#!/bin/bash
# deploy_azure_cpu.sh
# --------------------
# Deploys the web demo (app.py + index.html + serial/openmp filters) on a
# small, quota-free Azure CPU VM. Unlike deploy_azure.sh (which needs GPU
# quota approval), this uses a basic VM size available immediately on any
# new subscription, including Azure for Students.
#
# PREREQUISITES:
#   1. Signed up for Azure for Students (done)
#   2. Installed Azure CLI, logged in with `az login` (done)
#   3. Microsoft.Compute resource provider registered (done)
#   4. SSH key pair generated:  ssh-keygen -t rsa -b 4096 -f ~/.ssh/azure_vm_key
#
# USAGE:
#   Place this script in the same folder as: app.py, index.html,
#   serial_filter.c, openmp_filter.c, stb_image.h, stb_image_write.h
#   chmod +x deploy_azure_cpu.sh
#   ./deploy_azure_cpu.sh

set -e

# ---- CONFIGURATION ----
RESOURCE_GROUP="hpc-mini-project-rg"
LOCATION="centralindia"          # change if this region has issues; eastus is a safe fallback
VM_NAME="hpc-filter-cpu-vm"
VM_SIZE="Standard_DC2ads_v5"           # tiny, cheap, within default quota on any subscription
ADMIN_USERNAME="azureuser"
SSH_KEY_PATH="$HOME/.ssh/azure_vm_key.pub"
PROJECT_DIR="$(pwd)"
# ------------------------

echo "=== Step 1: Creating resource group '$RESOURCE_GROUP' in $LOCATION ==="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== Step 2: Creating CPU VM '$VM_NAME' (size: $VM_SIZE) ==="
az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --image "Ubuntu2204" \
  --size "$VM_SIZE" \
  --admin-username "$ADMIN_USERNAME" \
  --ssh-key-values "$SSH_KEY_PATH" \
  --public-ip-sku Standard

echo "=== Step 3: Opening port 5000 (web demo) ==="
az vm open-port --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --port 5000 --priority 900

VM_IP=$(az vm show -d --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --query publicIps -o tsv)
echo "VM public IP: $VM_IP"

echo "=== Step 4: Copying project files to the VM ==="
scp -i "${SSH_KEY_PATH%.pub}" -o StrictHostKeyChecking=no \
  "$PROJECT_DIR/app.py" \
  "$PROJECT_DIR/index.html" \
  "$PROJECT_DIR/serial_filter.c" \
  "$PROJECT_DIR/openmp_filter.c" \
  "$PROJECT_DIR/stb_image.h" \
  "$PROJECT_DIR/stb_image_write.h" \
  "$ADMIN_USERNAME@$VM_IP:~/"

echo "=== Step 5: Installing gcc + Flask, compiling, and starting the service ==="
ssh -i "${SSH_KEY_PATH%.pub}" -o StrictHostKeyChecking=no "$ADMIN_USERNAME@$VM_IP" << 'REMOTE_COMMANDS'
  sudo apt-get update
  sudo apt-get install -y build-essential python3-pip
  pip3 install flask
  gcc -O2 -o serial_filter serial_filter.c -lm
  gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm
  nohup python3 app.py > service.log 2>&1 &
  sleep 2
  echo "Service started. Health check:"
  curl -s http://localhost:5000/health
REMOTE_COMMANDS

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "Open this URL in your browser to see the live demo:"
echo "  http://$VM_IP:5000"
echo ""
echo "IMPORTANT: Deallocate the VM when done to avoid burning credit:"
echo "  az vm deallocate --resource-group $RESOURCE_GROUP --name $VM_NAME"
echo "  (or fully delete: az group delete --name $RESOURCE_GROUP --yes)"
