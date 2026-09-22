"""
cloud_gpu_app.py
-----------------
GPU-enabled Flask backend for the deployed image-filter demo.
Extends app.py to support CUDA (GPU) execution in addition to
serial and OpenMP methods.

This is designed for deployment on a cloud GPU instance using
Dockerfile.gpu (nvidia/cuda base image + --gpus all).

Endpoints:
  GET  /            -> serves index.html (the demo frontend)
  GET  /health      -> liveness check (includes GPU availability)
  POST /process     -> upload an image + method ('serial', 'openmp', or 'cuda'),
                       get back the edge-detected result + timing header

Requirements on the GPU VM:
  pip3 install flask
  serial_filter, openmp_filter, and cuda_filter binaries must be compiled
  and present in the same directory as this script.
  For CUDA: NVIDIA GPU + CUDA runtime must be available.

Run: python3 cloud_gpu_app.py   (listens on 0.0.0.0:5000)
"""

import os
import subprocess
import time
import uuid
from flask import Flask, request, send_file, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = "/tmp/filter_service"
os.makedirs(WORK_DIR, exist_ok=True)

BINARIES = {
    "serial": os.path.join(BASE_DIR, "serial_filter"),
    "openmp": os.path.join(BASE_DIR, "openmp_filter"),
    "cuda":   os.path.join(BASE_DIR, "cuda_filter"),
}


def check_gpu():
    """Check if an NVIDIA GPU is available on this machine."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


@app.route("/", methods=["GET"])
def index():
    """Serve the demo frontend."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Liveness check — reports binary availability and GPU status."""
    status = {name: os.path.isfile(path) for name, path in BINARIES.items()}
    gpu_name = check_gpu()
    all_ok = all(status.values())
    return jsonify({
        "status": "ok" if all_ok else "missing_binaries",
        "binaries": status,
        "gpu": gpu_name or "not available",
        "gpu_available": gpu_name is not None,
    })


@app.route("/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file field in request"}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    method = request.form.get("method", "serial")
    if method not in BINARIES:
        return jsonify({"error": f"Unknown method '{method}', expected 'serial', 'openmp', or 'cuda'"}), 400

    # Check GPU availability for CUDA method
    if method == "cuda" and check_gpu() is None:
        return jsonify({"error": "CUDA method requested but no GPU available on this server"}), 503

    binary_path = BINARIES[method]
    if not os.path.isfile(binary_path):
        return jsonify({"error": f"Binary for method '{method}' not found on server"}), 500

    request_id = uuid.uuid4().hex
    input_path = os.path.join(WORK_DIR, f"{request_id}_input.png")
    gray_path = os.path.join(WORK_DIR, f"{request_id}_gray.png")
    edges_path = os.path.join(WORK_DIR, f"{request_id}_edges.png")

    uploaded.save(input_path)

    # Build command based on method
    cmd = [binary_path, input_path, gray_path, edges_path]
    if method == "openmp":
        cmd.append("4")  # thread count
    # CUDA binary takes only 3 args (input, gray_out, edges_out)

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Filter binary failed", "stderr": e.stderr}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timed out"}), 504
    elapsed = time.time() - start

    response = send_file(edges_path, mimetype="image/png")
    response.headers["X-Processing-Time-Seconds"] = f"{elapsed:.6f}"
    response.headers["X-Method"] = method
    response.headers["X-GPU-Available"] = str(check_gpu() is not None)
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Processing-Time-Seconds, X-Method, X-GPU-Available"
    )

    for p in (input_path, gray_path, edges_path):
        try:
            os.remove(p)
        except OSError:
            pass

    return response


if __name__ == "__main__":
    gpu = check_gpu()
    print(f"GPU: {gpu or 'Not available'}")
    print(f"Available methods: {', '.join(k for k, v in BINARIES.items() if os.path.isfile(v))}")
    app.run(host="0.0.0.0", port=5000)
