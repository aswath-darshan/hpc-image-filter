"""
filter_service.py
------------------
Lightweight Flask API that wraps the compiled CUDA filter binary
(cuda_filter) as a deployable "service" -- this is what gets deployed on
the Azure GPU VM. A client uploads an image via POST, the service runs the
CUDA binary on it, and returns the edge-detected result plus timing info.

This directly demonstrates the "deploy the CUDA version as a service on a
cloud VM" requirement from the project plan.

Endpoints:
  GET  /health           -> simple liveness check
  POST /process          -> upload an image, get back processed image + timing

Usage (once deployed on the VM, or for local testing):
  python3 filter_service.py
  (listens on 0.0.0.0:5000 by default)

Client-side test (from your own machine, pointed at the VM's public IP):
  curl -X POST -F "image=@test_input.png" http://<VM_PUBLIC_IP>:5000/process \
       -o result_edges.png -w "\nTotal request time: %{time_total}s\n"

Requirements on the VM:
  pip3 install flask
  cuda_filter binary must be compiled and present in the same directory
  (compiled via: nvcc -O2 -o cuda_filter cuda_filter.cu)
"""

import os
import subprocess
import time
import uuid
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

# Directory for temporary uploaded/processed files
WORK_DIR = "/tmp/filter_service"
os.makedirs(WORK_DIR, exist_ok=True)

# Path to the compiled CUDA binary -- must exist alongside this script,
# or be compiled at deployment time (see deploy_azure.sh)
CUDA_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_filter")


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check -- confirms the service is up and the CUDA
    binary exists, without actually running a GPU job."""
    binary_exists = os.path.isfile(CUDA_BINARY)
    return jsonify({
        "status": "ok" if binary_exists else "cuda_binary_missing",
        "cuda_binary_path": CUDA_BINARY,
    })


@app.route("/process", methods=["POST"])
def process_image():
    """
    Accepts a multipart/form-data upload with field name 'image'.
    Runs the CUDA filter binary on it, returns the edge-detection result
    as a PNG, with server-side processing time reported in a response header.
    """
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file field in request"}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Use a unique ID per request so concurrent requests don't collide
    request_id = uuid.uuid4().hex
    input_path = os.path.join(WORK_DIR, f"{request_id}_input.png")
    gray_path = os.path.join(WORK_DIR, f"{request_id}_gray.png")
    edges_path = os.path.join(WORK_DIR, f"{request_id}_edges.png")

    uploaded.save(input_path)

    # Time the actual CUDA subprocess call (this is the "GPU processing
    # time" portion; the Flask request/response overhead is separate and
    # can be measured client-side via curl's %{time_total}, giving you
    # both "pure compute time" and "end-to-end service time" to report).
    start = time.time()
    try:
        result = subprocess.run(
            [CUDA_BINARY, input_path, gray_path, edges_path],
            capture_output=True, text=True, timeout=60, check=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "CUDA binary failed",
            "stderr": e.stderr,
        }), 500
    except FileNotFoundError:
        return jsonify({"error": f"CUDA binary not found at {CUDA_BINARY}"}), 500
    elapsed = time.time() - start

    response = send_file(edges_path, mimetype="image/png")
    response.headers["X-Processing-Time-Seconds"] = f"{elapsed:.6f}"
    response.headers["X-CUDA-Stdout"] = result.stdout.replace("\n", " | ")

    # Clean up temp files after sending (best-effort)
    for p in (input_path, gray_path, edges_path):
        try:
            os.remove(p)
        except OSError:
            pass

    return response


if __name__ == "__main__":
    # 0.0.0.0 so it's reachable from outside the VM (not just localhost) --
    # required for the Azure NSG-opened port to actually reach this service.
    app.run(host="0.0.0.0", port=5000)
