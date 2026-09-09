"""
app.py
------
Flask backend for the deployed image-filter demo. Serves the frontend
(index.html) and exposes /process, which runs either the serial or OpenMP
binary on an uploaded image and returns the filtered result.

This is what runs on the Azure CPU VM. (The original filter_service.py,
which wraps the CUDA binary, stays as-is for reference/future GPU
deployment -- this app.py is the CPU-only version used for the live demo.)

Endpoints:
  GET  /            -> serves index.html (the demo frontend)
  GET  /health      -> liveness check
  POST /process     -> upload an image + method ('serial' or 'openmp'),
                        get back the edge-detected result + timing header

Requirements on the VM:
  pip3 install flask
  serial_filter and openmp_filter binaries must be compiled and present
  in the same directory as this script.

Run: python3 app.py   (listens on 0.0.0.0:5000)
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
}


@app.route("/", methods=["GET"])
def index():
    """Serve the demo frontend."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    status = {name: os.path.isfile(path) for name, path in BINARIES.items()}
    all_ok = all(status.values())
    return jsonify({"status": "ok" if all_ok else "missing_binaries", "binaries": status})


@app.route("/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file field in request"}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    method = request.form.get("method", "serial")
    if method not in BINARIES:
        return jsonify({"error": f"Unknown method '{method}', expected 'serial' or 'openmp'"}), 400

    binary_path = BINARIES[method]
    if not os.path.isfile(binary_path):
        return jsonify({"error": f"Binary for method '{method}' not found on server"}), 500

    request_id = uuid.uuid4().hex
    input_path = os.path.join(WORK_DIR, f"{request_id}_input.png")
    gray_path = os.path.join(WORK_DIR, f"{request_id}_gray.png")
    edges_path = os.path.join(WORK_DIR, f"{request_id}_edges.png")

    uploaded.save(input_path)

    # OpenMP binary takes an extra thread-count argument; serial doesn't
    cmd = [binary_path, input_path, gray_path, edges_path]
    if method == "openmp":
        cmd.append("4")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Filter binary failed", "stderr": e.stderr}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timed out"}), 504
    elapsed = time.time() - start

    response = send_file(edges_path, mimetype="image/png")
    response.headers["X-Processing-Time-Seconds"] = f"{elapsed:.6f}"
    response.headers["X-Method"] = method
    response.headers["Access-Control-Expose-Headers"] = "X-Processing-Time-Seconds, X-Method"

    for p in (input_path, gray_path, edges_path):
        try:
            os.remove(p)
        except OSError:
            pass

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
