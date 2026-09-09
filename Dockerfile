# Dockerfile
# ----------
# Builds a Linux container that compiles serial_filter.c and openmp_filter.c
# with gcc, then runs the Flask app via gunicorn (production WSGI server,
# recommended over Flask's built-in dev server for anything deployed).
#
# Render (and most PaaS providers) auto-detect a Dockerfile and build/deploy
# it directly -- no separate VM provisioning, no quota requests, no region
# policy walls. This is the practical benefit of PaaS vs IaaS for this
# project's deployment story.

FROM python:3.12-slim

# Install gcc and OpenMP support (libgomp1 is the OpenMP runtime library)
RUN apt-get update && \
    apt-get install -y gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all project files into the container
COPY . .

# Compile the C filters fresh, inside this Linux container --
# this is what avoids the Windows/WSL binary-compatibility issue entirely
RUN gcc -O2 -o serial_filter serial_filter.c -lm && \
    gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Render sets $PORT automatically; gunicorn binds to it
CMD gunicorn --bind 0.0.0.0:$PORT app:app
