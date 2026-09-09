# Running cuda_filter.cu on Google Colab

Colab gives free GPU access with CUDA already installed — this is where you
should compile and test cuda_filter.cu FIRST, before any Azure deployment,
to confirm the code is correct.

## Steps

1. Go to https://colab.research.google.com and create a new notebook.
2. Enable GPU: Runtime menu -> Change runtime type -> Hardware accelerator -> GPU (T4) -> Save.
3. Upload these 5 files into the Colab session (folder icon on the left sidebar -> upload):
   - cuda_filter.cu
   - stb_image.h
   - stb_image_write.h
   - test_input.png
   - test_input_large.png   (optional, for the bigger benchmark)

4. In a Colab code cell, run:

   !nvcc -O2 -o cuda_filter cuda_filter.cu
   !./cuda_filter test_input.png gray_cuda.png edges_cuda.png

5. You should see output like:

   === CUDA Image Filter Pipeline ===
   Image size          : 512 x 512 (3 channels)
   Block size          : 16 x 16 threads
   Grid size           : 32 x 32 blocks
   Total time (s)      : 0.00XXXX  (includes H2D/D2H transfer)
   Throughput (img/s)  : XXXX.XXXX
   Saved: gray_cuda.png, edges_cuda.png

6. Correctness check — download gray_cuda.png and edges_cuda.png from Colab
   (right-click in file browser -> Download), then compare against your
   existing gray_serial.png / edges_serial.png visually, or run a hash
   comparison if you copy all files to the same machine:

     sha256sum edges_serial.png edges_cuda.png

   NOTE: unlike serial-vs-OpenMP, serial-vs-CUDA outputs may NOT be
   byte-identical due to floating point rounding differences between CPU
   and GPU (this is normal and expected, not a bug) — visually near-
   identical, with pixel value differences of at most 1-2 out of 255,
   is the correct outcome. If you see this, it's fine to report as
   "numerically equivalent" in your write-up.

7. Once confirmed correct, also run on test_input_large.png the same way
   and record the throughput — this becomes your "local/Colab GPU"
   benchmark number, to compare against the cloud-deployed Azure GPU
   number later.
