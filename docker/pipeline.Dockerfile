# The regeneration side: train, evaluate, and re-render everything the demo
# shows. Profile-gated in docker-compose.yml, so `docker compose up` on a
# machine with no NVIDIA runtime still brings up the demo and simply never
# touches this image.
#
#   docker compose --profile pipeline build
#   docker compose --profile pipeline run --rm pipeline python demo/build_demo.py --all
#
# Base image: plain python-slim, NOT nvidia/cuda.
#
# The CUDA-tagged base images are the obvious choice and the wrong one here.
# Their tags are tied to a specific CUDA minor version that has to be matched
# against the host driver, they go stale, and a tag that no longer exists turns
# into a confusing pull failure months later. Modern PyTorch wheels ship their
# own CUDA runtime as nvidia-* pip dependencies, so the only thing the host
# must provide is the driver plus nvidia-container-toolkit -- which is exactly
# what the `deploy.resources.reservations.devices` block in docker-compose.yml
# requests. This pins CUDA in one place (the index URL below) instead of two.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libGL/libglib: opencv-python-headless still dlopens these for some codecs.
# git: a couple of pip builds want it on the PATH.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch first and on its own layer: it is by far the largest install, and
# keeping it separate means editing requirements-pipeline.txt does not force a
# multi-GB re-download.
#
# TORCH_CUDA matches the training machine (torch 2.13.0+cu130). It is an ARG so
# a host on an older driver can build with, say, cu121 without editing this
# file -- there is exactly one place to change it.
ARG TORCH_CUDA=cu130
ARG TORCH_VERSION=2.13.0
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# Source only. The data this operates on -- scenario_data/, data/, outputs/,
# real_bags/, the .ply geometry -- is mounted at run time (see the root
# .dockerignore, which keeps a 15.9 GB bag out of the build context).
COPY . .

# Fails the build rather than the first run if a dependency is missing. This is
# how requirements.txt was found to be incomplete: the imports below all come
# from code paths the demo actually exercises.
RUN python -c "import torch, numpy, pandas, scipy, PIL, matplotlib, cv2, \
pytorch_msssim, rosbags, imageio_ffmpeg, fast_simplification; \
print('pipeline deps OK; torch', torch.__version__)"

CMD ["python", "demo/build_demo.py", "--check"]
