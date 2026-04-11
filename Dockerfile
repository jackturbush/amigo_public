# syntax=docker/dockerfile:1.7
# Multi-stage build for amigo.
#
# Stage 1 (builder): compiles the C++ extension into a wheel, using
# BuildKit cache mounts for ccache and pip so incremental rebuilds
# are fast even when layer caches are invalidated.
#
# Stage 2 (runtime): installs only runtime libraries and the built
# wheel, keeping the final image small and free of build tools.

ARG UBUNTU_VERSION=24.04
ARG A2D_REF=main

# -----------------------------------------------------------------------------
# Builder stage
# -----------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CCACHE_DIR=/root/.ccache \
    PATH=/usr/lib/ccache:$PATH \
    CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        cmake \
        git \
        gfortran \
        libopenmpi-dev \
        openmpi-bin \
        libopenblas-dev \
        liblapack-dev \
        libmumps-dev \
        python3 \
        python3-dev \
        python3-pip

WORKDIR /opt

ARG A2D_REF
RUN git clone https://github.com/smdogroup/a2d.git \
 && git -C a2d checkout ${A2D_REF}

WORKDIR /opt/amigo

# Install the Python build backend first so its layer is independent of
# the project source and only re-runs when pyproject.toml changes.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install scikit-build-core pybind11 mpi4py numpy

# Build wheels for amigo AND all its runtime dependencies. Deps with
# source-only releases (mpi4py) are compiled here against the builder's
# dev libraries; the runtime stage then installs everything from /wheels
# with --no-index, so it never needs compilers or -dev packages.
# ccache persists C++ object files across builds via the cache mount.
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    pip3 wheel --wheel-dir /wheels .

# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MUMPS_LIB_DIR=/usr/lib/x86_64-linux-gnu

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libopenmpi3 \
        libopenblas0 \
        liblapack3 \
        libmumps-dev \
        python3 \
        python3-pip

COPY --from=builder /wheels /wheels

# Install entirely from the local wheel directory. --no-index prevents
# pip from reaching PyPI, which guarantees the runtime never needs to
# compile anything and avoids pulling in -dev packages.
RUN pip3 install --no-index --find-links /wheels amigo \
 && rm -rf /wheels

WORKDIR /workspace
