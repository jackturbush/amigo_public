FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies: compilers, MPI, BLAS/LAPACK, CMake, Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenmpi-dev \
    openmpi-bin \
    libopenblas-dev \
    liblapack-dev \
    libmumps-dev \
    gfortran \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Allow pip install without venv (container is already isolated)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# MUMPS: tell amigo's ctypes loader where libdmumps.so lives
ENV MUMPS_LIB_DIR=/usr/lib/x86_64-linux-gnu

WORKDIR /opt

# Clone a2d (sibling directory expected by CMakeLists.txt)
RUN git clone --depth 1 https://github.com/smdogroup/a2d.git

# Copy amigo source
COPY . /opt/amigo

WORKDIR /opt/amigo

# Install Python build deps first (faster layer caching)
RUN pip3 install --no-cache-dir scikit-build-core pybind11 mpi4py numpy

# Install amigo (builds the C++ extension via scikit-build-core + CMake)
RUN pip3 install --no-cache-dir .

# Install remaining runtime deps not pulled in by the install
RUN pip3 install --no-cache-dir scipy matplotlib niceplots networkx pyvis tabulate icecream

WORKDIR /workspace
