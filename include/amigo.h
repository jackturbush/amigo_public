#ifndef AMIGO_H
#define AMIGO_H

#ifdef AMIGO_USE_CUDA
#define AMIGO_KERNEL __global__
#define AMGIO_DEVICE __device__
#define AMIGO_HOST_DEVICE __host__ __device___
#else
#define AMIGO_KERNEL
#define AMGIO_DEVICE
#define AMIGO_HOST_DEVICE
#endif  // AMIGO_USE_CUDA

#endif  // AMIGO_H