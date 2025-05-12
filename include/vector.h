#ifndef MDGO_VECTOR_H
#define MDGO_VECTOR_H

namespace mdgo {

template <typename T, class Layout>
class Vector {
 public:
  Vector(const Layout &layout)
      : layout(layout), h_array(nullptr), d_array(nullptr) {}

  ~Vector() {
    if (d_array) {
      cudaFree(d_array);
    }
    if (h_array) {
      delete[] h_array;
    }
  }

  void host_initialize() {
    if (!h_array) {
      h_array = new T[layout.size()];
    }
  }

  void device_initialize() {
    if (!d_array) {
      cudaMalloc((void **)&d_array, layout.size() * sizeof(T));
    }
  }

  void host_to_device() {
    cudaMemcpy(d_array, h_array, layout.size() * sizeof(T),
               cudaMemcpyHostToDevice);
  }

  void device_to_host() {
    cudaMemcpy(h_array, d_array, layout.size() * sizeof(T),
               cudaMemcpyDeviceToHost);
  }

 private:
  const Layout &layout;
  T *h_array;  // Host data
  T *d_array;  // Device data
};

}  // namespace mdgo

#endif  // MDGO_VECTOR_H