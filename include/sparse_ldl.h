#ifndef AMIGO_SPARSE_LDL_H
#define AMIGO_SPARSE_LDL_H

#include "blas_interface.h"
#include "csr_matrix.h"

namespace amigo {

template <typename T>
class SparseLDL {
 public:
  SparseLDL(std::shared_ptr<CSRMat<T>> mat) : mat(mat) {}

 private:
  // The matrix
  std::shared_ptr<CSRMat<T>> mat;
};

}  // namespace amigo
#endif AMIGO_SPARSE_LDL_H