#ifndef AMIGO_FIXED_VARIABLES_H
#define AMIGO_FIXED_VARIABLES_H

#include <memory>

#include "amigo.h"
#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

class FixedVariables {
 public:
  template <typename T>
  FixedVariables(std::shared_ptr<Vector<int>> dofs,
                 std::shared_ptr<CSRMat<T>> mat) {
    // Get the matrix data
    int nrows, ncols;
    mat->get_data(&nrows, &ncols, nullptr, nullptr, nullptr, nullptr);

    int n = nrows;
    if (ncols > nrows) {
      n = ncols;
    }

    // Get the indices corresponding to the boundary conditions
    int num_fixed = dofs->get_size();
    const int* dof_array = dofs->get_array();

    int* is_fixed = new int[n];
    std::fill(is_fixed, is_fixed + n, 0);
    for (int i = 0; i < num_fixed; i++) {
      // Get the global variable index
      int dof = dof_array[i];

      // Map to local variable
      int dof_local = dof;

      if (dof_local >= 0 && dof_local < n) {
        is_fixed[dof_local] = 1;
      }
    }

    // Find the column locations
    find_indices(is_fixed, mat);

    delete[] is_fixed;
  }

  /**
   * @brief Zero the components in a vector
   *
   * @param vec Zero the rows of the vector
   */
  template <ExecPolicy policy, typename T>
  void zero_rows(std::shared_ptr<Vector<T>> vec) {
    vec->template set_values<policy>(vec_zero_indices, T(0.0));
  }

  /**
   * @brief Zero the rows and the column positions
   *
   * @param mat The matrix to zero
   */
  template <ExecPolicy policy, typename T>
  void zero_rows_and_columns(std::shared_ptr<CSRMat<T>> mat) {
    mat->template set_values<policy>(mat_zero_indices, T(0.0));
    mat->template set_values<policy>(mat_diag_indices, T(1.0));
  }

  /**
   * @brief Copy the row and column indices to the device
   */
  void copy_host_to_device() {
    vec_zero_indices->copy_host_to_device();
    mat_zero_indices->copy_host_to_device();
    mat_diag_indices->copy_host_to_device();
  }

 private:
  /**
   * @brief Find the locations where to zero column entries within the matrix
   *
   * @param is_fixed Is the variable fixed? The array must be of length
   * max(nrows, ncols).
   * @param mat The matrix
   */
  template <typename T>
  void find_indices(const int is_fixed[], std::shared_ptr<CSRMat<T>> mat) {
    // Get the matrix data
    int nrows, ncols;
    const int *rowp, *cols;
    mat->get_data(&nrows, &ncols, nullptr, &rowp, &cols, nullptr);

    int nzeros = 0;
    int ndiag = 0;
    for (int i = 0; i < nrows; i++) {
      if (!is_fixed[i]) {
        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
          int j = cols[jp];
          if (is_fixed[j]) {
            nzeros++;
          }
        }
      } else if (is_fixed[i]) {
        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
          int j = cols[jp];
          if (i == j) {
            ndiag++;
          } else {
            nzeros++;
          }
        }
      }
    }

    vec_zero_indices = std::make_shared<Vector<int>>(ndiag);
    mat_zero_indices = std::make_shared<Vector<int>>(nzeros);
    mat_diag_indices = std::make_shared<Vector<int>>(ndiag);

    int* vec_array = vec_zero_indices->get_array();
    int* zero_array = mat_zero_indices->get_array();
    int* diag_array = mat_diag_indices->get_array();

    nzeros = 0;
    ndiag = 0;
    for (int i = 0; i < nrows; i++) {
      if (!is_fixed[i]) {
        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
          int j = cols[jp];
          if (is_fixed[j]) {
            zero_array[nzeros] = jp;
            nzeros++;
          }
        }
      } else if (is_fixed[i]) {
        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
          int j = cols[jp];
          if (i == j) {
            vec_array[ndiag] = i;
            diag_array[ndiag] = jp;
            ndiag++;
          } else {
            zero_array[nzeros] = jp;
            nzeros++;
          }
        }
      }
    }
  }

  std::shared_ptr<Vector<int>> vec_zero_indices;
  std::shared_ptr<Vector<int>> mat_zero_indices;
  std::shared_ptr<Vector<int>> mat_diag_indices;
};

}  // namespace amigo

#endif  // AMIGO_FIXED_VARIABLES_H