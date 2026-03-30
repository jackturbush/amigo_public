#ifndef AMIGO_BLAS_INTERFACE_H
#define AMIGO_BLAS_INTERFACE_H

#include <complex>

extern "C" {
// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void dsyrk_(const char* uplo, const char* trans, int* n, int* k,
                   double* alpha, double* a, int* lda, double* beta, double* c,
                   int* ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void dgemm_(const char* ta, const char* tb, int* m, int* n, int* k,
                   double* alpha, double* a, int* lda, double* b, int* ldb,
                   double* beta, double* c, int* ldc);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void dtpsv_(const char* uplo, const char* transa, const char* diag,
                   int* n, double* a, double* x, int* incx);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void dtptrs_(const char* uplo, const char* transa, const char* diag,
                    int* n, int* nrhs, double* a, double* b, int* ldb,
                    int* info);

// Factorization of packed storage matrices
extern void dpptrf_(const char* c, int* n, double* ap, int* info);

// Compute C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
extern void zsyrk_(const char* uplo, const char* trans, int* n, int* k,
                   std::complex<double>* alpha, std::complex<double>* a,
                   int* lda, std::complex<double>* beta,
                   std::complex<double>* c, int* ldc);

// Compute C := alpha*op( A )*op( B ) + beta*C,
extern void zgemm_(const char* ta, const char* tb, int* m, int* n, int* k,
                   std::complex<double>* alpha, std::complex<double>* a,
                   int* lda, std::complex<double>* b, int* ldb,
                   std::complex<double>* beta, std::complex<double>* c,
                   int* ldc);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void ztpsv_(const char* uplo, const char* transa, const char* diag,
                   int* n, std::complex<double>* a, std::complex<double>* x,
                   int* incx);

// Solve A*x = b or A^T*x = b where A is in packed format
extern void ztptrs_(const char* uplo, const char* transa, const char* diag,
                    int* n, int* nrhs, std::complex<double>* a,
                    std::complex<double>* b, int* ldb, int* info);

// Factorization of packed storage matrices
extern void zpptrf_(const char* c, int* n, std::complex<double>* ap, int* info);
}

namespace amigo {

template <typename T>
void blas_syrk(const char* uplo, const char* trans, int* n, int* k, T* alpha,
               T* a, int* lda, T* beta, T* c, int* ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_syrk only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_gemm(const char* ta, const char* tb, int* m, int* n, int* k, T* alpha,
               T* a, int* lda, T* b, int* ldb, T* beta, T* c, int* ldc) {
  if constexpr (std::is_same<T, double>::value) {
    dgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_gemm only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_tpsv(const char* uplo, const char* transa, const char* diag, int* n,
               T* a, T* x, int* incx) {
  if constexpr (std::is_same<T, double>::value) {
    dtpsv_(uplo, transa, diag, n, a, x, incx);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztpsv_(uplo, transa, diag, n, a, x, incx);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_tpsv only supports double and std::complex<double>");
  }
}

template <typename T>
void blas_tptrs(const char* uplo, const char* transa, const char* diag, int* n,
                int* nrhs, T* a, T* x, int* ldx, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dtptrs_(uplo, transa, diag, n, nrhs, a, x, ldx, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    ztptrs_(uplo, transa, diag, n, nrhs, a, x, ldx, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "blas_trsm only supports double and std::complex<double>");
  }
}

template <typename T>
void lapack_pptrf(const char* c, int* n, T* ap, int* info) {
  if constexpr (std::is_same<T, double>::value) {
    dpptrf_(c, n, ap, info);
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    zpptrf_(c, n, ap, info);
  } else {
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
        "lapack_pptrf only supports double and std::complex<double>");
  }
}

}  // namespace amigo

#endif  // AMIGO_BLAS_INTERFACE_H