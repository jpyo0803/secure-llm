#ifndef CIPHER_CPP_CIPHER_H_
#define CIPHER_CPP_CIPHER_H_

#include <stdint.h>
#include <vector>
#include <cassert>
#include <omp.h>
#include <random>
#include <numeric>

namespace jpyo0803 {

template <typename T>
void SumByRow(T** in, T* out, int M, int N) {
  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    T sum = 0;
    for (int j = 0; j < N; ++j) {
      sum += in[i][j];
    }
    out[i] = sum;
  }
}

template <typename T>
void SumByCol(T** in, T* out, int M, int N) {
  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      #pragma omp atomic
      out[j] += in[i][j];
    }
  }
}

template <typename T>
void Shift(T*** in, T amt, int B, int M, int N) {
  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        in[i][j][k] += amt;
      }
    }
  }
}

template <typename T>
void UndoShift(T*** in, T amt, T K, T** row_sum_x, T** col_sum_y, int B, int M, int N) {
  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        in[i][j][k] -= amt * (row_sum_x[i][j] + col_sum_y[i][k] + K * amt);
      }
    }
  }
}

template <typename T>
void TensorAdd(T*** in1, T*** in2, int B, int M, int N) {
  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        in1[i][j][k] += in2[i][j][k];
      }
    }
  }
}

template <typename T>
void TensorScalarAdd(T*** in1, T scalar, int B, int M, int N) {
  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        in1[i][j][k] += scalar;
      }
    }
  }
}

template <typename T>
void TensorSub(T*** in1, T*** in2, int B, int M, int N) {
  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        in1[i][j][k] -= in2[i][j][k];
      }
    }
  }
}

template <typename T>
void TensorScalarSub(T*** in1, T scalar, int B, int M, int N) {
  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        in1[i][j][k] -= scalar;
      }
    }
  }
}

template<typename T>
std::vector<std::vector<T>> Matmul(const std::vector<std::vector<T>>& X, const std::vector<std::vector<T>>& Y) {
    assert(!X.empty() && !Y.empty());

    int rowsX = X.size();
    int colsX = X[0].size();
    int rowsY = Y.size();
    int colsY = Y[0].size();

    assert(colsX == rowsY);

    // Resultant matrix dimensions must be rowsX x colsY
    std::vector<std::vector<T>> Z(rowsX, std::vector<T>(colsY, 0));

    // Matrix multiplication
    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < rowsX; i++) {
        for (int j = 0; j < colsY; j++) {
            for (int k = 0; k < colsX; k++) {
                Z[i][j] += X[i][k] * Y[k][j];
            }
        }
    }

    return Z;
}

template <typename T>
T GenerateRandomNumber(T low, T high) {
  static thread_local std::random_device rd;  // Non-deterministic random device
  static thread_local std::mt19937 rng(rd()); // Random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<T> uni(low, high);
  return uni(rng);
}

template <typename T>
std::vector<std::vector<T>> RandInt2D(T low, T high, int M, int N) {
  std::vector<std::vector<T>> ret(M, std::vector<T>(N));

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      ret[i][j] = GenerateRandomNumber<T>(low, high);
    }
  }
  return ret;
}

std::vector<uint32_t> GenerateKeySetA(uint64_t mod, int n);

std::vector<std::vector<uint32_t>> FindKeyInvModFull(const std::vector<uint32_t>& a_x, const std::vector<uint32_t>& a_y);

std::vector<std::vector<std::vector<int32_t>>> AsTypeS8S32(char*** in, int B, int M, int N);

void EncryptTensorLight(uint32_t*** tensor, uint32_t a, uint32_t* b, bool vertical, int B, int M, int N);

void DecryptTensorLight(uint32_t*** tensor, uint32_t key_inv, uint32_t** dec_row_sum_x, uint32_t** dec_col_sum_y, uint32_t b_factor, int B, int M, int N);

void EncryptMatrix2D(std::vector<std::vector<uint32_t>>& in, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint64_t m, bool vertical);

void EncryptMatrix2DFull(std::vector<std::vector<uint32_t>>& in, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, bool vertical);

void DecryptMatrix2D(std::vector<std::vector<uint32_t>>& in, uint32_t K, const std::vector<uint32_t> a1, const std::vector<uint32_t> b1, const std::vector<uint32_t> a2, const std::vector<uint32_t> b2, uint64_t m, const std::vector<uint32_t> row_sum_x, const std::vector<uint32_t> col_sum_y);

void DecryptMatrix2DFull(std::vector<std::vector<uint32_t>>& in, const std::vector<std::vector<uint32_t>>& key_inv, const std::vector<uint32_t>& dec_row_sum_x, const std::vector<uint32_t>& dec_col_sum_y, uint32_t b_factor);
}

#endif