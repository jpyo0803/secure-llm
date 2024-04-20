#include "cipher.h"
#include <iostream>

namespace jpyo0803 {

void EncryptMatrix2D(uint32_t** in, uint32_t a, uint32_t b, uint64_t m, int M, int N) {
  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      uint64_t tmp = static_cast<uint64_t>(in[i][j]);
      in[i][j] = static_cast<uint32_t>((tmp * a + b) % m);
    }
  }
}

void DecryptMatrix2D(uint32_t** in, uint32_t K, uint32_t a1, uint32_t b1, uint32_t a2, uint32_t b2, uint32_t a_12_inv, uint64_t m, uint32_t* row_sum_x, uint32_t* col_sum_y, int M, int N) {
  int64_t a1b2 = (static_cast<int64_t>(a1) * b2) % m;
  int64_t a2b1 = (static_cast<int64_t>(a2) * b1) % m;
  int64_t kb1b2 = (((static_cast<int64_t>(b1) * b2) % m) * K) % m;

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int64_t tmp = in[i][j];
      tmp -= (((a1b2 * row_sum_x[i]) % m) + ((a2b1 * col_sum_y[j]) % m) + kb1b2) % m;   
      tmp *= a_12_inv;
      if (tmp < 0) {
          int64_t f = ((-tmp) / m) + 1;
          tmp += f * m;
      }
      tmp %= m;  // This replaces the final BitwiseMod
      in[i][j] = static_cast<uint32_t>(tmp);
    }
  }
}

}