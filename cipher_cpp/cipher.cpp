#include "cipher.h"
#include <iostream>
#include <cassert>
#include <vector>

// #define BitwiseMod(val, m) ((val) & ((m) - 1))

namespace {
inline int64_t BitwiseMod(int64_t val, uint64_t m) {
  // m must be 2^x
  return val & (m - 1);
}

template <typename T>
T RepeatedSqrMod(T base, T exp, T mod) {
  if (exp == 0) {
    return 1;
  }

  T res = RepeatedSqrMod(base, exp / 2, mod);
  res *= res;
  res %= mod;

  if (exp % 2) res *= base;
  return res % mod;
}


uint64_t FindKeyInvModNonPrime(uint64_t key, uint64_t mod) {
  assert(key >= 1);
  assert(mod >= 1);

  uint64_t phi = (1LL << 31);
  return RepeatedSqrMod<uint64_t>(key, phi - 1, mod);
}

}

namespace jpyo0803 {

std::vector<uint32_t> GenerateKeySetA(uint64_t mod, int n) {
  std::vector<uint32_t> key_set(n);
  for (int i = 0; i < n; ++i) {
    while (true) {
      auto x = GenerateRandomNumber<uint64_t>(1, mod - 1);
      if (std::gcd(x, mod) == 1) {
        key_set[i] = static_cast<uint32_t>(x);
        break;
      }
    }
  }
  return key_set;
}

std::vector<std::vector<std::vector<int32_t>>> AsTypeS8S32(char*** in, int B, int M, int N) {
  std::vector<std::vector<std::vector<int32_t>>> out(B, std::vector<std::vector<int32_t>>(M, std::vector<int32_t>(N)));

  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        out[i][j][k] = static_cast<int32_t>(in[i][j][k]);
      }
    }
  }
  return out;
}

void EncryptTensorLight(uint32_t*** tensor, uint32_t a, uint32_t* b, bool vertical, int B, int M, int N) {
  /*
    tensor: (B, M, N)
    a: (1,)
    b: (N,) if horizontal
    b: (M,) if vertical
  */
  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        tensor[i][j][k] *= a;
        tensor[i][j][k] += (vertical ? b[j]: b[k]);
      }
    }
  }
}


void DecryptTensorLight(uint32_t*** tensor, uint32_t key_inv, uint32_t** dec_row_sum_x, uint32_t** dec_col_sum_y, uint32_t b_factor, int B, int M, int N) {
  // tensor: (batch_size, M, N)
  // key_inv: (1,)
  // dec_row_sum_x: (batch-size, M)
  // dec_col_sum_y: (batch-size, N)
  // b_factor: (1, )

  #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        uint32_t tmp = tensor[i][j][k];
        tmp -= dec_row_sum_x[i][j];
        tmp -= dec_col_sum_y[i][k];
        tmp -= b_factor;
        tensor[i][j][k] = tmp * key_inv;
      }
    }
  }
}

void EncryptMatrix2D(std::vector<std::vector<uint32_t>>& in, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint64_t m, bool vertical = false) {
  int M = in.size();
  int N = in[0].size();

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      uint64_t tmp = static_cast<uint64_t>(in[i][j]);
      in[i][j] = static_cast<uint32_t>(BitwiseMod(tmp * a[(vertical ? j : i)] + b[(vertical ? j : i)], m));
    }
  }
}

void EncryptMatrix2DFull(std::vector<std::vector<uint32_t>>& in, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, bool vertical = false) {
  int M = in.size();
  int N = in[0].size();

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      in[i][j] = in[i][j] * a[vertical ? j : i] + b[vertical ? i : j];
    }
  }
}

void DecryptMatrix2D(std::vector<std::vector<uint32_t>>& in, uint32_t K, const std::vector<uint32_t> a1, const std::vector<uint32_t> b1, const std::vector<uint32_t> a2, const std::vector<uint32_t> b2, uint64_t m, const std::vector<uint32_t> row_sum_x, const std::vector<uint32_t> col_sum_y) {
  int M = in.size();
  int N = in[0].size();

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int64_t a1b2 = BitwiseMod(static_cast<int64_t>(a1[i]) * b2[j], m);
      int64_t a2b1 = BitwiseMod(static_cast<int64_t>(a2[j]) * b1[i], m);
      int64_t kb1b2 = BitwiseMod(BitwiseMod(static_cast<int64_t>(b1[i]) * b2[j], m) * K, m);

      uint64_t key_inv = FindKeyInvModNonPrime(a1[i] * a2[j], m); // For now we rely on this

      int64_t tmp = in[i][j];
      tmp -= BitwiseMod(BitwiseMod(a1b2 * row_sum_x[i] + a2b1 * col_sum_y[j], m) + kb1b2, m);   
      tmp *= key_inv;
      if (tmp < 0) {
          int64_t f = ((-tmp) / m) + 1;
          tmp += f * m;
      }
      tmp = BitwiseMod(tmp, m);  // This replaces the final BitwiseMod
      in[i][j] = static_cast<uint32_t>(tmp);
    }
  }
}

std::vector<std::vector<uint32_t>> FindKeyInvModFull(const std::vector<uint32_t>& a_x, const std::vector<uint32_t>& a_y) {
  int M = a_x.size();
  int N = a_y.size();

  std::vector<std::vector<uint32_t>> result(M, std::vector<uint32_t>(N));

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      result[i][j] = static_cast<uint32_t>(FindKeyInvModNonPrime(a_x[i] * a_y[j], (1LL << 32)));
    }
  }
  return result;
}

void DecryptMatrix2DFull(std::vector<std::vector<uint32_t>>& in, const std::vector<std::vector<uint32_t>>& key_inv, const std::vector<uint32_t>& dec_row_sum_x, const std::vector<uint32_t>& dec_col_sum_y, uint32_t b_factor) {
  int M = in.size();
  int N = in[0].size();

  constexpr uint64_t m = (1LL << 32);

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int64_t tmp = in[i][j];
      tmp -= dec_row_sum_x[i] + dec_col_sum_y[j];
      tmp -= b_factor;
      tmp *= key_inv[i][j];
      if (tmp < 0) {
        int64_t f = ((-tmp) / m) + 1;
        tmp += f * m;
      }
      tmp = BitwiseMod(tmp, m);
      in[i][j] = static_cast<uint32_t>(tmp);
    }
  }
}


}
