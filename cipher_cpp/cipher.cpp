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



}
