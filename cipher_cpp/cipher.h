#ifndef CIPHER_CPP_CIPHER_H_
#define CIPHER_CPP_CIPHER_H_

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

}



#endif