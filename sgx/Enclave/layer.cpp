
#include "layer.h"
#include "Enclave.h"
#include <cmath>

// #include <omp.h>

extern "C" {
void LayerNorm(float* x, float* gamma, float* beta, float eps, int B, int M, int N) {
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = 0.0;
      float sum_sqr = 0.0;
      for (int k = 0; k < N; ++k) {
        float tmp = x[i * M * N + j * N + k];
        sum += tmp;
        sum_sqr += tmp * tmp;
      }
      float mean = sum / N;
      float var = sum_sqr / N - mean * mean;

      for (int k = 0; k < N; ++k) {
        float tmp = x[i * M * N + j * N + k];
        x[i * M * N + j * N + k] = (tmp - mean) / std::sqrt(var + eps) * gamma[k] + beta[k];
      }
    }
  } 
}

void ReLU(float* x, int B, int M, int N) {
  // #pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] = std::max(0.0f, x[i * M * N + j * N + k]);
      }
    }
  }
}

void Softmax(float* x, int B, int M, int N) {
  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      float max_val = x[i * M * N + j * N];
      for (int k = 1; k < N; ++k) {
        max_val = std::max(max_val, x[i * M * N + j * N + k]);
      }

      float sum = 0.0;
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] = std::exp(x[i * M * N + j * N + k] - max_val);
        sum += x[i * M * N + j * N + k];
      }

      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] /= sum;
      }
    }
  }
}

void ecall_layernorm(float* x, float* gamma, float* beta, float eps, int B, int M, int N) {
  LayerNorm(x,gamma,beta,eps,B,M,N);
}

void ecall_ReLU(float* x, int B, int M, int N) {
  ReLU(x,B,M,N);
}

void ecall_Softmax(float* x, int B, int M, int N) {
  Softmax(x,B,M,N);
}
}