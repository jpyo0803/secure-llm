#include "layer_struct_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void LS_SetLayerNormParams(float* gamma, float* beta, int N, float eps) {
  struct LayerNormParams* params =
      (struct LayerNormParams*)malloc(sizeof(struct LayerNormParams));
  params->gamma = gamma;
  params->beta = beta;
  params->N = N;
  params->eps = eps;
  layer_norm_params_list[g_layer_id++] = params;

  // TOOD(jpyo0803): Implement cleanup procedure
}

void LS_LayerNorm(float* x, int B, int M, int N, int layer_id) {
#pragma omp parallel for collapse(2)
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
        x[i * M * N + j * N + k] =
            (tmp - mean) / sqrtf(var + layer_norm_params_list[layer_id]->eps) *
                layer_norm_params_list[layer_id]->gamma[k] +
            layer_norm_params_list[layer_id]->beta[k];
      }
    }
  }
}

void LS_ReLU(float* x, int B, int M, int N) {
#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] = fmax(0.0f, x[i * M * N + j * N + k]);
      }
    }
  }
}

void LS_Softmax(float* x, int B, int M, int N) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      float max_val = x[i * M * N + j * N];
      for (int k = 1; k < N; ++k) {
        max_val = fmax(max_val, x[i * M * N + j * N + k]);
      }
      float sum = 0.0;
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] = expf(x[i * M * N + j * N + k] - max_val);
        sum += x[i * M * N + j * N + k];
      }
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] /= sum;
      }
    }
  }
}