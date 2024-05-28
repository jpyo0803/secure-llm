#include "layer_struct_c.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "aes_stream.h"

void LS_SetLinearParams_I8I8I8I8(char* weight, char* bias, int M, int N,
                                 float alpha, float beta) {
  struct LinearParamsI8I8* params =
      (struct LinearParamsI8I8*)malloc(sizeof(struct LinearParamsI8I8));
  params->weight = (char*)malloc(M * N * sizeof(char));
  for (int i = 0; i < M * N; ++i) {
    params->weight[i] = weight[i];
  }
  params->bias = (char*)malloc(N * sizeof(char));
  for (int i = 0; i < N; ++i) {
    params->bias[i] = bias[i];
  }
  params->M = M;
  params->N = N;
  params->alpha = alpha;
  params->beta = beta;

  linear_params_i8i8i8i8_list[g_linear_i8i8i8i8_id++] = params;

  // TOOD(jpyo0803): Implement cleanup procedure
}

void LS_SetLinearParams_I8I8I8FP32(char* weight, char* bias, int M, int N,
                                   float alpha, float beta) {
  struct LinearParamsI8I8* params =
      (struct LinearParamsI8I8*)malloc(sizeof(struct LinearParamsI8I8));
  params->weight = (char*)malloc(M * N * sizeof(char));
  for (int i = 0; i < M * N; ++i) {
    params->weight[i] = weight[i];
  }
  params->bias = (char*)malloc(N * sizeof(char));
  for (int i = 0; i < N; ++i) {
    params->bias[i] = bias[i];
  }
  params->M = M;
  params->N = N;
  params->alpha = alpha;
  params->beta = beta;

  linear_params_i8i8i8fp32_list[g_linear_i8i8i8fp32_id++] = params;

  // TOOD(jpyo0803): Implement cleanup procedure
}

void LS_SetLinearParams_I8I8FP32FP32(char* weight, float* bias, int M, int N,
                                     float alpha, float beta) {
  struct LinearParamsI8FP32* params =
      (struct LinearParamsI8FP32*)malloc(sizeof(struct LinearParamsI8FP32));
  params->weight = (char*)malloc(M * N * sizeof(char));
  for (int i = 0; i < M * N; ++i) {
    params->weight[i] = weight[i];
  }
  params->bias = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    params->bias[i] = bias[i];
  }
  params->M = M;
  params->N = N;
  params->alpha = alpha;
  params->beta = beta;

  linear_params_i8i8fp32fp32_list[g_linear_i8i8fp32fp32_id++] = params;

  // TOOD(jpyo0803): Implement cleanup procedure
}

void LS_SetLayerNormParams(float* gamma, float* beta, int N, float eps) {
  struct LayerNormParams* params =
      (struct LayerNormParams*)malloc(sizeof(struct LayerNormParams));
  params->gamma = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    params->gamma[i] = gamma[i];
  }
  params->beta = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    params->beta[i] = beta[i];
  }

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

void LS_Round(float* x, int B, int M, int N) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] = roundf(x[i * M * N + j * N + k]);
      }
    }
  }
}

void LS_Clamp(float* x, int B, int M, int N, float min_val, float max_val) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] =
            fmin(max_val, fmax(min_val, x[i * M * N + j * N + k]));
      }
    }
  }
}

void LS_ResidualAdd(float* x, float* y, int B, int M, int N) {
// x += y
#pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] += y[i * M * N + j * N + k];
      }
    }
  }
}

struct TensorUint32* CreateTensorUint32(int B, int M, int N) {
  struct TensorUint32* tensor =
      (struct TensorUint32*)malloc(sizeof(struct TensorUint32));
  tensor->num_bytes = B * M * N * sizeof(unsigned int);
  tensor->data = (unsigned int*)malloc(tensor->num_bytes);
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorUint32(struct TensorUint32* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorFloat* CreateTensorFloat(int B, int M, int N) {
  struct TensorFloat* tensor =
      (struct TensorFloat*)malloc(sizeof(struct TensorFloat));
  tensor->num_bytes = B * M * N * sizeof(float);
  tensor->data = (float*)malloc(tensor->num_bytes);
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorFloat(struct TensorFloat* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorInt8* CreateTensorInt8(int B, int M, int N) {
  struct TensorInt8* tensor =
      (struct TensorInt8*)malloc(sizeof(struct TensorInt8));
  tensor->num_bytes = B * M * N * sizeof(char);
  tensor->data = (char*)malloc(tensor->num_bytes);
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorInt8(struct TensorInt8* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorFloat* hidden_states_internal = NULL;
void LS_SetHiddenStatesInternal(float* hidden_states, int B, int M, int N) {
  // Delete previous hidden states to prevent memory leak
  if (hidden_states_internal != NULL) {
    DeleteTensorFloat(hidden_states_internal);
  }

  hidden_states_internal = CreateTensorFloat(B, M, N);
#pragma omp parallel for
  for (int i = 0; i < B * M * N; ++i) {
    hidden_states_internal->data[i] = hidden_states[i];
  }
}

struct TensorFloat* residual1_internal = NULL;
void LS_CopyResidual1Internal() {
  if (residual1_internal != NULL) {
    DeleteTensorFloat(residual1_internal);
  }

  assert(hidden_states_internal != NULL);

  int B = hidden_states_internal->B;
  int M = hidden_states_internal->M;
  int N = hidden_states_internal->N;
  residual1_internal = CreateTensorFloat(B, M, N);

#pragma omp parallel for
  for (int i = 0; i < B * M * N; ++i) {
    residual1_internal->data[i] = hidden_states_internal->data[i];
  }
}

struct TensorInt8* self_attn_layer_norm_q_internal =
    NULL;  // output of layer norm
void LS_SelfAttnLayerNormQInternal(int layer_id) {
  assert(hidden_states_internal != NULL);

  int B = hidden_states_internal->B;
  int M = hidden_states_internal->M;
  int N = hidden_states_internal->N;

  LS_LayerNorm(hidden_states_internal->data, B, M, N, layer_id);
  LS_Round(hidden_states_internal->data, B, M, N);
  LS_Clamp(hidden_states_internal->data, B, M, N, -128, 127);

  if (self_attn_layer_norm_q_internal != NULL) {
    DeleteTensorInt8(self_attn_layer_norm_q_internal);
  }

  self_attn_layer_norm_q_internal =
      CreateTensorInt8(B, M, N);  // output of layer norm

#pragma omp parallel for
  for (int i = 0; i < B * M * N; ++i) {
    self_attn_layer_norm_q_internal->data[i] =
        (char)hidden_states_internal->data[i];
  }
}

void LS_GetSelfAttnLayerNormQInternal(char* q, int B, int M, int N) {
  assert(self_attn_layer_norm_q_internal != NULL);
  assert(B == self_attn_layer_norm_q_internal->B);
  assert(M == self_attn_layer_norm_q_internal->M);
  assert(N == self_attn_layer_norm_q_internal->N);

#pragma omp parallel for
  for (int i = 0; i < B * M * N; ++i) {
    q[i] = self_attn_layer_norm_q_internal->data[i];
  }
}

void LS_GetResidual1Internal(float* out, int B, int M, int N) {
  assert(residual1_internal != NULL);
  assert(B == residual1_internal->B);
  assert(M == residual1_internal->M);
  assert(N == residual1_internal->N);

#pragma omp parallel for
  for (int i = 0; i < B * M * N; ++i) {
    out[i] = residual1_internal->data[i];
  }
}

struct TensorUint32* hidden_states_blind_factor;

void LS_EncryptHiddenStates(float* x, int B, int M, int N) {
  hidden_states_blind_factor = CreateTensorUint32(B, M, N);
  GetCPRNG((unsigned char*)hidden_states_blind_factor->data,
           hidden_states_blind_factor->num_bytes);
}