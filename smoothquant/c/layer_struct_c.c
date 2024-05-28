#include "layer_struct_c.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "aes_stream.h"

void LS_SetBmmParams(float alpha) { bmm_alpha_list[g_bmm_alpha_id++] = alpha; }

void LS_SetLinearParams_I8I8I8(char* weight, char* bias, int M, int N,
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

  linear_params_i8i8i8_list[g_linear_i8i8i8_id++] = params;

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

void LS_ResidualAdd1_Internal(float* hidden_states, int B, int M, int N) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        hidden_states[i * M * N + j * N + k] +=
            residual1_internal->data[i * M * N + j * N + k];
      }
    }
  }

  // also should set hidden states
  for (int i = 0; i < B * M * N; ++i) {
    hidden_states_internal->data[i] = hidden_states[i];
  }
}

void LS_Blind_Input_Op2_I8I8(int* x, int* y, int B, int M, int K, int N,
                             int blind_factor_id_u, int blind_factor_id_v) {
  if (blind_factor_list[blind_factor_id_u] != NULL) {
    DeleteTensorInt32(blind_factor_list[blind_factor_id_u]);
  }
  if (blind_factor_list[blind_factor_id_v] != NULL) {
    DeleteTensorInt32(blind_factor_list[blind_factor_id_v]);
  }

  blind_factor_list[blind_factor_id_u] = CreateTensorInt32(B, 1, K);
  GetDummyCPRNG((unsigned char*)blind_factor_list[blind_factor_id_u]->data,
                blind_factor_list[blind_factor_id_u]->num_bytes);

  blind_factor_list[blind_factor_id_v] = CreateTensorInt32(B, K, 1);
  GetDummyCPRNG((unsigned char*)blind_factor_list[blind_factor_id_v]->data,
                blind_factor_list[blind_factor_id_v]->num_bytes);

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < K; ++k) {
        x[i * M * K + j * K + k] +=
            blind_factor_list[blind_factor_id_u]->data[i * K + k];
      }
    }
  }

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < K; ++j) {
      for (int k = 0; k < N; ++k) {
        y[i * K * N + j * N + k] +=
            blind_factor_list[blind_factor_id_v]->data[i * K + j];
      }
    }
  }

  if (unblind_factor_xv_list[blind_factor_id_v] != NULL) {
    DeleteTensorInt32(unblind_factor_xv_list[blind_factor_id_v]);
  }
  if (unblind_factor_uy_list[blind_factor_id_u] != NULL) {
    DeleteTensorInt32(unblind_factor_uy_list[blind_factor_id_u]);
  }
  if (unblind_factor_uv_list[blind_factor_id_u] != NULL) {
    DeleteTensorInt32(unblind_factor_uv_list[blind_factor_id_u]);
  }

  unblind_factor_xv_list[blind_factor_id_v] =
      CreateTensorInt32(B, M, 1);  // (B x M x K) x (B x K x 1) = (B x M x 1)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      int sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += x[i * M * K + j * K + k] *
               blind_factor_list[blind_factor_id_v]->data[i * K + k];
      }
      unblind_factor_xv_list[blind_factor_id_v]->data[i * M + j] = sum;
    }
  }

  unblind_factor_uy_list[blind_factor_id_u] =
      CreateTensorInt32(B, 1, N);  // (B x 1 x K) x (B x K x N) = (B x 1 x N)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < N; ++j) {
      int sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += blind_factor_list[blind_factor_id_u]->data[i * K + k] *
               y[i * K * N + k * N + j];
      }
      unblind_factor_uy_list[blind_factor_id_u]->data[i * N + j] = sum;
    }
  }

  unblind_factor_uv_list[blind_factor_id_u] =
      CreateTensorInt32(B, 1, 1);  // (B x 1 x K) x (B x K x 1) = (B x 1 x 1)

  for (int i = 0; i < B; ++i) {
    int sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += blind_factor_list[blind_factor_id_u]->data[i * K + k] *
             blind_factor_list[blind_factor_id_v]->data[i * K + k];
    }
    unblind_factor_uv_list[blind_factor_id_u]->data[i] = sum;
  }
}

void LS_Unblind_Output_Op2_I8I8(int* x, int B, int M, int N,
                                int blind_factor_id_u, int blind_factor_id_v) {
  struct TensorInt32* unblind_factor_xv =
      unblind_factor_xv_list[blind_factor_id_v];
  struct TensorInt32* unblind_factor_uy =
      unblind_factor_uy_list[blind_factor_id_u];
  struct TensorInt32* unblind_factor_uv =
      unblind_factor_uv_list[blind_factor_id_u];  // just follow id_u for uv

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        int unblind_factor = unblind_factor_xv->data[i * M + j] +
                             unblind_factor_uy->data[i * N + k] +
                             unblind_factor_uv->data[i];
        x[i * M * N + j * N + k] -= unblind_factor;
      }
    }
  }
}
// In-place blind input
void LS_Blind_Input_Op1_I8I8I8(int* x, int B, int M, int N,
                               int blind_factor_id) {
  if (blind_factor_list[blind_factor_id] != NULL) {
    DeleteTensorInt32(blind_factor_list[blind_factor_id]);
  }

  blind_factor_list[blind_factor_id] = CreateTensorInt32(B, 1, N);
  GetCPRNG((unsigned char*)blind_factor_list[blind_factor_id]->data,
           blind_factor_list[blind_factor_id]->num_bytes);

  // for (int i = 0; i < 10; ++i) {
  //   printf("before blindfactor: %d\n",
  //   blind_factor_list[blind_factor_id]->data[i]);
  // }

#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        // printf("%d + %d\n", x[i * M * N + j * N + k],
        //        blind_factor_list[blind_factor_id]->data[i * N + k]);
        x[i * M * N + j * N + k] +=
            blind_factor_list[blind_factor_id]->data[i * N + k];
        // printf("res = %d\n", x[i * M * N + j * N + k]);
      }
    }
  }
}

void LS_Unblind_Output_Op1_I8I8I8(int* x, int B, int M, int N,
                                  int blind_factor_id, int linear_id) {
  struct TensorInt32* unblind_factor = CreateTensorInt32(B, 1, N);

  struct TensorInt32* blind_factor = blind_factor_list[blind_factor_id];  // 3D
  char* weight = linear_params_i8i8i8_list[linear_id]->weight;            // 2D

  // for (int i = 0; i < 10; ++i) {
  //   printf("after blindfactor: %d\n", blind_factor->data[i]);
  // }

  int prev_b = blind_factor->B;  // Must match 'B'
  int prev_m = blind_factor->M;  // should be 1
  int prev_k = blind_factor->N;

  int prev_n = linear_params_i8i8i8_list[linear_id]->N;

  // B == prev_b
  // prev_m == 1
  // N == prev_n

#pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < N; ++j) {
      int sum = 0;
      for (int k = 0; k < prev_k; ++k) {
        sum += blind_factor->data[i * prev_k + k] * weight[k * N + j];
      }
      // if (sum != 0) {
      //   printf("sum is not zero = %d\n", sum);
      //   exit(-1);
      // }
      unblind_factor->data[i * N + j] = sum;
    }
  }

#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        // printf("before x : %d\n", x[i * M * N + j * N + k]);
        // printf("unblind_factor : %d\n", unblind_factor->data[i * N + k]);
        x[i * M * N + j * N + k] -= unblind_factor->data[i * N + k];
        // if (unblind_factor->data[i * N + k] != 0) {
        //   printf("this is not zero = %d\n", unblind_factor->data[i * N +
        //   k]); exit(-1);
        // }
        // printf("after x : %d\n", x[i * M * N + j * N + k]);
        // exit(-1);
      }
    }
  }

  DeleteTensorInt32(unblind_factor);
}

void LS_ComputeEpilogue_I8I8I8(float* x, int B, int M, int N, int linear_id) {
  struct LinearParamsI8I8* params = linear_params_i8i8i8_list[linear_id];
  char* bias = params->bias;

#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] =
            x[i * M * N + j * N + k] * params->alpha + bias[k] * params->beta;
      }
    }
  }
}

void LS_Blind_Input_Op1_I8FP32FP32(int* x, int B, int M, int N,
                                   int blind_factor_id) {
  if (blind_factor_list[blind_factor_id] != NULL) {
    DeleteTensorInt32(blind_factor_list[blind_factor_id]);
  }

  blind_factor_list[blind_factor_id] = CreateTensorInt32(B, 1, N);
  GetCPRNG((unsigned char*)blind_factor_list[blind_factor_id]->data,
           blind_factor_list[blind_factor_id]->num_bytes);

#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] +=
            blind_factor_list[blind_factor_id]->data[i * N + k];
      }
    }
  }
}

void LS_Unblind_Output_Op1_I8FP32FP32(int* x, int B, int M, int N,
                                      int blind_factor_id, int linear_id) {
  struct TensorInt32* unblind_factor = CreateTensorInt32(B, 1, N);

  struct TensorInt32* blind_factor = blind_factor_list[blind_factor_id];  // 3D
  char* weight = linear_params_i8i8fp32fp32_list[linear_id]->weight;      // 2D

  int prev_b = blind_factor->B;  // Must match 'B'
  int prev_m = blind_factor->M;  // should be 1
  int prev_k = blind_factor->N;

  int prev_n = linear_params_i8i8fp32fp32_list[linear_id]->N;

  // B == prev_b
  // prev_m == 1
  // N == prev_n

#pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < N; ++j) {
      int sum = 0;
      for (int k = 0; k < prev_k; ++k) {
        sum += blind_factor->data[i * prev_k + k] * weight[k * N + j];
      }
      unblind_factor->data[i * N + j] = sum;
    }
  }

#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] -= unblind_factor->data[i * N + k];
      }
    }
  }

  DeleteTensorInt32(unblind_factor);
}

void LS_ComputeEpilogue_I8FP32FP32(float* x, int B, int M, int N,
                                   int linear_id) {
  struct LinearParamsI8FP32* params =
      linear_params_i8i8fp32fp32_list[linear_id];
  float* bias = params->bias;

#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] =
            x[i * M * N + j * N + k] * params->alpha + bias[k];
      }
    }
  }
}

void LS_ComputeEpilogue_BMM_I8I8(float* x, int B, int M, int N, int bmm_id) {
  float alpha = bmm_alpha_list[bmm_id];
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        x[i * M * N + j * N + k] *= alpha;
      }
    }
  }
}

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

struct TensorInt8* layer_norm_q_internal = NULL;  // output of layer norm
void LS_LayerNormQInternal(int layer_id) {
  assert(hidden_states_internal != NULL);

  int B = hidden_states_internal->B;
  int M = hidden_states_internal->M;
  int N = hidden_states_internal->N;

  LS_LayerNorm(hidden_states_internal->data, B, M, N, layer_id);
  LS_Round(hidden_states_internal->data, B, M, N);
  LS_Clamp(hidden_states_internal->data, B, M, N, -128, 127);

  if (layer_norm_q_internal != NULL) {
    DeleteTensorInt8(layer_norm_q_internal);
  }

  layer_norm_q_internal = CreateTensorInt8(B, M, N);  // output of layer norm

#pragma omp parallel for
  for (int i = 0; i < B * M * N; ++i) {
    layer_norm_q_internal->data[i] = (char)hidden_states_internal->data[i];
  }
}

void LS_GetLayerNormQInternal(char* q, int B, int M, int N) {
  assert(layer_norm_q_internal != NULL);
  assert(B == layer_norm_q_internal->B);
  assert(M == layer_norm_q_internal->M);
  assert(N == layer_norm_q_internal->N);

#pragma omp parallel for
  for (int i = 0; i < B * M * N; ++i) {
    q[i] = layer_norm_q_internal->data[i];
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