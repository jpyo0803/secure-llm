#include "layer_struct_c.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mod.h"
#include "aes_stream.h"

int Ex_Set_Hidden_States(float* hidden_states, int B, int M, int N) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  tensor_float_list[curr_id] =
      CreateTensorFloatFromData(hidden_states, B, M, N);
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Copy_Hidden_States(int src_id) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];

  tensor_float_list[curr_id] = CreateTensorFloatFromData(
      src_tensor->data, src_tensor->B, src_tensor->M, src_tensor->N);

  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

// Not offloaded so it does not need precomputation
int Ex_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps) {
  int curr_id = layer_norm_param_id;

  struct LayerNormParam* layer_norm_param =
      (struct LayerNormParam*)malloc(sizeof(struct LayerNormParam));
  layer_norm_param->gamma = CreateTensorFloatFromData(gamma, 1, 1, N);
  layer_norm_param->beta = CreateTensorFloatFromData(beta, 1, 1, N);
  layer_norm_param->eps = eps;

  layer_norm_param_list[curr_id] = layer_norm_param;
  layer_norm_param_id = (layer_norm_param_id + 1) % STATIC_LIST_LEN;
  return curr_id;
}

int Ex_Layer_Norm_Q(int src_id, int layer_norm_param_id) {
  int curr_id = tensor_int8_id;

  struct TensorFloat* src_tensor = tensor_float_list[src_id];
  struct LayerNormParam* layer_norm_param =
      layer_norm_param_list[layer_norm_param_id];

  struct TensorInt8* dst_tensor =
      CreateTensorInt8(src_tensor->B, src_tensor->M, src_tensor->N);

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      __m512 sum_vec = _mm512_setzero_ps();
      __m512 sum_sqr_vec = _mm512_setzero_ps();

      // Calculate sum and sum of squares
      for (int k = 0; k <= N - 16; k += 16) {
        __m512 src_vec =
            _mm512_loadu_ps(&src_tensor->data[i * M * N + j * N + k]);
        sum_vec = _mm512_add_ps(sum_vec, src_vec);
        sum_sqr_vec = _mm512_fmadd_ps(src_vec, src_vec, sum_sqr_vec);
      }

      // Horizontally add the elements in sum_vec and sum_sqr_vec
      float sum = 0.0, sum_sqr = 0.0;
      float sum_array[16], sum_sqr_array[16];
      _mm512_storeu_ps(sum_array, sum_vec);
      _mm512_storeu_ps(sum_sqr_array, sum_sqr_vec);
      for (int l = 0; l < 16; ++l) {
        sum += sum_array[l];
        sum_sqr += sum_sqr_array[l];
      }

      // Process any remaining elements
      for (int k = (N / 16) * 16; k < N; ++k) {
        float tmp = src_tensor->data[i * M * N + j * N + k];
        sum += tmp;
        sum_sqr += tmp * tmp;
      }

      float mean = sum / N;
      float var = sum_sqr / N - mean * mean;
      float inv_std = 1.0f / sqrtf(var + layer_norm_param->eps);

      __m512 mean_vec = _mm512_set1_ps(mean);
      __m512 inv_std_vec = _mm512_set1_ps(inv_std);
      __m512 gamma_vec, beta_vec;

      // Normalize and clamp
      for (int k = 0; k <= N - 16; k += 16) {
        __m512 src_vec =
            _mm512_loadu_ps(&src_tensor->data[i * M * N + j * N + k]);
        __m512 norm_vec = _mm512_sub_ps(src_vec, mean_vec);
        norm_vec = _mm512_mul_ps(norm_vec, inv_std_vec);

        gamma_vec = _mm512_loadu_ps(&layer_norm_param->gamma->data[k]);
        beta_vec = _mm512_loadu_ps(&layer_norm_param->beta->data[k]);

        norm_vec = _mm512_fmadd_ps(norm_vec, gamma_vec, beta_vec);
        norm_vec = _mm512_roundscale_ps(
            norm_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // Clamp between -128 and 127
        __m512i int_vec = _mm512_cvtps_epi32(norm_vec);
        int_vec = _mm512_max_epi32(int_vec, _mm512_set1_epi32(-128));
        int_vec = _mm512_min_epi32(int_vec, _mm512_set1_epi32(127));

        // Store the result as int8
        __m128i int8_vec = _mm512_cvtsepi32_epi8(int_vec);
        _mm_storeu_si128((__m128i*)&dst_tensor->data[i * M * N + j * N + k],
                         int8_vec);
      }

      // Process any remaining elements
      for (int k = (N / 16) * 16; k < N; ++k) {
        float tmp = src_tensor->data[i * M * N + j * N + k];
        tmp = (tmp - mean) * inv_std * layer_norm_param->gamma->data[k] +
              layer_norm_param->beta->data[k];
        tmp = roundf(tmp);

        // Clamp between -128 and 127
        if (tmp > 127.0) {
          tmp = 127.0;
        } else if (tmp < -128.0) {
          tmp = -128.0;
        }
        dst_tensor->data[i * M * N + j * N + k] = (char)tmp;
      }
    }
  }
  tensor_int8_list[curr_id] = dst_tensor;
  tensor_int8_id = (tensor_int8_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

// Need precomputation
int Ex_Set_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N,
                               float alpha, float beta) {
  int curr_id = linear_param_id;

  struct LinearParam* linear_param =
      (struct LinearParam*)malloc(sizeof(struct LinearParam));
  linear_param->weight = CreateTensorInt8FromData(weight, 1, M, N);
  linear_param->bias_int8 = CreateTensorInt8FromData(bias, 1, 1, M);
  linear_param->alpha = alpha;
  linear_param->beta = beta;
  linear_param->is_bias_fp32 = 0;
  linear_param->chosen_keys = NULL;

  // Generate blind factors
  linear_param->obfuscation_ratio =
      10;  // tentative, should be controllable outside
  linear_param->blind_factors_set =
      CreateTensorInt32(1, linear_param->obfuscation_ratio, N);
  for (int i = 0; i < linear_param->obfuscation_ratio; ++i) {
    GetCPRNG_ModP((unsigned char*)&linear_param->blind_factors_set->data[i * N],
             N * sizeof(int));
  }
  linear_param->precomputed_unblind_factors =
      MatmulS32S8S32_ModP(linear_param->blind_factors_set, linear_param->weight);
  // dimension should be (1, obfuscation_ratio, M)

  linear_param_list[curr_id] = linear_param;
  linear_param_id = (linear_param_id + 1) % STATIC_LIST_LEN;

  return curr_id;
}

int Ex_Set_Linear_Param_WS8BFP32(char* weight, float* bias, int M, int N,
                                 float alpha) {
  int curr_id = linear_param_id;

  struct LinearParam* linear_param =
      (struct LinearParam*)malloc(sizeof(struct LinearParam));
  linear_param->weight = CreateTensorInt8FromData(weight, 1, M, N);
  linear_param->bias_float = CreateTensorFloatFromData(bias, 1, 1, M);
  linear_param->alpha = alpha;
  linear_param->beta = 1.0;
  linear_param->is_bias_fp32 = 1;
  linear_param->chosen_keys = NULL;

  // Generate blind factors
  linear_param->obfuscation_ratio =
      10;  // tentative, should be controllable outside
  linear_param->blind_factors_set =
      CreateTensorInt32(1, linear_param->obfuscation_ratio, N);
  for (int i = 0; i < linear_param->obfuscation_ratio; ++i) {
    GetCPRNG_ModP((unsigned char*)&linear_param->blind_factors_set->data[i * N],
             N * sizeof(int));
  }
  linear_param->precomputed_unblind_factors =
      MatmulS32S8S32_ModP(linear_param->blind_factors_set, linear_param->weight);

  linear_param_list[curr_id] = linear_param;
  linear_param_id = (linear_param_id + 1) % STATIC_LIST_LEN;

  return curr_id;
}

void Ex_Get_Tensor_Dim_Int32(int src_id, int* dim) {
  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  dim[0] = src_tensor->B;
  dim[1] = src_tensor->M;
  dim[2] = src_tensor->N;
}

void Ex_Get_Tensor_Int32(int src_id, int* out) {
  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  int total_elements = src_tensor->B * src_tensor->M * src_tensor->N;

  int i;
  for (i = 0; i <= total_elements - 16; i += 16) {
    // Load 16 int32 elements from src_tensor
    __m512i src_vec = _mm512_loadu_si512((__m512i*)&src_tensor->data[i]);

    // Store 16 int32 elements to out
    _mm512_storeu_si512((__m512i*)&out[i], src_vec);
  }

  // Copy any remaining elements (if total_elements is not a multiple of 16)
  for (; i < total_elements; ++i) {
    out[i] = src_tensor->data[i];
  }
}

int Ex_Set_Tensor_Int32(int* data, int B, int M, int N) {
  int curr_id = tensor_int32_id;
  if (tensor_int32_list[curr_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[curr_id]);
  }

  tensor_int32_list[curr_id] = CreateTensorInt32FromData(data, B, M, N);
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int linear_param_id,
                                        int* out) {
  Ex_Get_Tensor_Int32(src_id, out);
  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  struct LinearParam* linear_param = linear_param_list[linear_param_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  // // // must choose random blind factors
  if (linear_param->chosen_keys != NULL) {
    DeleteTensorInt32(linear_param->chosen_keys);
  }
  linear_param->chosen_keys = CreateTensorInt32FromRandom(
      0, linear_param->obfuscation_ratio - 1, B, 1, M);

  int curr_id = tensor_int32_id;
  int blind_b = linear_param->blind_factors_set->B;
  int blind_m = linear_param->blind_factors_set->M;
  int blind_n = linear_param->blind_factors_set->N;

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      int chosen_value = linear_param->chosen_keys->data[i * M + j];
      int* blind_factor =
          &linear_param->blind_factors_set->data[chosen_value * N];

      for (int k = 0; k < N; ++k) {
        int64_t tmp = (int64_t)out[i * M * N + j * N + k] + (int64_t)blind_factor[k];
        out[i * M * N + j * N + k] =
            ModP(tmp);
      }
    }
  }
}

// depreciated
int Ex_Generate_Decryption_Key_Opr1_Int32(int blind_factor_id,
                                          int linear_param_id) {
  struct TensorInt32* blind_factor = tensor_int32_list[blind_factor_id];
  struct TensorInt8* linear_weight = linear_param_list[linear_param_id]->weight;

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }

  struct TensorInt32* decryption_key =
      MatmulS32S8S32(blind_factor, linear_weight);

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = decryption_key;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N,
                                       int linear_param_id) {
  struct LinearParam* linear_param = linear_param_list[linear_param_id];
  struct TensorInt32* chosen_keys = linear_param->chosen_keys;
  struct TensorInt32* unblind_factors =
      linear_param->precomputed_unblind_factors;
  int obf_ratio = linear_param->obfuscation_ratio;
  int bf_m = unblind_factors->M;
  int bf_n = unblind_factors->N;

  int chosen_key_n = chosen_keys->N;
  // dimension should be (1, obfuscation_ratio, M)

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      int chosen_value = linear_param->chosen_keys->data[i * M + j];
      int* unblind_factor = &unblind_factors->data[chosen_value * N];
      for (int k = 0; k < N; ++k) {
        int64_t tmp = (int64_t)data[i * M * N + j * N + k] -
                      (int64_t)unblind_factor[k];
        data[i * M * N + j * N + k] =
            ModP(tmp);
      }
    }
  }

  return Ex_Set_Tensor_Int32(data, B, M, N);
}

void Ex_Get_Encrypted_Tensor_QK_Int32(int src_id1, int src_id2, int* out1,
                                      int* out2, int* blind_factor_ids) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  struct TensorInt32* u = CreateTensorInt32(X->B, 1, X->N);
  GetCPRNG_ModP((unsigned char*)u->data, u->num_bytes);

  struct TensorInt32* v = CreateTensorInt32(X->B, 1, Y->N);
  GetCPRNG_ModP((unsigned char*)v->data, v->num_bytes);

  // Encrypt X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        int64_t tmp = (int64_t)X->data[i * X->M * X->N + j * X->N + k] + (int64_t)u->data[i * X->N + k];
        out1[i * X->M * X->N + j * X->N + k] =
            ModP(tmp);
      }
    }
  }

  // Encrypt Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        int64_t tmp = (int64_t)Y->data[i * Y->M * Y->N + j * Y->N + k] + (int64_t)v->data[i * Y->N + k];
        out2[i * Y->M * Y->N + j * Y->N + k] =
            ModP(tmp);
      }
    }
  }

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  tensor_int32_list[tensor_int32_id] = u;
  blind_factor_ids[0] = tensor_int32_id;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  tensor_int32_list[tensor_int32_id] = v;
  blind_factor_ids[1] = tensor_int32_id;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
}

int Ex_Generate_Decryption_Key_QK_Int32(int src_id1, int src_id2,
                                        int blind_factor_u_id,
                                        int blind_factor_v_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N

  struct TensorInt32* u = tensor_int32_list[blind_factor_u_id];
  struct TensorInt32* v = tensor_int32_list[blind_factor_v_id];

  struct TensorInt32* uy = MatmulS32S32S32_ModP(u, Y);  // B, 1, Y_M
  struct TensorInt32* xv = MatmulS32S32S32_ModP(X, v);  // B, X_M, 1

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }

  struct TensorInt32* uv = MatmulS32S32S32_ModP(u, v);  // B x 1 x 1

  struct TensorInt32* decryption_key = CreateTensorInt32(X->B, X->M, Y->M);

  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < Y->M; ++k) {
        int64_t tmp = (int64_t)uv->data[i] + (int64_t)xv->data[i * X->M + j] + (int64_t)uy->data[i * Y->M + k];
        decryption_key->data[i * X->M * Y->M + j * Y->M + k] =
            ModP(tmp);
      }
    }
  }

  DeleteTensorInt32(uy);
  DeleteTensorInt32(xv);
  DeleteTensorInt32(uv);

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = decryption_key;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Decrypted_Tensor_QK_Int32(int* data, int B, int M, int N,
                                     int decryption_key_id) {
  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);

  struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        int64_t tmp = (int64_t)data[i * M * N + j * N + k] - (int64_t)decryption_key->data[i * M * N + j * N + k];
        tensor->data[i * M * N + j * N + k] = ModP(tmp);
      }
    }
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Get_Encrypted_Tensor_PV_Int32(int src_id1, int src_id2, int* out1,
                                      int* out2, int* blind_factor_ids) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  struct TensorInt32* u = CreateTensorInt32(X->B, 1, X->N);
  GetCPRNG_ModP((unsigned char*)u->data, u->num_bytes);
  struct TensorInt32* v = CreateTensorInt32(X->B, 1, Y->M);
  GetCPRNG_ModP((unsigned char*)v->data, v->num_bytes);

  // Encrypt X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        int64_t tmp = (int64_t)X->data[i * X->M * X->N + j * X->N + k] + (int64_t)u->data[i * X->N + k];
        out1[i * X->M * X->N + j * X->N + k] =
            ModP(tmp);
      }
    }
  }

  // Encrypt Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        int64_t tmp = (int64_t)Y->data[i * Y->M * Y->N + j * Y->N + k] + (int64_t)v->data[i * Y->M + j];
        out2[i * Y->M * Y->N + j * Y->N + k] =
            ModP(tmp);
      }
    }
  }

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  tensor_int32_list[tensor_int32_id] = u;
  blind_factor_ids[0] = tensor_int32_id;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  tensor_int32_list[tensor_int32_id] = v;
  blind_factor_ids[1] = tensor_int32_id;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
}

int Ex_Generate_Decryption_Key_PV_Int32(int src_id1, int src_id2,
                                        int blind_factor_u_id,
                                        int blind_factor_v_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N

  struct TensorInt32* u = tensor_int32_list[blind_factor_u_id];
  struct TensorInt32* v = tensor_int32_list[blind_factor_v_id];

  struct TensorInt32* uy = MatmulS32S32S32_ModP(u, Y);  // B, 1, Y_M

  struct TensorInt32* xv = CreateTensorInt32(X->B, X->M, Y->M);

  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      int64_t sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += X->data[i * X->M * X->N + j * X->N + k];
        sum = ModP(sum);
      }

      for (int k = 0; k < Y->M; ++k) {
        xv->data[i * X->M * Y->M + j * Y->M + k] = ModP(sum * v->data[i * Y->M + k]);
      }
    }
  }

  struct TensorInt32* uv = CreateTensorInt32(X->B, 1, Y->M);
  for (int i = 0; i < X->B; ++i) {
    int64_t sum = 0;
    for (int j = 0; j < X->N; ++j) {
      sum += u->data[i * X->N + j];
      sum = ModP(sum);
    }

    for (int j = 0; j < Y->M; ++j) {
      uv->data[i * Y->M + j] = ModP(sum * v->data[i * Y->M + j]);
    }
  }

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }

  struct TensorInt32* decryption_key = CreateTensorInt32(X->B, X->M, Y->M);

  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < Y->M; ++k) {
        decryption_key->data[i * X->M * Y->M + j * Y->M + k] =
            ModP(uv->data[i * Y->M + k] + xv->data[i * X->M * Y->M + j * Y->M + k] +
            uy->data[i * Y->M + k]);
      }
    }
  }

  DeleteTensorInt32(uy);
  DeleteTensorInt32(xv);
  DeleteTensorInt32(uv);

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = decryption_key;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Decrypted_Tensor_PV_Int32(int* data, int B, int M, int N,
                                     int decryption_key_id) {
  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);

  struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        tensor->data[i * M * N + j * N + k] =
            data[i * M * N + j * N + k] -
            decryption_key->data[i * M * N + j * N + k];
        tensor->data[i * M * N + j * N + k] = ModP(tensor->data[i * M * N + j * N + k]);
      }
    }
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                   int* out1, int* out2,
                                                   int layer_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  // Generate blind factors if first time
  if (qk_blind_factor_u_list[layer_id] == NULL) {
    qk_blind_factor_u_list[layer_id] = CreateTensorInt32(X->B, 1, X->N);
    GetCPRNG_ModP((unsigned char*)qk_blind_factor_u_list[layer_id]->data,
             qk_blind_factor_u_list[layer_id]->num_bytes);

    qk_blind_factor_v_list[layer_id] = CreateTensorInt32(X->B, 1, Y->N);
    GetCPRNG_ModP((unsigned char*)qk_blind_factor_v_list[layer_id]->data,
             qk_blind_factor_v_list[layer_id]->num_bytes);
  }

  // Encrypt X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        int64_t tmp = (int64_t)X->data[i * X->M * X->N + j * X->N + k] +
                      (int64_t)qk_blind_factor_u_list[layer_id]->data[i * X->N + k];
        out1[i * X->M * X->N + j * X->N + k] =
            ModP(tmp);
        // out1[i * X->M * X->N + j * X->N + k] =
        //     X->data[i * X->M * X->N + j * X->N + k];
      }
    }
  }

  // Encrypt Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        int64_t tmp = (int64_t)Y->data[i * Y->M * Y->N + j * Y->N + k] +
                      (int64_t)qk_blind_factor_v_list[layer_id]->data[i * Y->N + k];
        out2[i * Y->M * Y->N + j * Y->N + k] =
            ModP(tmp);
        // out2[i * Y->M * Y->N + j * Y->N + k] =
        //     Y->data[i * Y->M * Y->N + j * Y->N + k];
      }
    }
  }
}

int Ex_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                     int layer_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N

  // printf("X: %d %d %d\n", X->B, X->M, X->N);
  // printf("Y: %d %d %d\n", Y->B, Y->M, Y->N);
  
  struct TensorInt32* uy =
      MatmulS32S32S32_ModP(qk_blind_factor_u_list[layer_id], Y);  // B, 1, Y_M
  struct TensorInt32* xv =
      MatmulS32S32S32_ModP(X, qk_blind_factor_v_list[layer_id]);  // B, X_M, 1

  // printf("uy shape %d %d %d\n", uy->B, uy->M, uy->N);
  // printf("xv shape %d %d %d\n", xv->B, xv->M, xv->N);

  if (qk_uv_dot_list[layer_id] == NULL) {
    qk_uv_dot_list[layer_id] =
        MatmulS32S32S32_ModP(qk_blind_factor_u_list[layer_id],
                        qk_blind_factor_v_list[layer_id]);  // B x 1 x 1
    qk_uy_unblind_factor_accum_list[layer_id] = uy;
  } else {
    // concat uy to qk_uy_unblind_factor_accum, to the last dimension of
    // qk_uy_unblind_factor_accum
    struct TensorInt32* new_uy =
        CreateTensorInt32(qk_uy_unblind_factor_accum_list[layer_id]->B, 1,
                          qk_uy_unblind_factor_accum_list[layer_id]->N + 1);
    // printf("qk uy shape %d %d %d\n", qk_uy_unblind_factor_accum_list[layer_id]->B, qk_uy_unblind_factor_accum_list[layer_id]->M, qk_uy_unblind_factor_accum_list[layer_id]->N);           
    // printf("new uy shape %d %d %d\n", uy->B, uy->M, uy->N);
    for (int i = 0; i < qk_uy_unblind_factor_accum_list[layer_id]->B; ++i) {
      for (int j = 0; j < qk_uy_unblind_factor_accum_list[layer_id]->N; ++j) {
        new_uy
            ->data[i * (qk_uy_unblind_factor_accum_list[layer_id]->N + 1) + j] =
            qk_uy_unblind_factor_accum_list[layer_id]
                ->data[i * qk_uy_unblind_factor_accum_list[layer_id]->N + j];
      }
      new_uy->data[i * (qk_uy_unblind_factor_accum_list[layer_id]->N + 1) +
                   qk_uy_unblind_factor_accum_list[layer_id]->N] = uy->data[i];
    }

    DeleteTensorInt32(qk_uy_unblind_factor_accum_list[layer_id]);
    qk_uy_unblind_factor_accum_list[layer_id] = new_uy;
    // printf("after make uy shape %d %d %d\n", qk_uy_unblind_factor_accum_list[layer_id]->B, qk_uy_unblind_factor_accum_list[layer_id]->M, qk_uy_unblind_factor_accum_list[layer_id]->N);
  }

  // Y->m , Dont use
  struct TensorInt32* decryption_key = CreateTensorInt32(X->B, X->M, qk_uy_unblind_factor_accum_list[layer_id]->N);
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < qk_uy_unblind_factor_accum_list[layer_id]->N; ++k) {
        int64_t tmp = (int64_t)qk_uv_dot_list[layer_id]->data[i] +
                      (int64_t)xv->data[i * X->M + j] +
                      (int64_t)qk_uy_unblind_factor_accum_list[layer_id]
                          ->data[i * qk_uy_unblind_factor_accum_list[layer_id]->N + k];
        decryption_key->data[i * X->M * qk_uy_unblind_factor_accum_list[layer_id]->N + j * qk_uy_unblind_factor_accum_list[layer_id]->N + k] =
            ModP(tmp);
      }
    }
  }
  // printf("decryption_key shape %d %d %d\n", decryption_key->B, decryption_key->M, decryption_key->N);

  int curr_id = tensor_int32_id;
  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  tensor_int32_list[curr_id] = decryption_key;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(int* data, int B, int M,
                                                  int N,
                                                  int decryption_key_id) {
  // printf("B %d M %d N %d\n", B, M, N);
  struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        int64_t tmp = (int64_t)data[i * M * N + j * N + k] -
                      (int64_t)decryption_key->data[i * M * N + j * N + k];
        tensor->data[i * M * N + j * N + k] =
            ModP(tmp);
      }
    }
  }

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                   int* out1, int* out2,
                                                   int layer_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  // Generate blind factors if first time
  if (pv_blind_factor_u_list[layer_id] == NULL) {
    pv_blind_factor_u_list[layer_id] = CreateTensorInt32(X->B, 1, X->N);
    GetCPRNG_ModP((unsigned char*)pv_blind_factor_u_list[layer_id]->data,
             pv_blind_factor_u_list[layer_id]->num_bytes);

    pv_blind_factor_v_list[layer_id] = CreateTensorInt32(X->B, 1, Y->M);
    GetCPRNG_ModP((unsigned char*)pv_blind_factor_v_list[layer_id]->data,
             pv_blind_factor_v_list[layer_id]->num_bytes);
  } else {
    // Copy all previous values of pv_blind_factor_u and append new random value
    struct TensorInt32* new_blind_factor_u =
        CreateTensorInt32(pv_blind_factor_u_list[layer_id]->B, 1,
                          pv_blind_factor_u_list[layer_id]->N + 1);
    int B = pv_blind_factor_u_list[layer_id]->B;
    int N = pv_blind_factor_u_list[layer_id]->N;

    for (int i = 0; i < B; ++i) {
      // Copy existing data
      for (int j = 0; j < N; ++j) {
        new_blind_factor_u->data[i * (N + 1) + j] =
            pv_blind_factor_u_list[layer_id]->data[i * N + j];
      }
      GetCPRNG_ModP((unsigned char*)&new_blind_factor_u->data[i * (N + 1) + N],
               sizeof(int));
    }

    DeleteTensorInt32(pv_blind_factor_u_list[layer_id]);
    pv_blind_factor_u_list[layer_id] = new_blind_factor_u;
  }

  // Encrypt X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        int64_t tmp = (int64_t)X->data[i * X->M * X->N + j * X->N + k] +
                      (int64_t)pv_blind_factor_u_list[layer_id]->data[i * X->N + k];
        out1[i * X->M * X->N + j * X->N + k] =
            ModP(tmp);
      }
    }
  }

  // Encrypt Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        int64_t tmp = (int64_t)Y->data[i * Y->M * Y->N + j * Y->N + k] +
                      (int64_t)pv_blind_factor_v_list[layer_id]->data[i * Y->M + j];
        out2[i * Y->M * Y->N + j * Y->N + k] =
            ModP(tmp);
      }
    }
  }
}

int Ex_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                     int layer_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N

  struct TensorInt32* xv = CreateTensorInt32(X->B, X->M, Y->M);

  // Calculate sum and multiply without vectorization
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      int64_t sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += X->data[i * X->M * X->N + j * X->N + k];
        sum = ModP(sum);
      }
      for (int k = 0; k < Y->M; ++k) {
        int64_t tmp = (int64_t)sum * (int64_t)pv_blind_factor_v_list[layer_id]->data[i * Y->M + k];
        xv->data[i * X->M * Y->M + j * Y->M + k] =
            ModP(tmp);
      }
    }
  }

  if (pv_uv_unblind_factor_accum_list[layer_id] == NULL) {
    pv_uy_unblind_factor_accum_list[layer_id] =
        MatmulS32S32S32_ModP(pv_blind_factor_u_list[layer_id], Y);  // B, 1, Y_M
    pv_uv_unblind_factor_accum_list[layer_id] =
        CreateTensorInt32(X->B, 1, Y->M);  // B, 1, Y_M

    for (int i = 0; i < X->B; ++i) {
      int64_t sum = 0;
      for (int j = 0; j < X->N; ++j) {
        sum += pv_blind_factor_u_list[layer_id]->data[i * X->N + j];
        sum = ModP(sum);
      }
      for (int j = 0; j < Y->M; ++j) {
        int64_t tmp = (int64_t)sum * (int64_t)pv_blind_factor_v_list[layer_id]->data[i * Y->M + j];
        pv_uv_unblind_factor_accum_list[layer_id]->data[i * Y->M + j] =
            ModP(tmp);
      }
    }
  } else {
    // Update pv_uy_unblind_factor_accum
    for (int i = 0; i < X->B; ++i) {
      int64_t u_last = pv_blind_factor_u_list[layer_id]->data[i * X->N + X->N - 1];
      for (int j = 0; j < Y->M; ++j) {
        int64_t tmp = (int64_t)u_last * (int64_t)Y->data[i * Y->M + j];
        pv_uy_unblind_factor_accum_list[layer_id]->data[i * Y->M + j] += ModP(tmp);
        pv_uy_unblind_factor_accum_list[layer_id]->data[i * Y->M + j] = ModP(pv_uy_unblind_factor_accum_list[layer_id]->data[i * Y->M + j]);

        tmp = (int64_t)u_last * (int64_t)pv_blind_factor_v_list[layer_id]->data[i * Y->M + j];
        pv_uv_unblind_factor_accum_list[layer_id]->data[i * Y->M + j] +=
            ModP(tmp);
        pv_uv_unblind_factor_accum_list[layer_id]->data[i * Y->M + j] = ModP(pv_uv_unblind_factor_accum_list[layer_id]->data[i * Y->M + j]);
      }
    }
  }

  struct TensorInt32* decryption_key = CreateTensorInt32(X->B, X->M, Y->M);
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < Y->M; ++k) {
        int64_t tmp = (int64_t)pv_uv_unblind_factor_accum_list[layer_id]->data[i * Y->M + k] +
                      (int64_t)xv->data[i * X->M * Y->M + j * Y->M + k] +
                      (int64_t)pv_uy_unblind_factor_accum_list[layer_id]->data[i * Y->M + k];
        decryption_key->data[i * X->M * Y->M + j * Y->M + k] =
            ModP(tmp);
      }
    }
  }

  int curr_id = tensor_int32_id;
  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  tensor_int32_list[curr_id] = decryption_key;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(int* data, int B, int M,
                                                  int N,
                                                  int decryption_key_id) {
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);
  struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];
  int total_elements = B * M * N;

  // Process data without vectorization
  for (int idx = 0; idx < total_elements; ++idx) {
    int64_t tmp = (int64_t)data[idx] - (int64_t)decryption_key->data[idx];
    tensor->data[idx] = ModP(tmp);
  }

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Pre_Init() {
  for (int i = 0; i < STATIC_LIST_LEN; ++i) {
    if (qk_blind_factor_u_list[i] != NULL) {
      DeleteTensorInt32(qk_blind_factor_u_list[i]);
      qk_blind_factor_u_list[i] = NULL;
    }
    if (qk_blind_factor_v_list[i] != NULL) {
      DeleteTensorInt32(qk_blind_factor_v_list[i]);
      qk_blind_factor_v_list[i] = NULL;
    }
    if (qk_uy_unblind_factor_accum_list[i] != NULL) {
      DeleteTensorInt32(qk_uy_unblind_factor_accum_list[i]);
      qk_uy_unblind_factor_accum_list[i] = NULL;
    }
    if (qk_uv_dot_list[i] != NULL) {
      DeleteTensorInt32(qk_uv_dot_list[i]);
      qk_uv_dot_list[i] = NULL;
    }
    if (pv_blind_factor_u_list[i] != NULL) {
      DeleteTensorInt32(pv_blind_factor_u_list[i]);
      pv_blind_factor_u_list[i] = NULL;
    }
    if (pv_blind_factor_v_list[i] != NULL) {
      DeleteTensorInt32(pv_blind_factor_v_list[i]);
      pv_blind_factor_v_list[i] = NULL;
    }
    if (pv_uy_unblind_factor_accum_list[i] != NULL) {
      DeleteTensorInt32(pv_uy_unblind_factor_accum_list[i]);
      pv_uy_unblind_factor_accum_list[i] = NULL;
    }
    if (pv_uv_unblind_factor_accum_list[i] != NULL) {
      DeleteTensorInt32(pv_uv_unblind_factor_accum_list[i]);
      pv_uv_unblind_factor_accum_list[i] = NULL;
    }
  }
}

int Ex_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id) {
  if (tensor_float_list[tensor_float_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[tensor_float_id]);
  }

  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  struct LinearParam* linear_param = linear_param_list[linear_param_id];

  float alpha = linear_param->alpha;

  float beta = linear_param->beta;
  struct TensorInt8* bias = linear_param->bias_int8;

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorFloat* dst_tensor = CreateTensorFloat(B, M, N);

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            alpha * src_tensor->data[i * M * N + j * N + k] +
            beta * bias->data[k];
      }
    }
  }

  int curr_id = tensor_float_id;
  tensor_float_list[curr_id] = dst_tensor;
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Compute_Epilogue_WS8BFP32(int src_id, int linear_param_id) {
  if (tensor_float_list[tensor_float_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[tensor_float_id]);
  }

  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  struct LinearParam* linear_param = linear_param_list[linear_param_id];

  float alpha = linear_param->alpha;

  float beta = linear_param->beta;
  struct TensorFloat* bias = linear_param->bias_float;

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorFloat* dst_tensor = CreateTensorFloat(B, M, N);

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            alpha * src_tensor->data[i * M * N + j * N + k] +
            beta * bias->data[k];
      }
    }
  }

  int curr_id = tensor_float_id;
  tensor_float_list[curr_id] = dst_tensor;
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Compute_Epilogue_BMM(int src_id, int bmm_param_id) {
  if (tensor_float_list[tensor_float_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[tensor_float_id]);
  }

  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  float alpha = bmm_param_list[bmm_param_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorFloat* dst_tensor = CreateTensorFloat(B, M, N);

  for (int i = 0; i < B * M * N; i++) {
    dst_tensor->data[i] = (float)src_tensor->data[i] * alpha;
  }

  int curr_id = tensor_float_id;
  tensor_float_list[curr_id] = dst_tensor;
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_ReLU(int src_id) {
  if (tensor_float_list[tensor_float_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[tensor_float_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorFloat* dst_tensor = CreateTensorFloat(B, M, N);

  int total_elements = B * M * N;
  int i;
  __m512 zero_vec = _mm512_setzero_ps();

  for (i = 0; i <= total_elements - 16; i += 16) {
    __m512 src_vec = _mm512_loadu_ps(&src_tensor->data[i]);
    __m512 result_vec = _mm512_max_ps(src_vec, zero_vec);
    _mm512_storeu_ps(&dst_tensor->data[i], result_vec);
  }

  for (; i < total_elements; ++i) {
    dst_tensor->data[i] = src_tensor->data[i] > 0.0 ? src_tensor->data[i] : 0.0;
  }

  int curr_id = tensor_float_id;
  tensor_float_list[curr_id] = dst_tensor;
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Softmax(int src_id) {
  if (tensor_float_list[tensor_float_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[tensor_float_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorFloat* dst_tensor = CreateTensorFloat(B, M, N);

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      float max_val = src_tensor->data[i * M * N + j * N];
      for (int k = 1; k < N; ++k) {
        if (src_tensor->data[i * M * N + j * N + k] > max_val) {
          max_val = src_tensor->data[i * M * N + j * N + k];
        }
      }

      float sum = 0.0;
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            expf(src_tensor->data[i * M * N + j * N + k] - max_val);
        sum += dst_tensor->data[i * M * N + j * N + k];
      }

      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] /= sum;
      }
    }
  }

  int curr_id = tensor_float_id;
  tensor_float_list[curr_id] = dst_tensor;
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Quantize_Post_Softmax(int src_id) {
  if (tensor_int8_list[tensor_int8_id] != NULL) {
    DeleteTensorInt8(tensor_int8_list[tensor_int8_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorInt8* dst_tensor = CreateTensorInt8(B, M, N);

  __m512 scale_vec = _mm512_set1_ps(127.0f);

  int total_elements = B * M * N;
  int i;

  for (i = 0; i <= total_elements - 16; i += 16) {
    __m512 src_vec = _mm512_loadu_ps(&src_tensor->data[i]);
    __m512 scaled_vec = _mm512_mul_ps(src_vec, scale_vec);
    __m512i int_vec = _mm512_cvtps_epi32(scaled_vec);
    __m512i clamped_vec =
        _mm512_min_epi32(_mm512_max_epi32(int_vec, _mm512_set1_epi32(-128)),
                         _mm512_set1_epi32(127));
    __m128i result_vec = _mm512_cvtsepi32_epi8(clamped_vec);
    _mm_storeu_si128((__m128i*)&dst_tensor->data[i], result_vec);
  }

  for (; i < total_elements; ++i) {
    dst_tensor->data[i] = (char)roundf(src_tensor->data[i] * 127.0f);
  }

  int curr_id = tensor_int8_id;
  tensor_int8_list[curr_id] = dst_tensor;
  tensor_int8_id = (tensor_int8_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Cast_From_Float_To_Int8(int src_id) {
  if (tensor_int8_list[tensor_int8_id] != NULL) {
    DeleteTensorInt8(tensor_int8_list[tensor_int8_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorInt8* dst_tensor = CreateTensorInt8(B, M, N);

  int total_elements = B * M * N;
  int i;

  for (i = 0; i <= total_elements - 16; i += 16) {
    __m512 src_vec = _mm512_loadu_ps(&src_tensor->data[i]);
    __m512i int_vec = _mm512_cvtps_epi32(src_vec);
    __m512i clamped_vec =
        _mm512_max_epi32(_mm512_set1_epi32(-128),
                         _mm512_min_epi32(int_vec, _mm512_set1_epi32(127)));
    __m128i result_vec = _mm512_cvtsepi32_epi8(clamped_vec);
    _mm_storeu_si128((__m128i*)&dst_tensor->data[i], result_vec);
  }

  for (; i < total_elements; ++i) {
    dst_tensor->data[i] = (char)src_tensor->data[i];
  }

  int curr_id = tensor_int8_id;
  tensor_int8_list[curr_id] = dst_tensor;
  tensor_int8_id = (tensor_int8_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Cast_From_Float_To_Int32(int src_id) {
  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorInt32* dst_tensor = CreateTensorInt32(B, M, N);

  int total_elements = B * M * N;
  int i;

  for (i = 0; i <= total_elements - 16; i += 16) {
    __m512 src_vec = _mm512_loadu_ps(&src_tensor->data[i]);
    __m512i int_vec = _mm512_cvtps_epi32(src_vec);
    _mm512_storeu_si512((__m512i*)&dst_tensor->data[i], int_vec);
  }

  for (; i < total_elements; ++i) {
    dst_tensor->data[i] = (char)src_tensor->data[i];
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = dst_tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Cast_From_Int8_To_Int32(int src_id) {
  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }

  struct TensorInt8* src_tensor = tensor_int8_list[src_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  struct TensorInt32* dst_tensor = CreateTensorInt32(B, M, N);

  int total_elements = B * M * N;
  int i;

  for (i = 0; i <= total_elements - 16; i += 16) {
    __m128i src_vec = _mm_loadu_si128((__m128i*)&src_tensor->data[i]);
    __m512i int_vec = _mm512_cvtepi8_epi32(src_vec);
    _mm512_storeu_si512((__m512i*)&dst_tensor->data[i], int_vec);
  }

  for (; i < total_elements; ++i) {
    dst_tensor->data[i] = (char)src_tensor->data[i];
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = dst_tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Get_Tensor_Dim_Int8(int src_id, int* dim) {
  struct TensorInt8* src_tensor = tensor_int8_list[src_id];
  dim[0] = src_tensor->B;
  dim[1] = src_tensor->M;
  dim[2] = src_tensor->N;
}

void Ex_Get_Tensor_Int8(int src_id, char* out) {
  struct TensorInt8* src_tensor = tensor_int8_list[src_id];
  int total_elements = src_tensor->B * src_tensor->M * src_tensor->N;

  int i;
  for (i = 0; i <= total_elements - 64; i += 64) {
    __m512i data_vec = _mm512_loadu_si512((__m512i*)&src_tensor->data[i]);
    _mm512_storeu_si512((__m512i*)&out[i], data_vec);
  }

  for (; i < total_elements; ++i) {
    out[i] = src_tensor->data[i];
  }
}

int Ex_Set_Tensor_Int8(char* data, int B, int M, int N) {
  int curr_id = tensor_int8_id;
  if (tensor_int8_list[curr_id] != NULL) {
    DeleteTensorInt8(tensor_int8_list[curr_id]);
  }

  tensor_int8_list[curr_id] = CreateTensorInt8FromData(data, B, M, N);
  tensor_int8_id = (tensor_int8_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Get_Tensor_Dim_Float(int src_id, int* dim) {
  struct TensorFloat* src_tensor = tensor_float_list[src_id];
  dim[0] = src_tensor->B;
  dim[1] = src_tensor->M;
  dim[2] = src_tensor->N;
}

void Ex_Get_Tensor_Float(int src_id, float* out) {
  struct TensorFloat* src_tensor = tensor_float_list[src_id];
  int total_elements = src_tensor->B * src_tensor->M * src_tensor->N;

  int i;
  for (i = 0; i <= total_elements - 16; i += 16) {
    __m512 data_vec = _mm512_loadu_ps(&src_tensor->data[i]);
    _mm512_storeu_ps(&out[i], data_vec);
  }

  for (; i < total_elements; ++i) {
    out[i] = src_tensor->data[i];
  }
}

int Ex_Set_Tensor_Float(float* data, int B, int M, int N) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  tensor_float_list[curr_id] = CreateTensorFloatFromData(data, B, M, N);
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Bmm_Param(float alpha) {
  int curr_id = bmm_param_id;
  bmm_param_list[curr_id] = alpha;
  bmm_param_id = (bmm_param_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Residual_Add(int residual, int hidden_states) {
  if (tensor_float_list[tensor_float_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[tensor_float_id]);
  }

  struct TensorFloat* residual_tensor = tensor_float_list[residual];
  struct TensorFloat* hidden_states_tensor = tensor_float_list[hidden_states];

  int B = residual_tensor->B;
  int M = residual_tensor->M;
  int N = residual_tensor->N;

  struct TensorFloat* dst_tensor = CreateTensorFloat(B, M, N);

  int total_elements = B * M * N;
  int i;

  for (i = 0; i <= total_elements - 16; i += 16) {
    __m512 residual_vec = _mm512_loadu_ps(&residual_tensor->data[i]);
    __m512 hidden_vec = _mm512_loadu_ps(&hidden_states_tensor->data[i]);
    __m512 result_vec = _mm512_add_ps(residual_vec, hidden_vec);
    _mm512_storeu_ps(&dst_tensor->data[i], result_vec);
  }

  for (; i < total_elements; ++i) {
    dst_tensor->data[i] =
        residual_tensor->data[i] + hidden_states_tensor->data[i];
  }

  int curr_id = tensor_float_id;
  tensor_float_list[curr_id] = dst_tensor;
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_CPU_Bmm(int src_id1, int src_id2) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  struct TensorInt32* Z = MatmulS32S32S32(X, Y);  // B x M x N

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = Z;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}