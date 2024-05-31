
#include "sgx_layer_struct.h"

#include "Enclave.h"
#include <immintrin.h>

extern "C" {

int Sgx_Set_Hidden_States(float* hidden_states, int B, int M, int N) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  tensor_float_list[curr_id] =
      CreateTensorFloatFromData(hidden_states, B, M, N);
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Copy_Hidden_States(int src_id) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];

  tensor_float_list[curr_id] = CreateTensorFloatFromData(src_tensor->data, src_tensor->B, src_tensor->M, src_tensor->N);

  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps) {
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

int Sgx_Layer_Norm_Q(int src_id, int layer_norm_param_id) {
    int curr_id = tensor_int8_id;

    struct TensorFloat* src_tensor = tensor_float_list[src_id];
    struct LayerNormParam* layer_norm_param = layer_norm_param_list[layer_norm_param_id];

    struct TensorInt8* dst_tensor = CreateTensorInt8(src_tensor->B, src_tensor->M, src_tensor->N);

    int B = src_tensor->B;
    int M = src_tensor->M;
    int N = src_tensor->N;

    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < M; ++j) {
            __m512 sum_vec = _mm512_setzero_ps();
            __m512 sum_sqr_vec = _mm512_setzero_ps();

            // Calculate sum and sum of squares
            for (int k = 0; k <= N - 16; k += 16) {
                __m512 src_vec = _mm512_loadu_ps(&src_tensor->data[i * M * N + j * N + k]);
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
                __m512 src_vec = _mm512_loadu_ps(&src_tensor->data[i * M * N + j * N + k]);
                __m512 norm_vec = _mm512_sub_ps(src_vec, mean_vec);
                norm_vec = _mm512_mul_ps(norm_vec, inv_std_vec);

                gamma_vec = _mm512_loadu_ps(&layer_norm_param->gamma->data[k]);
                beta_vec = _mm512_loadu_ps(&layer_norm_param->beta->data[k]);

                norm_vec = _mm512_fmadd_ps(norm_vec, gamma_vec, beta_vec);
                norm_vec = _mm512_roundscale_ps(norm_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Clamp between -128 and 127
                __m512i int_vec = _mm512_cvtps_epi32(norm_vec);
                int_vec = _mm512_max_epi32(int_vec, _mm512_set1_epi32(-128));
                int_vec = _mm512_min_epi32(int_vec, _mm512_set1_epi32(127));

                // Store the result as int8
                __m128i int8_vec = _mm512_cvtsepi32_epi8(int_vec);
                _mm_storeu_si128((__m128i*)&dst_tensor->data[i * M * N + j * N + k], int8_vec);
            }

            // Process any remaining elements
            for (int k = (N / 16) * 16; k < N; ++k) {
                float tmp = src_tensor->data[i * M * N + j * N + k];
                tmp = (tmp - mean) * inv_std * layer_norm_param->gamma->data[k] + layer_norm_param->beta->data[k];
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

int Sgx_Set_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N,
                               float alpha, float beta) {
  int curr_id = linear_param_id;

  struct LinearParam* linear_param =
      (struct LinearParam*)malloc(sizeof(struct LinearParam));
  linear_param->weight = CreateTensorInt8FromData(weight, 1, M, N);
  linear_param->bias_int8 = CreateTensorInt8FromData(bias, 1, 1, M);
  linear_param->alpha = alpha;
  linear_param->beta = beta;
  linear_param->is_bias_fp32 = 0;

  linear_param_list[curr_id] = linear_param;
  linear_param_id = (linear_param_id + 1) % STATIC_LIST_LEN;

  return curr_id;
}

int Sgx_Set_Linear_Param_WS8BFP32(char* weight, float* bias, int M, int N,
                                 float alpha) {
  int curr_id = linear_param_id;

  struct LinearParam* linear_param =
      (struct LinearParam*)malloc(sizeof(struct LinearParam));
  linear_param->weight = CreateTensorInt8FromData(weight, 1, M, N);
  linear_param->bias_float = CreateTensorFloatFromData(bias, 1, 1, M);
  linear_param->alpha = alpha;
  linear_param->beta = 1.0;
  linear_param->is_bias_fp32 = 1;

  linear_param_list[curr_id] = linear_param;
  linear_param_id = (linear_param_id + 1) % STATIC_LIST_LEN;

  return curr_id;
}

void Sgx_Get_Tensor_Dim_Int32(int src_id, int* dim) {
  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  dim[0] = src_tensor->B;
  dim[1] = src_tensor->M;
  dim[2] = src_tensor->N;
}

void Sgx_Get_Tensor_Int32(int src_id, int* out) {
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

int Sgx_Set_Tensor_Int32(int* data, int B, int M, int N) {
  int curr_id = tensor_int32_id;
  if (tensor_int32_list[curr_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[curr_id]);
  }

  tensor_int32_list[curr_id] = CreateTensorInt32FromData(data, B, M, N);
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int* out) {
    Sgx_Get_Tensor_Int32(src_id, out);

    struct TensorInt32* src_tensor = tensor_int32_list[src_id];

    int B = src_tensor->B;
    int M = src_tensor->M;
    int N = src_tensor->N;

    int curr_id = tensor_int32_id;
    struct TensorInt32* blind_factor = CreateTensorInt32(B, 1, N);

    GetCPRNG((unsigned char*)blind_factor->data, blind_factor->num_bytes);

    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int k = 0; k <= N - 16; k += 16) {
                __m512i out_vec = _mm512_loadu_si512((__m512i*)&out[i * M * N + j * N + k]);
                __m512i blind_vec = _mm512_loadu_si512((__m512i*)&blind_factor->data[i * N + k]);
                out_vec = _mm512_add_epi32(out_vec, blind_vec);
                _mm512_storeu_si512((__m512i*)&out[i * M * N + j * N + k], out_vec);
            }
            for (int k = (N / 16) * 16; k < N; ++k) {
                out[i * M * N + j * N + k] += blind_factor->data[i * N + k];
            }
        }
    }

    tensor_int32_list[curr_id] = blind_factor;
    tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
    return curr_id;
}


int Sgx_Generate_Decryption_Key_Opr1_Int32(int blind_factor_id,
                                          int linear_param_id) {
  struct TensorInt32* blind_factor = tensor_int32_list[blind_factor_id];
  struct TensorInt8* linear_weight = linear_param_list[linear_param_id]->weight;

  struct TensorInt32* decryption_key =
      MatmulS32S8S32(blind_factor, linear_weight);

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = decryption_key;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N,
                                       int decryption_key_id) {
  struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  for (int i = 0; i < B; ++i) {
      for (int j = 0; j < M; ++j) {
          for (int k = 0; k <= N - 16; k += 16) {
              __m512i data_vec = _mm512_loadu_si512((__m512i*)&data[i * M * N + j * N + k]);
              __m512i key_vec = _mm512_loadu_si512((__m512i*)&decryption_key->data[i * N + k]);
              data_vec = _mm512_sub_epi32(data_vec, key_vec);
              _mm512_storeu_si512((__m512i*)&data[i * M * N + j * N + k], data_vec);
          }
          for (int k = (N / 16) * 16; k < N; ++k) {
              data[i * M * N + j * N + k] -= decryption_key->data[i * N + k];
          }
      }
  }

  return Sgx_Set_Tensor_Int32(data, B, M, N);
}

void Sgx_Get_Encrypted_Tensor_Opr2_Int32(int src_id1, int src_id2, int* out1,
                                        int* out2, int* blind_factor_ids) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->M;

  struct TensorInt32* u = CreateTensorInt32(B, 1, K);
  GetCPRNG((unsigned char*)u->data, u->num_bytes);

  struct TensorInt32* v = CreateTensorInt32(B, 1, K);
  GetCPRNG((unsigned char*)v->data, v->num_bytes);

  // Encrypt X
  for (int i = 0; i < B; ++i) {
      for (int j = 0; j < M; ++j) {
          for (int k = 0; k <= K - 16; k += 16) {
              __m512i x_vec = _mm512_loadu_si512((__m512i*)&X->data[i * M * K + j * K + k]);
              __m512i u_vec = _mm512_loadu_si512((__m512i*)&u->data[i * K + k]);
              __m512i out_vec = _mm512_add_epi32(x_vec, u_vec);
              _mm512_storeu_si512((__m512i*)&out1[i * M * K + j * K + k], out_vec);
          }
          for (int k = (K / 16) * 16; k < K; ++k) {
              out1[i * M * K + j * K + k] = X->data[i * M * K + j * K + k] + u->data[i * K + k];
          }
      }
  }

  // Encrypt Y
  for (int i = 0; i < B; ++i) {
      for (int j = 0; j < N; ++j) {
          for (int k = 0; k <= K - 16; k += 16) {
              __m512i y_vec = _mm512_loadu_si512((__m512i*)&Y->data[i * N * K + j * K + k]);
              __m512i v_vec = _mm512_loadu_si512((__m512i*)&v->data[i * K + k]);
              __m512i out_vec = _mm512_add_epi32(y_vec, v_vec);
              _mm512_storeu_si512((__m512i*)&out2[i * N * K + j * K + k], out_vec);
          }
          for (int k = (K / 16) * 16; k < K; ++k) {
              out2[i * N * K + j * K + k] = Y->data[i * N * K + j * K + k] + v->data[i * K + k];
          }
      }
  }

  blind_factor_ids[0] = tensor_int32_id;
  tensor_int32_list[blind_factor_ids[0]] = u;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;

  blind_factor_ids[1] = tensor_int32_id;
  tensor_int32_list[blind_factor_ids[1]] = v;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
}

int Sgx_Generate_Decryption_Key_Opr2_Int32(int src_id1, int src_id2,
                                          int blind_factor_u_id,
                                          int blind_factor_v_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  struct TensorInt32* u = tensor_int32_list[blind_factor_u_id];
  struct TensorInt32* v = tensor_int32_list[blind_factor_v_id];

  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->M;

  struct TensorInt32* xv = MatmulS32S32S32(X, v);  // B x M x 1
  struct TensorInt32* uy = MatmulS32S32S32(u, Y);
  struct TensorInt32* uv = MatmulS32S32S32(u, v);

  struct TensorInt32* decryption_key = CreateTensorInt32(B, M, N);

  for (int i = 0; i < B; ++i) {
      for (int j = 0; j < M; ++j) {
          for (int k = 0; k <= N - 16; k += 16) {
              __m512i uv_vec = _mm512_set1_epi32(uv->data[i]);
              __m512i xv_vec = _mm512_set1_epi32(xv->data[i * M + j]);
              __m512i uy_vec = _mm512_loadu_si512((__m512i*)&uy->data[i * N + k]);
              __m512i sum_vec = _mm512_add_epi32(uv_vec, xv_vec);
              sum_vec = _mm512_add_epi32(sum_vec, uy_vec);
              _mm512_storeu_si512((__m512i*)&decryption_key->data[i * M * N + j * N + k], sum_vec);
          }
          for (int k = (N / 16) * 16; k < N; ++k) {
              decryption_key->data[i * M * N + j * N + k] =
                  uv->data[i] + xv->data[i * M + j] + uy->data[i * N + k];
          }
      }
  }

  DeleteTensorInt32(xv);
  DeleteTensorInt32(uy);
  DeleteTensorInt32(uv);

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = decryption_key;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Set_Decrypted_Tensor_Opr2_Int32(int* data, int B, int M, int N,
                                       int decryption_key_id) {
  struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);

  int total_elements = B * M * N;

  int i;
  for (i = 0; i <= total_elements - 16; i += 16) {
      __m512i data_vec = _mm512_loadu_si512((__m512i*)&data[i]);
      __m512i key_vec = _mm512_loadu_si512((__m512i*)&decryption_key->data[i]);
      __m512i result_vec = _mm512_sub_epi32(data_vec, key_vec);
      _mm512_storeu_si512((__m512i*)&tensor->data[i], result_vec);
  }

  for (; i < total_elements; ++i) {
      tensor->data[i] = data[i] - decryption_key->data[i];
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id) {
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

int Sgx_Compute_Epilogue_WS8BFP32(int src_id, int linear_param_id) {
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

int Sgx_Compute_Epilogue_BMM(int src_id, int bmm_param_id) {
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

int Sgx_ReLU(int src_id) {
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


int Sgx_Softmax(int src_id) {
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

int Sgx_Quantize_Post_Softmax(int src_id) {
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
        __m512i clamped_vec = _mm512_min_epi32(_mm512_max_epi32(int_vec, _mm512_set1_epi32(-128)), _mm512_set1_epi32(127));
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

int Sgx_Cast_From_Float_To_Int8(int src_id) {
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
        __m512i clamped_vec = _mm512_max_epi32(_mm512_set1_epi32(-128), _mm512_min_epi32(int_vec, _mm512_set1_epi32(127)));
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

int Sgx_Cast_From_Float_To_Int32(int src_id) {
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

int Sgx_Cast_From_Int8_To_Int32(int src_id) {
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

void Sgx_Get_Tensor_Dim_Int8(int src_id, int* dim) {
  struct TensorInt8* src_tensor = tensor_int8_list[src_id];
  dim[0] = src_tensor->B;
  dim[1] = src_tensor->M;
  dim[2] = src_tensor->N;
}

void Sgx_Get_Tensor_Int8(int src_id, char* out) {
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

int Sgx_Set_Tensor_Int8(char* data, int B, int M, int N) {
  int curr_id = tensor_int8_id;
  if (tensor_int8_list[curr_id] != NULL) {
    DeleteTensorInt8(tensor_int8_list[curr_id]);
  }

  tensor_int8_list[curr_id] = CreateTensorInt8FromData(data, B, M, N);
  tensor_int8_id = (tensor_int8_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Sgx_Get_Tensor_Dim_Float(int src_id, int* dim) {
  struct TensorFloat* src_tensor = tensor_float_list[src_id];
  dim[0] = src_tensor->B;
  dim[1] = src_tensor->M;
  dim[2] = src_tensor->N;
}


void Sgx_Get_Tensor_Float(int src_id, float* out) {
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

int Sgx_Set_Tensor_Float(float* data, int B, int M, int N) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  tensor_float_list[curr_id] = CreateTensorFloatFromData(data, B, M, N);
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Set_Bmm_Param(float alpha) {
  int curr_id = bmm_param_id;
  bmm_param_list[curr_id] = alpha;
  bmm_param_id = (bmm_param_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Residual_Add(int residual, int hidden_states) {
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
        dst_tensor->data[i] = residual_tensor->data[i] + hidden_states_tensor->data[i];
    }

    int curr_id = tensor_float_id;
    tensor_float_list[curr_id] = dst_tensor;
    tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
    return curr_id;
}

void ecall_Sgx_Set_Hidden_States(float* hidden_states, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Hidden_States(hidden_states,B,M,N);
}

void ecall_Sgx_Copy_Hidden_States(int src_id, int* ret_id) {
  *ret_id = Sgx_Copy_Hidden_States(src_id);
}

void ecall_Sgx_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps, int* ret_id) {
  *ret_id = Sgx_Set_Layer_Norm_Param(gamma,beta,N,eps);
}

void ecall_Sgx_Layer_Norm_Q(int src_id, int layer_norm_param_id, int* ret_id) {
  *ret_id = Sgx_Layer_Norm_Q(src_id,layer_norm_param_id);
}

void ecall_Sgx_Set_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N, float alpha, float beta, int* ret_id) {
  *ret_id = Sgx_Set_Linear_Param_WS8BS8(weight,bias,M,N,alpha,beta);
}

void ecall_Sgx_Set_Linear_Param_WS8BFP32(char* weight, float* bias, int M, int N, float alpha, int* ret_id) {
  *ret_id = Sgx_Set_Linear_Param_WS8BFP32(weight,bias,M,N,alpha);
}

void ecall_Sgx_Get_Tensor_Dim_Int32(int src_id, int* dim) {
  Sgx_Get_Tensor_Dim_Int32(src_id,dim);
}

void ecall_Sgx_Get_Tensor_Int32(int src_id, int* out) {
  Sgx_Get_Tensor_Int32(src_id,out);
}

void ecall_Sgx_Set_Tensor_Int32(int* data, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Tensor_Int32(data,B,M,N);
}

void ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int* out, int* ret_id) {
  *ret_id = Sgx_Get_Encrypted_Tensor_Opr1_Int32(src_id,out);
}

void ecall_Sgx_Generate_Decryption_Key_Opr1_Int32(int blind_factor_id, int linear_param_id, int* ret_id) {
  *ret_id = Sgx_Generate_Decryption_Key_Opr1_Int32(blind_factor_id,linear_param_id);
}

void ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int decryption_key_id, int* ret_id) {
  *ret_id = Sgx_Set_Decrypted_Tensor_Opr1_Int32(data,B,M,N,decryption_key_id);
}

void ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32(int src_id1, int src_id2, int* out1, int* out2, int* blind_factor_ids) {
  Sgx_Get_Encrypted_Tensor_Opr2_Int32(src_id1,src_id2,out1,out2, blind_factor_ids);
}

void ecall_Sgx_Generate_Decryption_Key_Opr2_Int32(int src_id1, int src_id2, int blind_factor_u_id, int blind_factor_v_id, int* ret_id) {
  *ret_id = Sgx_Generate_Decryption_Key_Opr2_Int32(src_id1,src_id2,blind_factor_u_id,blind_factor_v_id);
}

void ecall_Sgx_Set_Decrypted_Tensor_Opr2_Int32(int* data, int B, int M, int N, int unblind_factor_id, int* ret_id) {
  *ret_id = Sgx_Set_Decrypted_Tensor_Opr2_Int32(data,B,M,N,unblind_factor_id);
}

void ecall_Sgx_Get_Tensor_Dim_Int8(int src_id, int* dim) {
  Sgx_Get_Tensor_Dim_Int8(src_id,dim);
}

void ecall_Sgx_Get_Tensor_Int8(int src_id, char* out) {
  Sgx_Get_Tensor_Int8(src_id,out);
}

void ecall_Sgx_Set_Tensor_Int8(char* data, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Tensor_Int8(data,B,M,N);
}

void ecall_Sgx_Get_Tensor_Dim_Float(int src_id, int* dim) {
  Sgx_Get_Tensor_Dim_Float(src_id,dim);
}

void ecall_Sgx_Get_Tensor_Float(int src_id, float* out) {
  Sgx_Get_Tensor_Float(src_id,out);
}

void ecall_Sgx_Set_Tensor_Float(float* data, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Tensor_Float(data,B,M,N);
}

void ecall_Sgx_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id, int* ret_id) {
  *ret_id = Sgx_Compute_Epilogue_WS8BS8(src_id,linear_param_id);
}

void ecall_Sgx_Compute_Epilogue_WS8BFP32(int src_id, int linear_param_id, int* ret_id) {
  *ret_id = Sgx_Compute_Epilogue_WS8BFP32(src_id,linear_param_id);
}

void ecall_Sgx_Compute_Epilogue_BMM(int src_id, int bmm_param_id, int* ret_id) {
  *ret_id = Sgx_Compute_Epilogue_BMM(src_id,bmm_param_id);
}

void ecall_Sgx_ReLU(int src_id, int* ret_id) {
  *ret_id = Sgx_ReLU(src_id);
}

void ecall_Sgx_Softmax(int src_id, int* ret_id) {
  *ret_id = Sgx_Softmax(src_id);
}

void ecall_Sgx_Quantize_Post_Softmax(int src_id, int* ret_id) {
  *ret_id = Sgx_Quantize_Post_Softmax(src_id);
}

void ecall_Sgx_Cast_From_Float_To_Int8(int src_id, int* ret_id) {
  *ret_id = Sgx_Cast_From_Float_To_Int8(src_id);
}

void ecall_Sgx_Cast_From_Float_To_Int32(int src_id, int* ret_id) {
  *ret_id = Sgx_Cast_From_Float_To_Int32(src_id);
}

void ecall_Sgx_Cast_From_Int8_To_Int32(int src_id, int* ret_id) {
  *ret_id = Sgx_Cast_From_Int8_To_Int32(src_id);
}

void ecall_Sgx_Set_Bmm_Param(float alpha, int* ret_id) {
  *ret_id = Sgx_Set_Bmm_Param(alpha);
}

void ecall_Sgx_Residual_Add(int residual, int hidden_states, int* ret_id) {
  *ret_id = Sgx_Residual_Add(residual,hidden_states);
}

}