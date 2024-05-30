
#include "sgx_layer_struct.h"
#include <immintrin.h>
#include "Enclave.h"
// #include <omp.h>

extern "C" {

int Sgx_Set_Hidden_States(float* hidden_states, int B, int M, int N) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  tensor_float_list[curr_id] = CreateTensorFloat(B, M, N);
  struct TensorFloat* tensor = tensor_float_list[curr_id];
  int total_elements = B * M * N;

  int i;
  for (i = 0; i <= total_elements - 8; i += 8) {
    __m256 vec = _mm256_loadu_ps(&hidden_states[i]);
    _mm256_storeu_ps(&tensor->data[i], vec);
  }

  // Handle remaining elements
  for (; i < total_elements; i++) {
    tensor->data[i] = hidden_states[i];
  }

  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Sgx_Copy_Hidden_States(int src_id) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];
  tensor_float_list[curr_id] = CreateTensorFloat(src_tensor->B, src_tensor->M, src_tensor->N);

  struct TensorFloat* dst_tensor = tensor_float_list[curr_id];
  int total_elements = src_tensor->B * src_tensor->M * src_tensor->N;

  int i;
  for (i = 0; i <= total_elements - 8; i += 8) {
    __m256 vec = _mm256_loadu_ps(&src_tensor->data[i]);
    _mm256_storeu_ps(&dst_tensor->data[i], vec);
  }

  // Handle remaining elements
  for (; i < total_elements; i++) {
    dst_tensor->data[i] = src_tensor->data[i];
  }

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

// Helper function to reduce sum of __m256
inline float _mm256_reduce_add_ps(__m256 x) {
  __m128 hi = _mm256_extractf128_ps(x, 1);
  __m128 lo = _mm256_castps256_ps128(x);
  __m128 sum = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum);
  __m128 sums = _mm_add_ps(sum, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
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
      __m256 sum_vec = _mm256_setzero_ps();
      __m256 sum_sqr_vec = _mm256_setzero_ps();

      // Calculate sum and sum of squares using vectorization
      int k;
      for (k = 0; k <= N - 8; k += 8) {
        __m256 tmp_vec = _mm256_loadu_ps(&src_tensor->data[i * M * N + j * N + k]);
        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
        sum_sqr_vec = _mm256_add_ps(sum_sqr_vec, _mm256_mul_ps(tmp_vec, tmp_vec));
      }

      // Reduce sum and sum of squares
      float sum = _mm256_reduce_add_ps(sum_vec);
      float sum_sqr = _mm256_reduce_add_ps(sum_sqr_vec);

      // Handle remaining elements
      for (; k < N; ++k) {
        float tmp = src_tensor->data[i * M * N + j * N + k];
        sum += tmp;
        sum_sqr += tmp * tmp;
      }

      float mean = sum / N;
      float var = sum_sqr / N - mean * mean;
      float inv_stddev = 1.0f / sqrtf(var + layer_norm_param->eps);

      // Normalize and clamp using vectorization
      __m256 mean_vec = _mm256_set1_ps(mean);
      __m256 inv_stddev_vec = _mm256_set1_ps(inv_stddev);

      for (k = 0; k <= N - 8; k += 8) {
        __m256 tmp_vec = _mm256_loadu_ps(&src_tensor->data[i * M * N + j * N + k]);
        __m256 gamma_vec = _mm256_loadu_ps(&layer_norm_param->gamma->data[k]);
        __m256 beta_vec = _mm256_loadu_ps(&layer_norm_param->beta->data[k]);

        __m256 norm_vec = _mm256_mul_ps(_mm256_sub_ps(tmp_vec, mean_vec), inv_stddev_vec);
        __m256 scaled_vec = _mm256_fmadd_ps(norm_vec, gamma_vec, beta_vec);
        __m256 rounded_vec = _mm256_round_ps(scaled_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i int_vec = _mm256_cvtps_epi32(rounded_vec);

        // Clamp the results to fit into int8 range
        __m256i clamped_vec = _mm256_max_epi32(_mm256_min_epi32(int_vec, _mm256_set1_epi32(127)), _mm256_set1_epi32(-128));

        // Convert to int8 and store
        char* dst_ptr = &dst_tensor->data[i * M * N + j * N + k];
        for (int l = 0; l < 8; ++l) {
          dst_ptr[l] = (char)_mm256_extract_epi32(clamped_vec, l);
        }
      }

      // Handle remaining elements
      for (; k < N; ++k) {
        float tmp = src_tensor->data[i * M * N + j * N + k];
        tmp = (tmp - mean) * inv_stddev * layer_norm_param->gamma->data[k] + layer_norm_param->beta->data[k];
        tmp = roundf(tmp);
        if (tmp > 127.0f) {
          tmp = 127.0f;
        } else if (tmp < -128.0f) {
          tmp = -128.0f;
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
  linear_param->bias_int8 = CreateTensorInt8FromData(bias, 1, 1, N);
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
  linear_param->bias_float = CreateTensorFloatFromData(bias, 1, 1, N);
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
  for (i = 0; i <= total_elements - 8; i += 8) {
    __m256i vec = _mm256_loadu_si256((__m256i*)&src_tensor->data[i]);
    _mm256_storeu_si256((__m256i*)&out[i], vec);
  }

  // Handle remaining elements
  for (; i < total_elements; i++) {
    out[i] = src_tensor->data[i];
  }
}

// Need to return blind factor id
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
      int k;
      for (k = 0; k <= N - 8; k += 8) {
        __m256i out_vec = _mm256_loadu_si256((__m256i*)&out[i * M * N + j * N + k]);
        __m256i blind_vec = _mm256_loadu_si256((__m256i*)&blind_factor->data[i * N + k]);
        __m256i result_vec = _mm256_add_epi32(out_vec, blind_vec);
        _mm256_storeu_si256((__m256i*)&out[i * M * N + j * N + k], result_vec);
      }

      // Handle remaining elements
      for (; k < N; ++k) {
        out[i * M * N + j * N + k] += blind_factor->data[i * N + k];
      }
    }
  }

  tensor_int32_list[curr_id] = blind_factor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
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

int Sgx_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N,
                                       int blind_factor_id,
                                       int linear_param_id) {
  // Compute Unblinding factor
  struct TensorInt32* blind_factor = tensor_int32_list[blind_factor_id];
  struct TensorInt8* linear_weight = linear_param_list[linear_param_id]->weight;

  struct TensorInt32* unblind_factor = MatmulS32S8S32(blind_factor, linear_weight);

  // Perform vectorized subtraction using AVX2
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      int k;
      for (k = 0; k <= N - 8; k += 8) {
        __m256i data_vec = _mm256_loadu_si256((__m256i*)&data[i * M * N + j * N + k]);
        __m256i unblind_vec = _mm256_loadu_si256((__m256i*)&unblind_factor->data[i * N + k]);
        __m256i result_vec = _mm256_sub_epi32(data_vec, unblind_vec);
        _mm256_storeu_si256((__m256i*)&data[i * M * N + j * N + k], result_vec);
      }

      // Handle remaining elements
      for (; k < N; ++k) {
        data[i * M * N + j * N + k] -= unblind_factor->data[i * N + k];
      }
    }
  }

  return Sgx_Set_Tensor_Int32(data, B, M, N);
}

int Sgx_Get_Encrypted_Tensor_Opr2_Int32(int src_id1, int src_id2, int* out1,
                                       int* out2) {
    struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
    struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

    int B = X->B;
    int M = X->M;
    int K = X->N;
    int N = Y->M;

    struct TensorInt32* Y_trans = TransposeLastTwoDimsInt32(Y);  // B x K x N

    struct TensorInt32* u = CreateTensorInt32(B, 1, K);  // temporary
    GetCPRNG((unsigned char*)u->data, u->num_bytes);

    struct TensorInt32* v = CreateTensorInt32(B, K, 1);  // temporary
    GetCPRNG((unsigned char*)v->data, v->num_bytes);

    // Encrypt X
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < M; ++j) {
            int k;
            for (k = 0; k <= K - 8; k += 8) {
                __m256i x_vec = _mm256_loadu_si256((__m256i*)&X->data[i * M * K + j * K + k]);
                __m256i u_vec = _mm256_loadu_si256((__m256i*)&u->data[i * K + k]);
                __m256i result_vec = _mm256_add_epi32(x_vec, u_vec);
                _mm256_storeu_si256((__m256i*)&out1[i * M * K + j * K + k], result_vec);
            }

            // Handle remaining elements
            for (; k < K; ++k) {
                out1[i * M * K + j * K + k] = X->data[i * M * K + j * K + k] + u->data[i * K + k];
            }
        }
    }

    // Encrypt Y^T
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < K; ++j) {
            int k;
            for (k = 0; k <= N - 8; k += 8) {
                __m256i y_vec = _mm256_loadu_si256((__m256i*)&Y_trans->data[i * K * N + j * N + k]);
                __m256i v_vec = _mm256_set1_epi32(v->data[i * K + j]);
                __m256i result_vec = _mm256_add_epi32(y_vec, v_vec);
                _mm256_storeu_si256((__m256i*)&out2[i * K * N + j * N + k], result_vec);
            }

            // Handle remaining elements
            for (; k < N; ++k) {
                out2[i * K * N + j * N + k] = Y_trans->data[i * K * N + j * N + k] + v->data[i * K + j];
            }
        }
    }

    struct TensorInt32* xv = MatmulS32S32S32(X, v);  // B x M x 1
    struct TensorInt32* uy = MatmulS32S32S32(u, Y_trans);
    struct TensorInt32* uv = MatmulS32S32S32(u, v);

    struct TensorInt32* unblind_factor = CreateTensorInt32(B, M, N);
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < M; ++j) {
            int k;
            for (k = 0; k <= N - 8; k += 8) {
                __m256i xv_vec = _mm256_set1_epi32(xv->data[i * M + j]);
                __m256i uy_vec = _mm256_loadu_si256((__m256i*)&uy->data[i * N + k]);
                __m256i uv_vec = _mm256_set1_epi32(uv->data[i]);
                __m256i result_vec = _mm256_add_epi32(xv_vec, _mm256_add_epi32(uy_vec, uv_vec));
                _mm256_storeu_si256((__m256i*)&unblind_factor->data[i * M * N + j * N + k], result_vec);
            }

            // Handle remaining elements
            for (; k < N; ++k) {
                unblind_factor->data[i * M * N + j * N + k] = xv->data[i * M + j] + uy->data[i * N + k] + uv->data[i];
            }
        }
    }

    // Cleanup
    DeleteTensorInt32(u);
    DeleteTensorInt32(v);
    DeleteTensorInt32(xv);
    DeleteTensorInt32(uy);
    DeleteTensorInt32(uv);
    DeleteTensorInt32(Y_trans);

    int curr_id = tensor_int32_id;
    tensor_int32_list[curr_id] = unblind_factor;
    tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
    return curr_id;
}

int Sgx_Set_Decrypted_Tensor_Opr2_Int32(int* data, int B, int M, int N,
                                       int unblind_factor_id) {
  struct TensorInt32* unblind_factor = tensor_int32_list[unblind_factor_id];

  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);
  int total_elements = B * M * N;

  // Perform vectorized subtraction using AVX2
  int i;
  for (i = 0; i <= total_elements - 8; i += 8) {
    __m256i data_vec = _mm256_loadu_si256((__m256i*)&data[i]);
    __m256i unblind_vec = _mm256_loadu_si256((__m256i*)&unblind_factor->data[i]);
    __m256i result_vec = _mm256_sub_epi32(data_vec, unblind_vec);
    _mm256_storeu_si256((__m256i*)&tensor->data[i], result_vec);
  }

  // Handle remaining elements
  for (; i < total_elements; i++) {
    tensor->data[i] = data[i] - unblind_factor->data[i];
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp parallel for simd
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp parallel for simd
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

  // #pragma omp parallel for simd
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp simd
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            src_tensor->data[i * M * N + j * N + k] > 0.0
                ? src_tensor->data[i * M * N + j * N + k]
                : 0.0;
      }
    }
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      float max_val = src_tensor->data[i * M * N + j * N];
      // #pragma omp simd reduction(max:max_val)
      for (int k = 1; k < N; ++k) {
        if (src_tensor->data[i * M * N + j * N + k] > max_val) {
          max_val = src_tensor->data[i * M * N + j * N + k];
        }
      }

      float sum = 0.0;
      // #pragma omp simd reduction(+:sum)
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            expf(src_tensor->data[i * M * N + j * N + k] - max_val);
        sum += dst_tensor->data[i * M * N + j * N + k];
      }

      // #pragma omp simd
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp simd
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            (char)roundf(src_tensor->data[i * M * N + j * N + k] * 127.0);
      }
    }
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp simd
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            (char)src_tensor->data[i * M * N + j * N + k];
      }
    }
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp simd
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            (int)src_tensor->data[i * M * N + j * N + k];
      }
    }
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp simd
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            (int)src_tensor->data[i * M * N + j * N + k];
      }
    }
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

  // #pragma omp parallel for simd
  for (int i = 0; i < src_tensor->B * src_tensor->M * src_tensor->N; i++) {
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

  // #pragma omp parallel for simd
  for (int i = 0; i < src_tensor->B * src_tensor->M * src_tensor->N; i++) {
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

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      // #pragma omp simd
      for (int k = 0; k < N; ++k) {
        dst_tensor->data[i * M * N + j * N + k] =
            residual_tensor->data[i * M * N + j * N + k] +
            hidden_states_tensor->data[i * M * N + j * N + k];
      }
    }
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

void ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int* out, int* ret_id) {
  *ret_id = Sgx_Get_Encrypted_Tensor_Opr1_Int32(src_id,out);
}

void ecall_Sgx_Set_Tensor_Int32(int* data, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Tensor_Int32(data,B,M,N);
}

void ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int blind_factor_id, int linear_param_id, int* ret_id) {
  *ret_id = Sgx_Set_Decrypted_Tensor_Opr1_Int32(data,B,M,N,blind_factor_id,linear_param_id);
}

void ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32(int src_id1, int src_id2, int* out1, int* out2, int* ret_id) {
  *ret_id = Sgx_Get_Encrypted_Tensor_Opr2_Int32(src_id1,src_id2,out1,out2);
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