
#include "secure_llm.h"

#include <chrono>
#include <numeric>

#include "common/aes_stream.h"
#include "common/dynamic_glob_data.h"
#include "common/static_glob_data.h"
#include "common/tensor.h"
#include <cmath>
#include <cassert>

#if SGX_ENABLE == 1
#include "Enclave/Enclave.h"
#include <sgx_trts.h>
#include "tools_sgx.h"
#else
#include "tools.h"
#include <iostream>
#endif

using namespace std;


#define MAX_NUM_HEAD 40
#define MAX_BATCH_SIZE 32
#define MAX_SEQ_LEN 2048
#define MAX_EMBED_DIM (5120 * 4)

#define SHIFT_AMT 128

#define SECRET_KEY_POOL_SIZE 1024

#define MODULO (1LL << 32)

extern "C" {

uint64_t RepeatedSqr(uint64_t base, uint64_t exp, uint64_t mod) {
  uint64_t result = 1;
  base = base % mod;
  while (exp > 0) {
    if (exp % 2 == 1) {
      result = (result * base) % mod;
    }
    exp = exp >> 1;
    base = (base * base) % mod;
  }
  return result;
}

void Ex_Reset() {
  z_row_factor = std::vector<std::vector<uint32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
  z_col_factor = std::vector<std::vector<uint32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
  z_dot_product_factor = std::vector<uint32_t>(MAX_BATCH_SIZE * MAX_NUM_HEAD);

  key_a = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
  key_b = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
  key_c = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
  key_d = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);

  x_row_sum_buffer = std::vector<std::vector<int32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
  y_col_sum_buffer = std::vector<std::vector<int32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);

  for (int i = 0; i < MAX_NUM_LAYERS; ++i) {
    x_row_sum_buffer_opt_list[i] = std::vector<std::vector<int32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    y_col_sum_buffer_opt_list[i] = std::vector<std::vector<int32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);

    z_row_factor_opt_list[i] = std::vector<std::vector<uint32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    z_col_factor_opt_list[i] = std::vector<std::vector<uint32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    z_dot_product_factor_opt_list[i] = std::vector<uint32_t>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    z_dot_product_factor_opt_list_done[i] = false;

    key_a_opt_list[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    key_b_opt_list[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    key_c_opt_list[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    key_d_opt_list[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);

    x_row_sum_buffer_opt_list2[i] = std::vector<std::vector<int32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    y_col_sum_buffer_opt_list2[i] = std::vector<std::vector<int32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);

    z_row_factor_opt_list2[i] = std::vector<std::vector<uint32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    z_col_factor_opt_list2[i] = std::vector<std::vector<uint32_t>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    z_dot_product_factor_opt_list2[i] = std::vector<uint32_t>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    z_dot_product_factor_opt_list_done2[i] = false;

    key_a_opt_list2[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    key_b_opt_list2[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    key_c_opt_list2[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
    key_d_opt_list2[i] = std::vector<std::vector<std::pair<uint32_t, int32_t>>>(MAX_BATCH_SIZE * MAX_NUM_HEAD);
  }
}

void Ex_Pre_Init() {
  if (mult_key_pool.empty()) {
    mult_key_pool = std::vector<uint32_t>(SECRET_KEY_POOL_SIZE); // 1D array
    add_key_pool = std::vector<uint32_t>(SECRET_KEY_POOL_SIZE); // 1D array

    for (int i = 0; i < SECRET_KEY_POOL_SIZE; ++i) {
      uint32_t prng;
      while (true) {
        GetCPRNG((unsigned char*)&prng, sizeof(uint32_t));
        if (std::gcd(prng, MODULO) == 1) {
          break;
        }
      } 
      mult_key_pool.at(i) = prng;
      assert (std::gcd(mult_key_pool.at(i), MODULO) == 1);
    }
    GetCPRNG((unsigned char*)add_key_pool.data(), SECRET_KEY_POOL_SIZE * sizeof(uint32_t));

    // Precompute key inv, 2D array
    mult_key_inv_precompute = std::vector<std::vector<uint32_t>>(SECRET_KEY_POOL_SIZE, std::vector<uint32_t>(SECRET_KEY_POOL_SIZE));
    for (int i = 0; i < SECRET_KEY_POOL_SIZE; ++i) {
      for (int j = 0; j < SECRET_KEY_POOL_SIZE; ++j) {
        uint64_t ab = (uint64_t)mult_key_pool.at(i) * (uint64_t)mult_key_pool.at(j);
        mult_key_inv_precompute.at(i).at(j) = (uint32_t)RepeatedSqr(ab, (1LL << 31) - 1, MODULO);
        assert(((uint64_t)mult_key_inv_precompute[i][j] * ab % MODULO) == 1);
      }
    }
  }

  Ex_Reset();

  if (decryption_key_buffer == NULL) {
    // Not used now
    // decryption_key_buffer = CreateTensorInt32(MAX_BATCH_SIZE, MAX_SEQ_LEN, MAX_EMBED_DIM);
  }
}

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
          float sum = 0.0, sum_sqr = 0.0;

          // Calculate sum and sum of squares
          for (int k = 0; k < N; ++k) {
              float src_val = src_tensor->data[i * M * N + j * N + k];
              sum += src_val;
              sum_sqr += src_val * src_val;
          }

          float mean = sum / N;
          float var = sum_sqr / N - mean * mean;
          float inv_std = 1.0f / sqrtf(var + layer_norm_param->eps);  // Use sqrtf directly

          // Normalize and clamp
          for (int k = 0; k < N; ++k) {
              float src_val = src_tensor->data[i * M * N + j * N + k];
              float norm_val = (src_val - mean) * inv_std;
              norm_val = norm_val * layer_norm_param->gamma->data[k] + layer_norm_param->beta->data[k];
              norm_val = roundf(norm_val);

              // Clamp between -128 and 127
              if (norm_val > 127.0f) {
                  norm_val = 127.0f;
              } else if (norm_val < -128.0f) {
                  norm_val = -128.0f;
              }

              dst_tensor->data[i * M * N + j * N + k] = (char)norm_val;
          }
      }
  }

  tensor_int8_list[curr_id] = dst_tensor;
  tensor_int8_id = (tensor_int8_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

// Need precomputation
int Ex_Set_Linear_Param_WS8BS8(int8_t* weight, int8_t* bias, int M, int N,
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
    GetCPRNG((unsigned char*)&linear_param->blind_factors_set->data[i * N],
             N * sizeof(int));
  }
  linear_param->precomputed_unblind_factors =
      MatmulS32S8S32_Naive(linear_param->blind_factors_set, linear_param->weight);
  // dimension should be (1, obfuscation_ratio, M)

  linear_param_list[curr_id] = linear_param;
  linear_param_id = (linear_param_id + 1) % STATIC_LIST_LEN;

  return curr_id;
}

int Ex_Set_Linear_Param_WS8BFP32(int8_t* weight, float* bias, int M, int N,
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
    GetCPRNG((unsigned char*)&linear_param->blind_factors_set->data[i * N],
             N * sizeof(int));
  }
  linear_param->precomputed_unblind_factors =
      MatmulS32S8S32_Naive(linear_param->blind_factors_set, linear_param->weight);

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

void Ex_Get_Tensor_Int32(int src_id, int32_t* out) {
  struct TensorInt32* src_tensor = tensor_int32_list[src_id];

  int total_elements = src_tensor->B * src_tensor->M * src_tensor->N; // be careful of overflow
  
  for (int i = 0; i < total_elements; ++i) {
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
        out[i * M * N + j * N + k] = out[i * M * N + j * N + k] + blind_factor[k];
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
      MatmulS32S8S32_Naive(blind_factor, linear_weight);

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
        data[i * M * N + j * N + k] = data[i * M * N + j * N + k] -
                      unblind_factor[k];
      }
    }
  }

  return Ex_Set_Tensor_Int32(data, B, M, N);
}
void Ex_Get_Encrypted_Tensor_QK_Int32(int src_id1, int src_id2, unsigned int* out1,
                                      unsigned int* out2) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K
  // auto start = std::chrono::steady_clock::now();
  // Compute X_Row_Sum
  share_dim = X->N; // need it for unshifting

  for (int i = 0; i < X->B; ++i) {
    x_row_sum_buffer.at(i).clear();
    for (int j = 0; j < X->M; ++j) { 
      int sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += X->data[i * X->M * X->N + j * X->N + k];
      }
      x_row_sum_buffer.at(i).push_back(sum);
    }
  }

  // Compute Y_Col_sum
  for (int i = 0; i < Y->B; ++i) {
    y_col_sum_buffer.at(i).clear();
    for (int j = 0; j < Y->M; ++j) {
      int sum = 0;
      for (int k = 0; k < Y->N; ++k) {
        sum += Y->data[i * Y->M * Y->N + j * Y->N + k];
      }
      y_col_sum_buffer.at(i).push_back(sum);
    }
  }


  // Shift X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        X->data[i * X->M * X->N + j * X->N + k] += SHIFT_AMT;
      }
    }
  }

  // Shift Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        Y->data[i * Y->M * Y->N + j * Y->N + k] += SHIFT_AMT;
      }
    }
  }

  // Randomly sample random keys
  for (int i = 0; i < X->B; ++i) {
    key_a.at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_a.at(i).emplace_back(mult_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < Y->B; ++i) {
    key_b.at(i).clear();
    for (int j = 0; j < Y->M; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_b.at(i).emplace_back(mult_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < X->B; ++i) {
    key_c.at(i).clear();
    for (int j = 0; j < X->N; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_c.at(i).emplace_back(add_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < Y->B; ++i) {
    key_d.at(i).clear();
    for (int j = 0; j < Y->N; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_d.at(i).emplace_back(add_key_pool.at(idx), idx);
    }
  }

  // Encrypt X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        // cout << "Before: " << X->data[i * X->M * X->N + j * X->N + k] << endl;
        out1[i * X->M * X->N + j * X->N + k] = (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_a.at(i).at(j).first + key_c.at(i).at(k).first;
        // cout << "After: " << out1[i * X->M * X->N + j * X->N + k] << endl;
        // cout << "Used keys : " << key_a.at(i).at(j).first << " " << key_c.at(i).at(k).first << endl;
        // out1[i * X->M * X->N + j * X->N + k] = (unsigned int)X->data[i * X->M * X->N + j * X->N + k];
      }
    }
  }

  // Encrypt Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        // cout << "Before : " << Y->data[i * Y->M * Y->N + j * Y->N + k] << endl;
        out2[i * Y->M * Y->N + j * Y->N + k] = (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * key_b.at(i).at(j).first + key_d.at(i).at(k).first;
        // cout << "After : " << out2[i * Y->M * Y->N + j * Y->N + k] << endl;
        // cout << "Used keys : " << key_b.at(i).at(j).first << " " << key_d.at(i).at(k).first << endl; 
        // exit(-1);
        // out2[i * Y->M * Y->N + j * Y->N + k] = (unsigned int)Y->data[i * Y->M * Y->N + j * Y->N + k];
      }
    }
  }

  // auto end = std::chrono::steady_clock::now();
  // std::cout << "Time taken for encryption: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

void Ex_Generate_Decryption_Key_QK_Int32(int src_id1, int src_id2) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N
  // auto start = std::chrono::steady_clock::now();
  assert(X->B == Y->B);
  // compute x_row_factor, y_col_sum
  for (int i = 0; i < X->B; ++i) {
    z_row_factor.at(i).clear();
    z_col_factor.at(i).clear();

    for (int j = 0; j < X->M; ++j) {
      uint32_t sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_d.at(i).at(k).first;
      }
      sum *= key_a.at(i).at(j).first;
      z_row_factor.at(i).push_back(sum);
    }
    assert (z_row_factor.at(i).size() == X->M);

    for (int j = 0; j < Y->M; ++j) {
      uint32_t sum = 0;
      for (int k = 0; k < Y->N; ++k) {
        sum += (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * key_c.at(i).at(k).first;
      }
      sum *= key_b.at(i).at(j).first;
      z_col_factor.at(i).push_back(sum);
    }

    assert (z_col_factor.at(i).size() == Y->M);

    uint32_t sum = 0;
    for (int j = 0; j < X->N; ++j) {
      sum += key_c.at(i).at(j).first * key_d.at(i).at(j).first;
    }
    z_dot_product_factor.at(i) = sum;
  }
  // auto end = std::chrono::steady_clock::now();
  // std::cout << "Time taken for decryption key generation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

int Ex_Set_Decrypted_Tensor_QK_Int32(unsigned int* data, int B, int M, int N) {
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);

  // struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        uint32_t tmp = data[i * M * N + j * N + k] - z_row_factor.at(i).at(j) - z_col_factor.at(i).at(k) - z_dot_product_factor.at(i);
        tmp = (tmp * mult_key_inv_precompute.at(key_a.at(i).at(j).second).at(key_b.at(i).at(k).second)) % MODULO;
        tensor->data[i * M * N + j * N + k] = (int)tmp;
      }
    }
  }

  // Undo shift 
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        int undo_shift_factor = SHIFT_AMT * (x_row_sum_buffer.at(i).at(j) + y_col_sum_buffer.at(i).at(k) + share_dim * SHIFT_AMT);
        tensor->data[i * M * N + j * N + k] -= undo_shift_factor;
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

void Ex_Get_Encrypted_Tensor_PV_Int32(int src_id1, int src_id2, unsigned int* out1,
                                      unsigned int* out2) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  // struct TensorInt32* u = CreateTensorInt32(X->B, 1, X->N);
  // GetCPRNG((unsigned char*)u->data, u->num_bytes);
  // struct TensorInt32* v = CreateTensorInt32(X->B, 1, Y->M);
  // GetCPRNG((unsigned char*)v->data, v->num_bytes);

  share_dim = X->N;
  // Compute x_row_sum
  for (int i = 0; i < X->B; ++i) {
    x_row_sum_buffer.at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      int sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += X->data[i * X->M * X->N + j * X->N + k];
      }
      x_row_sum_buffer.at(i).push_back(sum);
    }
  }

  // Compute y_row_sum
  for (int i = 0; i < Y->B; ++i) {
    y_col_sum_buffer.at(i).clear();
    for (int j = 0; j < Y->M; ++j) {
      int sum = 0;
      for (int k = 0; k < Y->N; ++k) {
        sum += Y->data[i * Y->M * Y->N + j * Y->N + k];
      }
      y_col_sum_buffer.at(i).push_back(sum);
    }
  }

  // Shift X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        X->data[i * X->M * X->N + j * X->N + k] += SHIFT_AMT;
      }
    }
  }

  // Shift Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        Y->data[i * Y->M * Y->N + j * Y->N + k] += SHIFT_AMT;
      }
    }
  }

  // Randomly sample random keys
  for (int i = 0; i < X->B; ++i) {
    key_a.at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_a.at(i).emplace_back(mult_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < Y->B; ++i) {
    key_b.at(i).clear();
    for (int j = 0; j < Y->M; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_b.at(i).emplace_back(mult_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < X->B; ++i) {
    key_c.at(i).clear();
    for (int j = 0; j < X->N; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_c.at(i).emplace_back(add_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < Y->B; ++i) {
    key_d.at(i).clear();
    for (int j = 0; j < Y->N; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_d.at(i).emplace_back(add_key_pool.at(idx), idx);
    }
  }

  // Encrypt X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        out1[i * X->M * X->N + j * X->N + k] = (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_a.at(i).at(j).first + key_c.at(i).at(k).first;
      }
    }
  }

  // Encrypt Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        out2[i * Y->M * Y->N + j * Y->N + k] = (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * key_b.at(i).at(j).first + key_d.at(i).at(k).first;
      }
    }
  }
}

void Ex_Generate_Decryption_Key_PV_Int32(int src_id1, int src_id2) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N
  // auto start = std::chrono::steady_clock::now();
  // jump
  assert(X->B == Y->B);
  // compute x_row_factor, y_col_sum
  for (int i = 0; i < X->B; ++i) {
    z_row_factor.at(i).clear();
    z_col_factor.at(i).clear();

    for (int j = 0; j < X->M; ++j) {
      uint32_t sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_d.at(i).at(k).first;
      }
      sum *= key_a.at(i).at(j).first;
      z_row_factor.at(i).push_back(sum);
    }
    assert (z_row_factor.at(i).size() == X->M);

    for (int j = 0; j < Y->M; ++j) {
      uint32_t sum = 0;
      for (int k = 0; k < Y->N; ++k) {
        sum += (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * key_c.at(i).at(k).first;
      }
      sum *= key_b.at(i).at(j).first;
      z_col_factor.at(i).push_back(sum);
    }

    assert (z_col_factor.at(i).size() == Y->M);

    uint32_t sum = 0;
    for (int j = 0; j < X->N; ++j) {
      sum += key_c.at(i).at(j).first * key_d.at(i).at(j).first;
    }
    z_dot_product_factor.at(i) = sum;
  }
}

int Ex_Set_Decrypted_Tensor_PV_Int32(unsigned int* data, int B, int M, int N) {
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);

  // struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        uint32_t tmp = data[i * M * N + j * N + k] - z_row_factor.at(i).at(j) - z_col_factor.at(i).at(k) - z_dot_product_factor.at(i);
        tmp = (tmp * mult_key_inv_precompute.at(key_a.at(i).at(j).second).at(key_b.at(i).at(k).second)) % MODULO;
        tensor->data[i * M * N + j * N + k] = (int)tmp;
      }
    }
  }

  // Undo shift 
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        int undo_shift_factor = SHIFT_AMT * (x_row_sum_buffer.at(i).at(j) + y_col_sum_buffer.at(i).at(k) + share_dim * SHIFT_AMT);
        tensor->data[i * M * N + j * N + k] -= undo_shift_factor;
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

void Ex_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                   uint32_t* out1, uint32_t* out2,
                                                   int layer_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K


  // x_row_sum[b][x_m] = sum(X[b][x_m][x_n]), must do it every time, x does not reused
  // y_col_sum[b][y_m] = sum(Y[b][y_m][y_n]), compute new part and concat

  // Compute X_Row_Sum

    // Later, must separate to handle different seq_len for each batch 
  share_dim = X->N; // for QK, it is fixed

  for (int i = 0; i < X->B; ++i) {
    x_row_sum_buffer_opt_list[layer_id].at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      int sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += X->data[i * X->M * X->N + j * X->N + k];
      }
      x_row_sum_buffer_opt_list[layer_id].at(i).push_back(sum);
    }
  }

  // Compute Y_Col_sum
  for (int i = 0; i < Y->B; ++i) {
    // y_col_sum_buffer.at(i).clear(); // dont clear, must be stacked
    for (int j = 0; j < Y->M; ++j) {
      int sum = 0;
      for (int k = 0; k < Y->N; ++k) {
        sum += Y->data[i * Y->M * Y->N + j * Y->N + k];
      }
      y_col_sum_buffer_opt_list[layer_id].at(i).push_back(sum);
    }
  }

  // Shift X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        X->data[i * X->M * X->N + j * X->N + k] += SHIFT_AMT;
      }
    }
  }

  // Shift Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        Y->data[i * Y->M * Y->N + j * Y->N + k] += SHIFT_AMT;
      }
    }
  }

  // Randomly sample random keys, must be sampled every time
  for (int i = 0; i < X->B; ++i) {
    key_a_opt_list[layer_id].at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_a_opt_list[layer_id].at(i).emplace_back(mult_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < Y->B; ++i) {
    // key_b.at(i).clear(); // dont clear, must be stacked
    for (int j = 0; j < Y->M; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_b_opt_list[layer_id].at(i).emplace_back(mult_key_pool.at(idx), idx);
    }
  }

  // Keys c and d only sampled first time, D does not increase
  for (int i = 0; i < X->B; ++i) {
    if (key_c_opt_list[layer_id].at(i).empty()) {
      for (int j = 0; j < X->N; ++j) {
        uint32_t idx = GenerateRandomNumber_Uint32();
        idx = idx % SECRET_KEY_POOL_SIZE;
        key_c_opt_list[layer_id].at(i).emplace_back(add_key_pool.at(idx), idx);
      }
    } 
  }

  for (int i = 0; i < Y->B; ++i) {
    if (key_d_opt_list[layer_id].at(i).empty()) {
      for (int j = 0; j < Y->N; ++j) {
        uint32_t idx = GenerateRandomNumber_Uint32();
        idx = idx % SECRET_KEY_POOL_SIZE;
        key_d_opt_list[layer_id].at(i).emplace_back(add_key_pool.at(idx), idx);
      }
    }
  }

  // Encrypt X as usual
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      uint32_t key_a = key_a_opt_list[layer_id].at(i).at(j).first;
      for (int k = 0; k < X->N; ++k) {
        // out1[i * X->M * X->N + j * X->N + k] = (uint32_t)X->data[i * X->M * X->N + j * X->N + k];
        out1[i * X->M * X->N + j * X->N + k] = (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_a + key_c_opt_list[layer_id].at(i).at(k).first;
      }
    }
  }

  // Encrypt Y must use the last b BE CAREFUL!!!, Only during generation
  for (int i = 0; i < Y->B; ++i) {
    bool is_gen = Y->M == 1; // if Y_M is 1, then it is generation phase  
    for (int j = 0; j < Y->M; ++j) {
      uint32_t b = is_gen ? key_b_opt_list[layer_id].at(i).back().first : key_b_opt_list[layer_id].at(i).at(j).first;
      for (int k = 0; k < Y->N; ++k) {
        // out2[i * Y->M * Y->N + j * Y->N + k] = (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k];
        out2[i * Y->M * Y->N + j * Y->N + k] = (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * b + key_d_opt_list[layer_id].at(i).at(k).first;
      }
    }
  }
}

void Ex_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                     int layer_id) {
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N

  // z_row_factor is used once
  // z_col_factor must be stacked

  for (int i = 0; i < X->B; ++i) {
    z_row_factor_opt_list[layer_id].at(i).clear();

    // Z COL FACTOR MUST BE STACKED
    // z_col_factor_opt_list[layer_id].at(i).clear(); 

    for (int j = 0; j < X->M; ++j) {
      uint32_t sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_d_opt_list[layer_id].at(i).at(k).first;
      }
      sum *= key_a_opt_list[layer_id].at(i).at(j).first;
      z_row_factor_opt_list[layer_id].at(i).push_back(sum);
    }

    // Z COL FACTOR MUST BE STACKED
    for (int j = 0; j < Y->M; ++j) {
      uint32_t sum = 0;
      for (int k = 0; k < Y->N; ++k) {
        sum += (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * key_c_opt_list[layer_id].at(i).at(k).first;
      }
      bool is_gen = Y->M == 1; // if Y_M is 1, then it is generation phase
      uint32_t b = is_gen ? key_b_opt_list[layer_id].at(i).back().first : key_b_opt_list[layer_id].at(i).at(j).first;

      sum *= b; // use the last one, during gen
      z_col_factor_opt_list[layer_id].at(i).push_back(sum);
    }
  }

  if (z_dot_product_factor_opt_list_done[layer_id] == false) {
    for (int i = 0; i < X->B; ++i) {
      uint32_t sum = 0;
      for (int j = 0; j < X->N; ++j) {
        sum += key_c_opt_list[layer_id].at(i).at(j).first * key_d_opt_list[layer_id].at(i).at(j).first;
      }
      z_dot_product_factor_opt_list[layer_id].at(i) = sum;
    }
    z_dot_product_factor_opt_list_done[layer_id] = true;
  }
}

int Ex_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(uint32_t* data, int B, int M,
                                                  int N, int layer_id) {
  // printf("B %d M %d N %d\n", B, M, N);
  // struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];
  // auto start  = std::chrono::steady_clock::now();
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);

  // bool is_gen = M == 1; // if Y_M is 1, then it is generation phase

  for (int i = 0; i < B; ++i) {
    uint32_t z_dot_product_factor = z_dot_product_factor_opt_list[layer_id].at(i);
    for (int j = 0; j < M; ++j) {
      uint32_t sub_factor = z_row_factor_opt_list[layer_id].at(i).at(j) + z_dot_product_factor;
      int a_index = key_a_opt_list[layer_id].at(i).at(j).second;
      for (int k = 0; k < N; ++k) {
        int b_index = key_b_opt_list[layer_id].at(i).at(k).second;
        uint32_t tmp = data[i * M * N + j * N + k] - z_col_factor_opt_list[layer_id].at(i).at(k) - sub_factor;
        tmp *= mult_key_inv_precompute.at(a_index).at(b_index);
        tensor->data[i * M * N + j * N + k] = (int) tmp;
      }
    }
  }

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      int sub_factor = x_row_sum_buffer_opt_list[layer_id].at(i).at(j) + share_dim * SHIFT_AMT;
      for (int k = 0; k < N; ++k) {
        int undo_shift_factor = SHIFT_AMT * (y_col_sum_buffer_opt_list[layer_id].at(i).at(k) + sub_factor);
        tensor->data[i * M * N + j * N + k] -= undo_shift_factor;
      }
    }
  }
  // auto end = std::chrono::steady_clock::now();
  // std::cout << "Time taken for decryption: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }
  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

void Ex_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                   uint32_t* out1, uint32_t* out2,
                                                   int layer_id) {
  // auto start = std::chrono::steady_clock::now();
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  share_dim_pv_opt = X->N; // for PV, it is increasing
  
  // Compute X_Row_Sum
  for (int i = 0; i < X->B; ++i) {
    x_row_sum_buffer_opt_list2[layer_id].at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      int sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += X->data[i * X->M * X->N + j * X->N + k];
      }
      x_row_sum_buffer_opt_list2[layer_id].at(i).push_back(sum);
    }
  }

  // cout << "X dim : " << X->B << " " << X->M << " " << X->N << endl; // " " << X->M << " " << X->N << "
  // cout << "Y dim : " << Y->B << " " << Y->M << " " << Y->N << endl;
  // Compute Y_Col_Sum, you dont grow, but accumulate
  for (int i = 0; i < Y->B; ++i) {
    // y_col_sum_buffer_opt_list[layer_id].at(i).clear();
    if (y_col_sum_buffer_opt_list2[layer_id].at(i).empty()) {
      for (int j = 0; j < Y->M; ++j) {
        y_col_sum_buffer_opt_list2[layer_id].at(i).push_back(0);
      }
      assert (y_col_sum_buffer_opt_list2[layer_id].at(i).size() == Y->M);
    }

    for (int j = 0; j < Y->M; ++j) {
      int sum = 0;
      for (int k = 0; k < Y->N; ++k) {
        sum += Y->data[i * Y->M * Y->N + j * Y->N + k];
      }
      y_col_sum_buffer_opt_list2[layer_id].at(i).at(j) += sum;
      // cout << j << " : " <<  y_col_sum_buffer_opt_list2[layer_id].at(i).at(j) << "\n";
    }
  }

  // SHIFT x
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      for (int k = 0; k < X->N; ++k) {
        X->data[i * X->M * X->N + j * X->N + k] += SHIFT_AMT;
        // out1[i * X->M * X->N + j * X->N + k] = (uint32_t)X->data[i * X->M * X->N + j * X->N + k];
      }
    }
  }

  // SHIFT Y
  for (int i = 0; i < Y->B; ++i) {
    for (int j = 0; j < Y->M; ++j) {
      for (int k = 0; k < Y->N; ++k) {
        Y->data[i * Y->M * Y->N + j * Y->N + k] += SHIFT_AMT;
        // out2[i * Y->M * Y->N + j * Y->N + k] = (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k];
      }
    }
  }

  for (int i = 0; i < X->B; ++i) {
    key_a_opt_list2[layer_id].at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_a_opt_list2[layer_id].at(i).emplace_back(mult_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < Y->B; ++i) {
    // you only sample one time, Y->B does not increase
    if (key_b_opt_list2[layer_id].at(i).empty()) {
      for (int j = 0; j < Y->M; ++j) {
        uint32_t idx = GenerateRandomNumber_Uint32();
        idx = idx % SECRET_KEY_POOL_SIZE;
        key_b_opt_list2[layer_id].at(i).emplace_back(mult_key_pool.at(idx), idx);
      }
    }
  }

  for (int i = 0; i < X->B; ++i) {
    // it increases 
    if (key_c_opt_list2[layer_id].at(i).empty()) {
      for (int j = 0; j < X->N; ++j) {
        uint32_t idx = GenerateRandomNumber_Uint32();
        idx = idx % SECRET_KEY_POOL_SIZE;
        key_c_opt_list2[layer_id].at(i).emplace_back(add_key_pool.at(idx), idx);
      }
    } else {
      // sample 1
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_c_opt_list2[layer_id].at(i).emplace_back(add_key_pool.at(idx), idx);
    }
  }

  for (int i = 0; i < Y->B; ++i) {
    if (key_d_opt_list2[layer_id].at(i).empty()) {
      for (int j = 0; j < Y->N; ++j) {
        uint32_t idx = GenerateRandomNumber_Uint32();
        idx = idx % SECRET_KEY_POOL_SIZE;
        key_d_opt_list2[layer_id].at(i).emplace_back(add_key_pool.at(idx), idx);
      }
    } else {
      // sample 1
      uint32_t idx = GenerateRandomNumber_Uint32();
      idx = idx % SECRET_KEY_POOL_SIZE;
      key_d_opt_list2[layer_id].at(i).emplace_back(add_key_pool.at(idx), idx);
    }
  }

  // Encrypt X
  for (int i = 0; i < X->B; ++i) {
    for (int j = 0; j < X->M; ++j) {
      uint32_t key_a = key_a_opt_list2[layer_id].at(i).at(j).first;
      for (int k = 0; k < X->N; ++k) {
        out1[i * X->M * X->N + j * X->N + k] = (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_a + key_c_opt_list2[layer_id].at(i).at(k).first;
      }
    }
  }

  // Encrypt Y
  for (int i = 0; i < Y->B; ++i) {
    bool is_gen = Y->N == 1; // if Y_N is 1, then it is generation phase
    for (int j = 0; j < Y->M; ++j) {
      uint32_t key_b = key_b_opt_list2[layer_id].at(i).at(j).first;
      for (int k = 0; k < Y->N; ++k) {
        uint32_t d = is_gen ? key_d_opt_list2[layer_id].at(i).back().first : key_d_opt_list2[layer_id].at(i).at(k).first;
        out2[i * Y->M * Y->N + j * Y->N + k] = (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * key_b + d;
      }
    }
  }
}

void Ex_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                     int layer_id) {
  // auto start = std::chrono::steady_clock::now();
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B, X_M, X_N
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B, Y_M, Y_N
 
  for (int i = 0; i < X->B; ++i) {
    // z_row_factor is trivial
    z_row_factor_opt_list2[layer_id].at(i).clear();
    for (int j = 0; j < X->M; ++j) {
      uint32_t sum = 0;
      for (int k = 0; k < X->N; ++k) {
        sum += (uint32_t)X->data[i * X->M * X->N + j * X->N + k] * key_d_opt_list2[layer_id].at(i).at(k).first;
      }
      sum *= key_a_opt_list2[layer_id].at(i).at(j).first;
      z_row_factor_opt_list2[layer_id].at(i).push_back(sum);
    }

    // Not stacked like QK but updated
    if (z_col_factor_opt_list2[layer_id].at(i).empty()) {
      for (int j = 0; j < Y->M; ++j) {
        z_col_factor_opt_list2[layer_id].at(i).push_back(0);
      }
    }

    for (int j = 0; j < Y->M; ++j) {
      uint32_t sum = 0;

      bool is_gen = Y->N == 1; // if Y_N is 1, then it is generation phase
      for (int k = 0; k < Y->N; ++k) {
        uint32_t c = is_gen ? key_c_opt_list2[layer_id].at(i).back().first : key_c_opt_list2[layer_id].at(i).at(k).first;
        sum += (uint32_t)Y->data[i * Y->M * Y->N + j * Y->N + k] * c;
      }
      sum *= key_b_opt_list2[layer_id].at(i).at(j).first;
      z_col_factor_opt_list2[layer_id].at(i).at(j) += sum;
    }
  }

  if (z_dot_product_factor_opt_list_done2[layer_id] == false) {
    for (int i = 0; i < X->B; ++i) {
      uint32_t sum = 0;
      for (int j = 0; j < X->N; ++j) {
        sum += key_c_opt_list2[layer_id].at(i).at(j).first * key_d_opt_list2[layer_id].at(i).at(j).first;
      }
      z_dot_product_factor_opt_list2[layer_id].at(i) = sum;
    }
    z_dot_product_factor_opt_list_done2[layer_id] = true;
  } else {
    // update with newly added add keys
    for (int i = 0; i < X->B; ++i) {
      z_dot_product_factor_opt_list2[layer_id].at(i) += key_c_opt_list2[layer_id].at(i).back().first * key_d_opt_list2[layer_id].at(i).back().first;
    }
  }
}

int Ex_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(uint32_t* data, int B, int M,
                                                  int N, int layer_id) {
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);
  // struct TensorInt32* decryption_key = tensor_int32_list[decryption_key_id];

  for (int i = 0; i < B; ++i) {
    uint32_t z_dot_product_factor = z_dot_product_factor_opt_list2[layer_id].at(i);
    for (int j = 0; j < M; ++j) {
      int a_index = key_a_opt_list2[layer_id].at(i).at(j).second;
      uint32_t sub_factor = z_row_factor_opt_list2[layer_id].at(i).at(j) + z_dot_product_factor;
      for (int k = 0; k < N; ++k) {
        int b_index = key_b_opt_list2[layer_id].at(i).at(k).second;
        uint32_t tmp = data[i * M * N + j * N + k] - z_col_factor_opt_list2[layer_id].at(i).at(k) - sub_factor;
        tmp *= mult_key_inv_precompute.at(a_index).at(b_index);
        tensor->data[i * M * N + j * N + k] = (int) tmp;
      }
    }
  }

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      int sub_factor = x_row_sum_buffer_opt_list2[layer_id].at(i).at(j) + share_dim_pv_opt * SHIFT_AMT;
      for (int k = 0; k < N; ++k) {
        int undo_shift_factor = SHIFT_AMT * (y_col_sum_buffer_opt_list2[layer_id].at(i).at(k) + sub_factor);
        tensor->data[i * M * N + j * N + k] -= undo_shift_factor;
        // tensor->data[i * M * N + j * N + k] = data[i * M * N + j * N + k];
      }
    }
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
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
  for (int i = 0; i < total_elements; ++i) {
    dst_tensor->data[i] = std::max(src_tensor->data[i], 0.0f);
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


  int total_elements = B * M * N; // be careful of overflow
  for (int i = 0; i < total_elements; ++i) {
    dst_tensor->data[i] = (int8_t)round(src_tensor->data[i] * 127.0f);
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
  
  for (int i = 0; i < total_elements; ++i) {
    dst_tensor->data[i] = (int8_t)src_tensor->data[i];
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
  for (int i = 0; i < total_elements; ++i) {
    dst_tensor->data[i] = (int)src_tensor->data[i];
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

  for (int i = 0; i < total_elements; ++i) {
    dst_tensor->data[i] = (int32_t)src_tensor->data[i]; 
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

void Ex_Get_Tensor_Int8(int src_id, int8_t* out) {
  struct TensorInt8* src_tensor = tensor_int8_list[src_id];
  int total_elements = src_tensor->B * src_tensor->M * src_tensor->N;

  for (int i = 0; i < total_elements; ++i) {
    out[i] = src_tensor->data[i];
  }
}

int Ex_Set_Tensor_Int8(int8_t* data, int B, int M, int N) {
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

  for (int i = 0; i < total_elements; ++i) {
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

  int total_elements = B * M * N; // be careful of overflow
  for (int i = 0; i < total_elements; ++i) {
    dst_tensor->data[i] = residual_tensor->data[i] + hidden_states_tensor->data[i];
  }

  int curr_id = tensor_float_id;
  tensor_float_list[curr_id] = dst_tensor;
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_CPU_Bmm(int src_id1, int src_id2) {
  // auto start = std::chrono::steady_clock::now();
  struct TensorInt32* X = tensor_int32_list[src_id1];  // B x M x K
  struct TensorInt32* Y = tensor_int32_list[src_id2];  // B x N x K

  // printf("X CPU BMM: %d %d %d\n", X->B, X->M, X->N);
  // printf("Y CPU BMM: %d %d %d\n", Y->B, Y->N, Y->M);

  // measure time with steady clock from chrono library
  struct TensorInt32* Z = MatmulS32S32S32_Naive(X, Y);  // B x M x N
  // printf("CPU BMM Time: %f\n", elapsed_seconds.count());

  if (tensor_int32_list[tensor_int32_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[tensor_int32_id]);
  }

  int curr_id = tensor_int32_id;
  tensor_int32_list[curr_id] = Z;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;

  // auto end = std::chrono::steady_clock::now();
  // auto diff = end - start;
  // printf("CPU BMM time: %lld\n", std::chrono::duration_cast<std::chrono::microseconds>(diff).count());
  return curr_id;
}

}