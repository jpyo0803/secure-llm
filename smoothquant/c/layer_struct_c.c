#include "layer_struct_c.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "aes_stream.h"

int Ex_Set_Hidden_States(float* hidden_states, int B, int M, int N) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  tensor_float_list[curr_id] = CreateTensorFloatFromData(hidden_states, B, M, N);
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Copy_Hidden_States(int src_id) {
  int curr_id = tensor_float_id;
  if (tensor_float_list[curr_id] != NULL) {
    DeleteTensorFloat(tensor_float_list[curr_id]);
  }

  struct TensorFloat* src_tensor = tensor_float_list[src_id];
  tensor_float_list[curr_id] = CreateTensorFloat(src_tensor->B, src_tensor->M, src_tensor->N);
  for (int i = 0; i < src_tensor->B * src_tensor->M * src_tensor->N; i++) {
    tensor_float_list[curr_id]->data[i] = src_tensor->data[i];
  }
  tensor_float_id = (tensor_float_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps) {
  int curr_id = layer_norm_param_id;

  struct LayerNormParam* layer_norm_param = (struct LayerNormParam*)malloc(sizeof(struct LayerNormParam));
  layer_norm_param->gamma = CreateTensorFloatFromData(gamma, 1, 1, N);
  layer_norm_param->beta = CreateTensorFloatFromData(beta, 1, 1, N);
  layer_norm_param->eps = eps;

  layer_norm_param_list[curr_id] = layer_norm_param;
  layer_norm_param_id = (layer_norm_param_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Layer_Norm_Q(int src_id, int layer_norm_param_id) {
  int curr_id = tensor_int32_id;

  struct TensorFloat* src_tensor = tensor_float_list[src_id];
  struct LayerNormParam* layer_norm_param = layer_norm_param_list[layer_norm_param_id];

  struct TensorInt32* dst_tensor = CreateTensorInt32(src_tensor->B, src_tensor->M, src_tensor->N);

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = 0.0;
      float sum_sqr = 0.0;
      for (int k = 0; k < N; ++k) {
        float tmp = src_tensor->data[i * M * N + j * N + k];
        sum += tmp;
        sum_sqr += tmp * tmp;
      }
      float mean = sum / N;
      float var = sum_sqr / N - mean * mean;

      for (int k = 0; k < N; ++k) {
        float tmp = src_tensor->data[i * M * N + j * N + k];
        tmp = (tmp - mean) / sqrtf(var + layer_norm_param->eps) * layer_norm_param->gamma->data[k] + layer_norm_param->beta->data[k];
        tmp = roundf(tmp);
        // Now clamp between -128 and 127
        if (tmp > 127.0) {
          tmp = 127.0;
        } else if (tmp < -128.0) {
          tmp = -128.0;
        }
        dst_tensor->data[i * M * N + j * N + k] = (int)tmp;
      }
    }
  }

  tensor_int32_list[curr_id] = dst_tensor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}

int Ex_Set_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N, float alpha, float beta) {
  int curr_id = linear_param_ws8bs8_id;

  struct LinearParamWS8BS8* linear_param = (struct LinearParamWS8BS8*)malloc(sizeof(struct LinearParamWS8BS8));
  linear_param->weight = CreateTensorInt8FromData(weight, 1, M, N);
  linear_param->bias = CreateTensorInt8FromData(bias, 1, 1, N); 
  linear_param->alpha = alpha;
  linear_param->beta = beta;

  linear_param_wb8bs8_list[curr_id] = linear_param;
  linear_param_ws8bs8_id = (linear_param_ws8bs8_id + 1) % DYNAMIC_LIST_LEN;

  return curr_id;
}

void Ex_Get_Tensor_Dim_Int32(int src_id, int* B, int* M, int *N) {
  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  *B = src_tensor->B;
  *M = src_tensor->M;
  *N = src_tensor->N;
}

void Ex_Get_Tensor(int src_id, int* out) {
  struct TensorInt32* src_tensor = tensor_int32_list[src_id];
  for (int i = 0; i < src_tensor->B * src_tensor->M * src_tensor->N; i++) {
    out[i] = src_tensor->data[i];
  }
}

// Need to return blind factor id
int Ex_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int* out) { 
  Ex_Get_Input(src_id, out);

  struct TensorInt32* src_tensor = tensor_int32_list[src_id];

  int B = src_tensor->B;
  int M = src_tensor->M;
  int N = src_tensor->N;

  int curr_id = tensor_int32_id;
  struct TensorInt32* blind_factor = CreateTensorInt32(B, 1, N);

  GetCPRNG((unsigned char*)blind_factor->data, blind_factor->num_bytes);

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < N; ++k) {
        out[i * M * N + j * N + k] += blind_factor->data[i * N + k];
      }
    }
  }

  tensor_int32_list[curr_id] = blind_factor;
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
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

int Ex_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int blind_factor_id, int layer_weight_id) {
  int curr_id = tensor_int32_id;
  if (tensor_int32_list[curr_id] != NULL) {
    DeleteTensorInt32(tensor_int32_list[curr_id]);
  }

  // Compute Unblinding factor
  struct* 

  tensor_int32_list[curr_id] = CreateTensorInt32FromData(data, B, M, N);
  tensor_int32_id = (tensor_int32_id + 1) % DYNAMIC_LIST_LEN;
  return curr_id;
}