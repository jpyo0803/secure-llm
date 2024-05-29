
#include "sgx_layer_struct.h"
#include "Enclave.h"
#include <cmath>
#include "c/layer_struct_c.h"

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

void ecall_Hidden_States(float* hidden_states, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Hidden_States(hidden_states,B,M,N);
}

void ecall_Sgx_Copy_Hidden_States(int src_id, int* ret_id) {
  *ret_id = Sgx_Copy_Hidden_States(src_id);
}

void ecall_Layer_Norm_Param(float* gamma, float* beta, int N, float eps, int* ret_id) {
  *ret_id = Sgx_Set_Layer_Norm_Param(gamma,beta,N,eps);
}

void ecall_Layer_Norm_Q(int src_id, int layer_norm_param_id) {
  Sgx_Layer_Norm_Q(src_id,layer_norm_param_id);
}

void ecall_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N, float alpha, float beta, int* ret_id) {
  *ret_id = Sgx_Set_Linear_Param_WS8BS8(weight,bias,M,N,alpha,beta);
}

void ecall_Linear_Param_WS8BFP32(char* weight, float* bias, int M, int N, float alpha, int* ret_id) {
  *ret_id = Sgx_Set_Linear_Param_WS8BFP32(weight,bias,M,N,alpha);
}

void ecall_Get_Tensor_Dim_Int32(int src_id, int* dim) {
  Sgx_Get_Tensor_Dim_Int32(src_id,dim);
}

void ecall_Get_Tensor_Int32(int src_id, int* out) {
  Sgx_Get_Tensor_Int32(src_id,out);
}

void ecall_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int* out, int* ret_id) {
  *ret_id = Sgx_Get_Encrypted_Tensor_Opr1_Int32(src_id,out);
}

void ecall_Set_Tensor_Int32(int* data, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Tensor_Int32(data,B,M,N);
}

void ecall_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int blind_factor_id, int linear_param_id, int* ret_id) {
  *ret_id = Sgx_Set_Decrypted_Tensor_Opr1_Int32(data,B,M,N,blind_factor_id,linear_param_id);
}

void ecall_Get_Encrypted_Tensor_Opr2_Int32(int src_id1, int src_id2, int* out1, int* out2, int* ret_id) {
  *ret_id = Sgx_Get_Encrypted_Tensor_Opr2_Int32(src_id1,src_id2,out1,out2);
}

void ecall_Set_Decrypted_Tensor_Opr2_Int32(int* data, int B, int M, int N, int unblind_factor_id, int* ret_id) {
  *ret_id = Sgx_Set_Decrypted_Tensor_Opr2_Int32(data,B,M,N,unblind_factor_id);
}

void ecall_Get_Tensor_Dim_Int8(int src_id, int* dim) {
  Sgx_Get_Tensor_Dim_Int8(src_id,dim);
}

void ecall_Get_Tensor_Int8(int src_id, char* out) {
  Sgx_Get_Tensor_Int8(src_id,out);
}

void ecall_Set_Tensor_Int8(char* data, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Tensor_Int8(data,B,M,N);
}

void ecall_Get_Tensor_Dim_Float(int src_id, int* dim) {
  Sgx_Get_Tensor_Dim_Float(src_id,dim);
}

void ecall_Get_Tensor_Float(int src_id, float* out) {
  Sgx_Get_Tensor_Float(src_id,out);
}

void ecall_Set_Tensor_Float(float* data, int B, int M, int N, int* ret_id) {
  *ret_id = Sgx_Set_Tensor_Float(data,B,M,N);
}

void ecall_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id, int* ret_id) {
  *ret_id = Sgx_Compute_Epilogue_WS8BS8(src_id,linear_param_id);
}

void ecall_Compute_Epilogue_WS8BFP32(int src_id, int linear_param_id, int* ret_id) {
  *ret_id = Sgx_Compute_Epilogue_WS8BFP32(src_id,linear_param_id);
}

void ecall_Compute_Epilogue_BMM(int src_id, int bmm_param_id, int* ret_id) {
  *ret_id = Sgx_Compute_Epilogue_BMM(src_id,bmm_param_id);
}

void ecall_ReLU(int src_id, int* ret_id) {
  *ret_id = Sgx_ReLU(src_id);
}

void ecall_Softmax(int src_id, int* ret_id) {
  *ret_id = Sgx_Softmax(src_id);
}

void ecall_Quantize_Post_Softmax(int src_id, int* ret_id) {
  *ret_id = Sgx_Quantize_Post_Softmax(src_id);
}

void ecall_Cast_From_Float_To_Int8(int src_id, int* ret_id) {
  *ret_id = Sgx_Cast_From_Float_To_Int8(src_id);
}

void ecall_Cast_From_Float_To_Int32(int src_id, int* ret_id) {
  *ret_id = Sgx_Cast_From_Float_To_Int32(src_id);
}

void ecall_Cast_From_Int8_To_Int32(int src_id, int* ret_id) {
  *ret_id = Sgx_Cast_From_Int8_To_Int32(src_id);
}

void ecall_Set_Bmm_Param(float alpha, int* ret_id) {
  *ret_id = Sgx_Set_Bmm_Param(alpha);
}

void ecall_Residual_Add(int residual, int hidden_states, int* ret_id) {
  *ret_id = Sgx_Residual_Add(residual,hidden_states);
}


}