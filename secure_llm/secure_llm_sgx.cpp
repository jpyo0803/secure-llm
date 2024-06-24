#ifndef SECURE_LLM_SECURE_LLM_SECURE_LLM_SGX_WRAPPER_H
#define SECURE_LLM_SECURE_LLM_SECURE_LLM_SGX_WRAPPER_H

/*
  This is secure_llm ecall proxy
*/

#include "secure_llm_sgx.h"
#include "secure_llm.h"


extern "C" {

void ecall_Sgx_Get_Encrypted_Tensor_QK_Int32(int src_id1, int src_id2, uint32_t* out1, uint32_t* out2) {
  Ex_Get_Encrypted_Tensor_QK_Int32(src_id1,src_id2,out1,out2);
}

void ecall_Sgx_Generate_Decryption_Key_QK_Int32(int src_id1, int src_id2) {
  Ex_Generate_Decryption_Key_QK_Int32(src_id1,src_id2);
}

void ecall_Sgx_Set_Decrypted_Tensor_QK_Int32(uint32_t* data, int B, int M, int N, int* ret_id) {
  *ret_id = Ex_Set_Decrypted_Tensor_QK_Int32(data,B,M,N);
}

void ecall_Sgx_Get_Encrypted_Tensor_PV_Int32(int src_id1, int src_id2, uint32_t* out1, uint32_t* out2) {
  Ex_Get_Encrypted_Tensor_PV_Int32(src_id1,src_id2,out1,out2);
}

void ecall_Sgx_Generate_Decryption_Key_PV_Int32(int src_id1, int src_id2) {
  Ex_Generate_Decryption_Key_PV_Int32(src_id1,src_id2);
}

void ecall_Sgx_Set_Decrypted_Tensor_PV_Int32(uint32_t* data, int B, int M, int N, int* ret_id) {
  *ret_id = Ex_Set_Decrypted_Tensor_PV_Int32(data,B,M,N);
}

void ecall_Sgx_Set_Hidden_States(float* hidden_states, int B, int M, int N, int* ret_id) {
  *ret_id = Ex_Set_Hidden_States(hidden_states,B,M,N);
}

void ecall_Sgx_Copy_Hidden_States(int src_id, int* ret_id) {
  *ret_id = Ex_Copy_Hidden_States(src_id);
}

void ecall_Sgx_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps, int* ret_id) {
  *ret_id = Ex_Set_Layer_Norm_Param(gamma,beta,N,eps);
}

void ecall_Sgx_Layer_Norm_Q(int src_id, int layer_norm_param_id, int* ret_id) {
  *ret_id = Ex_Layer_Norm_Q(src_id,layer_norm_param_id);
}

void ecall_Sgx_Set_Linear_Param_WS8BS8(int8_t* weight, int8_t* bias, int M, int N, float alpha, float beta, int* ret_id) {
  *ret_id = Ex_Set_Linear_Param_WS8BS8(weight,bias,M,N,alpha,beta);
}

void ecall_Sgx_Set_Linear_Param_WS8BFP32(int8_t* weight, float* bias, int M, int N, float alpha, int* ret_id) {
  *ret_id = Ex_Set_Linear_Param_WS8BFP32(weight,bias,M,N,alpha);
}

void ecall_Sgx_Get_Tensor_Dim_Int32(int src_id, int* dim) {
  Ex_Get_Tensor_Dim_Int32(src_id,dim);
}

void ecall_Sgx_Get_Tensor_Int32(int src_id, int* out) {
  Ex_Get_Tensor_Int32(src_id,out);
}

void ecall_Sgx_Set_Tensor_Int32(int* data, int B, int M, int N, int* ret_id) {
  *ret_id = Ex_Set_Tensor_Int32(data,B,M,N);
}

void ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int linear_param_id, int* out) {
  Ex_Get_Encrypted_Tensor_Opr1_Int32(src_id, linear_param_id,out);
}

void ecall_Sgx_Generate_Decryption_Key_Opr1_Int32(int blind_factor_id, int linear_param_id, int* ret_id) {
  *ret_id = Ex_Generate_Decryption_Key_Opr1_Int32(blind_factor_id,linear_param_id);
}

void ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int linear_param_id, int* ret_id) {
  *ret_id = Ex_Set_Decrypted_Tensor_Opr1_Int32(data,B,M,N,linear_param_id);
}

void ecall_Sgx_Get_Tensor_Dim_Int8(int src_id, int* dim) {
  Ex_Get_Tensor_Dim_Int8(src_id,dim);
}

void ecall_Sgx_Get_Tensor_Int8(int src_id, int8_t* out) {
  Ex_Get_Tensor_Int8(src_id,out);
}

void ecall_Sgx_Set_Tensor_Int8(int8_t* data, int B, int M, int N, int* ret_id) {
  *ret_id = Ex_Set_Tensor_Int8(data,B,M,N);
}

void ecall_Sgx_Get_Tensor_Dim_Float(int src_id, int* dim) {
  Ex_Get_Tensor_Dim_Float(src_id,dim);
}

void ecall_Sgx_Get_Tensor_Float(int src_id, float* out) {
  Ex_Get_Tensor_Float(src_id,out);
}

void ecall_Sgx_Set_Tensor_Float(float* data, int B, int M, int N, int* ret_id) {
  *ret_id = Ex_Set_Tensor_Float(data,B,M,N);
}

void ecall_Sgx_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id, int* ret_id) {
  *ret_id = Ex_Compute_Epilogue_WS8BS8(src_id,linear_param_id);
}

void ecall_Sgx_Compute_Epilogue_WS8BFP32(int src_id, int linear_param_id, int* ret_id) {
  *ret_id = Ex_Compute_Epilogue_WS8BFP32(src_id,linear_param_id);
}

void ecall_Sgx_Compute_Epilogue_BMM(int src_id, int bmm_param_id, int* ret_id) {
  *ret_id = Ex_Compute_Epilogue_BMM(src_id,bmm_param_id);
}

void ecall_Sgx_ReLU(int src_id, int* ret_id) {
  *ret_id = Ex_ReLU(src_id);
}

void ecall_Sgx_Softmax(int src_id, int* ret_id) {
  *ret_id = Ex_Softmax(src_id);
}

void ecall_Sgx_Quantize_Post_Softmax(int src_id, int* ret_id) {
  *ret_id = Ex_Quantize_Post_Softmax(src_id);
}

void ecall_Sgx_Cast_From_Float_To_Int8(int src_id, int* ret_id) {
  *ret_id = Ex_Cast_From_Float_To_Int8(src_id);
}

void ecall_Sgx_Cast_From_Float_To_Int32(int src_id, int* ret_id) {
  *ret_id = Ex_Cast_From_Float_To_Int32(src_id);
}

void ecall_Sgx_Cast_From_Int8_To_Int32(int src_id, int* ret_id) {
  *ret_id = Ex_Cast_From_Int8_To_Int32(src_id);
}

void ecall_Sgx_Set_Bmm_Param(float alpha, int* ret_id) {
  *ret_id = Ex_Set_Bmm_Param(alpha);
}

void ecall_Sgx_Residual_Add(int residual, int hidden_states, int* ret_id) {
  *ret_id = Ex_Residual_Add(residual,hidden_states);
}

void ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2, uint32_t* out1, uint32_t* out2, int layer_id) {
  Ex_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(src_id1, src_id2, out1, out2, layer_id);
}

void ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2, int layer_id) {
  Ex_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(src_id1, src_id2, layer_id);
}

void ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(uint32_t* data, int B, int M, int N, int layer_id, int* ret_id) {
  *ret_id = Ex_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(data, B, M, N, layer_id);
}

void ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2, uint32_t* out1, uint32_t* out2, int layer_id) {
  Ex_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(src_id1, src_id2, out1, out2, layer_id);
}

void ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2, int layer_id) {
  Ex_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(src_id1, src_id2, layer_id);
}

void ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(uint32_t* data, int B, int M, int N, int layer_id, int* ret_id) {
  *ret_id = Ex_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(data, B, M, N, layer_id);
}

void ecall_Sgx_Pre_Init() {
  Ex_Pre_Init();
}

void ecall_Sgx_Reset() {
  Ex_Reset();
}

void ecall_Sgx_CPU_Bmm(int src_id1, int src_id2, int* ret_id) {
  *ret_id = Ex_CPU_Bmm(src_id1, src_id2);
}

}

#endif