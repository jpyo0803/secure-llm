#ifndef SECURE_LLM_SGX_ENCLAVE_SGX_LAYER_STRUCT_H
#define SECURE_LLM_SGX_ENCLAVE_SGX_LAYER_STRUCT_H


#include <assert.h>
#include <math.h>
// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "cpp/aes_stream.h"

#include "cpp/dynamic_glob_data.h"
#include "cpp/static_glob_data.h"
#include "cpp/tensor.h"


extern "C" {

int Sgx_Set_Hidden_States(float* hidden_states, int B, int M, int N);

int Sgx_Copy_Hidden_States(int src_id);

int Sgx_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps);

int Sgx_Layer_Norm_Q(int src_id, int layer_norm_param_id);

int Sgx_Set_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N,
                                float alpha, float beta);
int Sgx_Set_Linear_Param_WS8BFP32(char* weight, float* bias, int M, int N,
                                  float alpha);

void Sgx_Get_Tensor_Dim_Int32(int src_id, int* dim);
void Sgx_Get_Tensor_Int32(int src_id, int* out);
int Sgx_Set_Tensor_Int32(int* data, int B, int M, int N);

void Sgx_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int linear_param_id, int* out);
int Sgx_Generate_Decryption_Key_Opr1_Int32(int blind_factor_id,
                                          int linear_param_id);
int Sgx_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int linear_param_id);

void Sgx_Get_Tensor_Dim_Int8(int src_id, int* dim);
void Sgx_Get_Tensor_Int8(int src_id, char* out);
int Sgx_Set_Tensor_Int8(char* data, int B, int M, int N);

void Sgx_Get_Tensor_Dim_Float(int src_id, int* dim);
void Sgx_Get_Tensor_Float(int src_id, float* out);
int Sgx_Set_Tensor_Float(float* data, int B, int M, int N);

int Sgx_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id);
int Sgx_Compute_Epilogue_WS8BFP32(int src_id, int linear_param_id);
int Sgx_Compute_Epilogue_BMM(int src_id, int bmm_param_id);

int Sgx_ReLU(int src_id);
int Sgx_Softmax(int src_id);
int Sgx_Quantize_Post_Softmax(int src_id);

int Sgx_Cast_From_Float_To_Int8(int src_id);
int Sgx_Cast_From_Float_To_Int32(int src_id);
int Sgx_Cast_From_Int8_To_Int32(int src_id);

int Sgx_Set_Bmm_Param(float alpha);

int Sgx_Residual_Add(int residual, int hidden_states);

// Handle blind / unblind factors inside
void Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                     int* out1, int* out2, int layer_id);
int Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(int src_id1, int src_id2, int layer_id);
int Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(int* data, int B, int M, int N, int decryption_key_id);

void Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2,
                                                     int* out1, int* out2, int layer_id);
int Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(int src_id1, int src_id2, int layer_id);
int Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(int* data, int B, int M, int N, int decryption_key_id);

void Sgx_Pre_Init();

void Sgx_Get_Encrypted_Tensor_QK_Int32(int src_id1, int src_id2, int* out1,
                                      int* out2, int* blind_factor_ids);
int Sgx_Generate_Decryption_Key_QK_Int32(int src_id1, int src_id2,
                                        int blind_factor_u_id,
                                        int blind_factor_v_id);
int Sgx_Set_Decrypted_Tensor_QK_Int32(int* data, int B, int M, int N,
                                      int decryption_key_id);

void Sgx_Get_Encrypted_Tensor_PV_Int32(int src_id1, int src_id2, int* out1,
                                      int* out2, int* blind_factor_ids);
int Sgx_Generate_Decryption_Key_PV_Int32(int src_id1, int src_id2,
                                        int blind_factor_u_id,
                                        int blind_factor_v_id);
int Sgx_Set_Decrypted_Tensor_PV_Int32(int* data, int B, int M, int N, int decryption_key_id);


}

#endif 