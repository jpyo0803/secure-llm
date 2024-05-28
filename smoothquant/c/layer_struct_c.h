#ifndef SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H
#define SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H

#include "tensor.h"

#include "static_glob_data.h"
#include "dynamic_glob_data.h"

#include <stdlib.h>

int Ex_Set_Hidden_States(float* hidden_states, int B, int M, int N);

int Ex_Copy_Hidden_States(int src_id);

int Ex_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps);

int Ex_Layer_Norm_Q(int src_id, int layer_norm_param_id);

int Ex_Set_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N, float alpha, float beta);
int Ex_Set_Linear_Param_WS8BFP32(char* weight, float* bias, int M, int N, float alpha);

void Ex_Get_Tensor_Dim_Int32(int src_id, int* B, int* M, int *N);
void Ex_Get_Tensor_Int32(int src_id, int* out);
int Ex_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int* out);
int Ex_Set_Tensor_Int32(int* data, int B, int M, int N);
int Ex_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int blind_factor_id, int linear_param_id);

void Ex_Get_Tensor_Dim_Int8(int src_id, int* B, int* M, int *N);
void Ex_Get_Tensor_Int8(int src_id, char* out);
int Ex_Set_Tensor_Int8(char* data, int B, int M, int N);

int Ex_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id);

int Ex_ReLU(int src_id);

int Ex_Cast_From_Float_To_Int8(int src_id);
int Ex_Cast_From_Float_To_Int32(int src_id);

int Ex_Set_Bmm_Param(float alpha);

#endif