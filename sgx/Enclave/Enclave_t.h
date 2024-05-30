#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void ecall_Sgx_Set_Hidden_States(float* hidden_states, int B, int M, int N, int* ret_id);
void ecall_Sgx_Copy_Hidden_States(int src_id, int* ret_id);
void ecall_Sgx_Set_Layer_Norm_Param(float* gamma, float* beta, int N, float eps, int* ret_id);
void ecall_Sgx_Layer_Norm_Q(int src_id, int layer_norm_param_id, int* ret_id);
void ecall_Sgx_Set_Linear_Param_WS8BS8(char* weight, char* bias, int M, int N, float alpha, float beta, int* ret_id);
void ecall_Sgx_Set_Linear_Param_WS8BFP32(char* weight, float* bias, int M, int N, float alpha, int* ret_id);
void ecall_Sgx_Get_Tensor_Dim_Int32(int src_id, int* dim);
void ecall_Sgx_Get_Tensor_Int32(int src_id, int* out);
void ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(int src_id, int* out, int* ret_id);
void ecall_Sgx_Set_Tensor_Int32(int* data, int B, int M, int N, int* ret_id);
void ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(int* data, int B, int M, int N, int blind_factor_id, int linear_param_id, int* ret_id);
void ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32(int src_id1, int src_id2, int* out1, int* out2, int* ret_id);
void ecall_Sgx_Set_Decrypted_Tensor_Opr2_Int32(int* data, int B, int M, int N, int unblind_factor_id, int* ret_id);
void ecall_Sgx_Get_Tensor_Dim_Int8(int src_id, int* dim);
void ecall_Sgx_Get_Tensor_Int8(int src_id, char* out);
void ecall_Sgx_Set_Tensor_Int8(char* data, int B, int M, int N, int* ret_id);
void ecall_Sgx_Get_Tensor_Dim_Float(int src_id, int* dim);
void ecall_Sgx_Get_Tensor_Float(int src_id, float* out);
void ecall_Sgx_Set_Tensor_Float(float* data, int B, int M, int N, int* ret_id);
void ecall_Sgx_Compute_Epilogue_WS8BS8(int src_id, int linear_param_id, int* ret_id);
void ecall_Sgx_Compute_Epilogue_WS8BFP32(int src_id, int linear_param_id, int* ret_id);
void ecall_Sgx_Compute_Epilogue_BMM(int src_id, int bmm_param_id, int* ret_id);
void ecall_Sgx_ReLU(int src_id, int* ret_id);
void ecall_Sgx_Softmax(int src_id, int* ret_id);
void ecall_Sgx_Quantize_Post_Softmax(int src_id, int* ret_id);
void ecall_Sgx_Cast_From_Float_To_Int8(int src_id, int* ret_id);
void ecall_Sgx_Cast_From_Float_To_Int32(int src_id, int* ret_id);
void ecall_Sgx_Cast_From_Int8_To_Int32(int src_id, int* ret_id);
void ecall_Sgx_Set_Bmm_Param(float alpha, int* ret_id);
void ecall_Sgx_Residual_Add(int residual, int hidden_states, int* ret_id);

sgx_status_t SGX_CDECL ocall_print_string(const char* str);
sgx_status_t SGX_CDECL ocall_start_clock(void);
sgx_status_t SGX_CDECL ocall_get_time(double* retval);
sgx_status_t SGX_CDECL ocall_end_clock(const char* str);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
