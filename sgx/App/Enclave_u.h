#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_status_t etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OCALL_PRINT_STRING_DEFINED__
#define OCALL_PRINT_STRING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print_string, (const char* str));
#endif
#ifndef OCALL_START_CLOCK_DEFINED__
#define OCALL_START_CLOCK_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_start_clock, (void));
#endif
#ifndef OCALL_GET_TIME_DEFINED__
#define OCALL_GET_TIME_DEFINED__
double SGX_UBRIDGE(SGX_NOCONVENTION, ocall_get_time, (void));
#endif
#ifndef OCALL_END_CLOCK_DEFINED__
#define OCALL_END_CLOCK_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_end_clock, (const char* str));
#endif

sgx_status_t ecall_Sgx_Get_Encrypted_Tensor_QK_Int32(sgx_enclave_id_t eid, int src_id1, int src_id2, int* out1, int* out2, int* blind_factor_ids);
sgx_status_t ecall_Sgx_Generate_Decryption_Key_QK_Int32(sgx_enclave_id_t eid, int src_id1, int src_id2, int blind_factor_u_id, int blind_factor_v_id);
sgx_status_t ecall_Sgx_Set_Decrypted_Tensor_QK_Int32(sgx_enclave_id_t eid, int* data, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Get_Encrypted_Tensor_PV_Int32(sgx_enclave_id_t eid, int src_id1, int src_id2, int* out1, int* out2, int* blind_factor_ids);
sgx_status_t ecall_Sgx_Generate_Decryption_Key_PV_Int32(sgx_enclave_id_t eid, int src_id1, int src_id2, int blind_factor_u_id, int blind_factor_v_id);
sgx_status_t ecall_Sgx_Set_Decrypted_Tensor_PV_Int32(sgx_enclave_id_t eid, int* data, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(sgx_enclave_id_t eid, int src_id1, int src_id2, int* out1, int* out2, int layer_id);
sgx_status_t ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(sgx_enclave_id_t eid, int src_id1, int src_id2, int layer_id);
sgx_status_t ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(sgx_enclave_id_t eid, int* data, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(sgx_enclave_id_t eid, int src_id1, int src_id2, int* out1, int* out2, int layer_id);
sgx_status_t ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(sgx_enclave_id_t eid, int src_id1, int src_id2, int layer_id);
sgx_status_t ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(sgx_enclave_id_t eid, int* data, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Pre_Init(sgx_enclave_id_t eid);
sgx_status_t ecall_Sgx_Set_Hidden_States(sgx_enclave_id_t eid, float* hidden_states, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Copy_Hidden_States(sgx_enclave_id_t eid, int src_id, int* ret_id);
sgx_status_t ecall_Sgx_Set_Layer_Norm_Param(sgx_enclave_id_t eid, float* gamma, float* beta, int N, float eps, int* ret_id);
sgx_status_t ecall_Sgx_Layer_Norm_Q(sgx_enclave_id_t eid, int src_id, int layer_norm_param_id, int* ret_id);
sgx_status_t ecall_Sgx_Set_Linear_Param_WS8BS8(sgx_enclave_id_t eid, char* weight, char* bias, int M, int N, float alpha, float beta, int* ret_id);
sgx_status_t ecall_Sgx_Set_Linear_Param_WS8BFP32(sgx_enclave_id_t eid, char* weight, float* bias, int M, int N, float alpha, int* ret_id);
sgx_status_t ecall_Sgx_Get_Tensor_Dim_Int32(sgx_enclave_id_t eid, int src_id, int* dim);
sgx_status_t ecall_Sgx_Get_Tensor_Int32(sgx_enclave_id_t eid, int src_id, int* out);
sgx_status_t ecall_Sgx_Set_Tensor_Int32(sgx_enclave_id_t eid, int* data, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(sgx_enclave_id_t eid, int src_id, int linear_param_id, int* out);
sgx_status_t ecall_Sgx_Generate_Decryption_Key_Opr1_Int32(sgx_enclave_id_t eid, int blind_factor_id, int linear_param_id, int* ret_id);
sgx_status_t ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(sgx_enclave_id_t eid, int* data, int B, int M, int N, int linear_param_id, int* ret_id);
sgx_status_t ecall_Sgx_Get_Tensor_Dim_Int8(sgx_enclave_id_t eid, int src_id, int* dim);
sgx_status_t ecall_Sgx_Get_Tensor_Int8(sgx_enclave_id_t eid, int src_id, char* out);
sgx_status_t ecall_Sgx_Set_Tensor_Int8(sgx_enclave_id_t eid, char* data, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Get_Tensor_Dim_Float(sgx_enclave_id_t eid, int src_id, int* dim);
sgx_status_t ecall_Sgx_Get_Tensor_Float(sgx_enclave_id_t eid, int src_id, float* out);
sgx_status_t ecall_Sgx_Set_Tensor_Float(sgx_enclave_id_t eid, float* data, int B, int M, int N, int* ret_id);
sgx_status_t ecall_Sgx_Compute_Epilogue_WS8BS8(sgx_enclave_id_t eid, int src_id, int linear_param_id, int* ret_id);
sgx_status_t ecall_Sgx_Compute_Epilogue_WS8BFP32(sgx_enclave_id_t eid, int src_id, int linear_param_id, int* ret_id);
sgx_status_t ecall_Sgx_Compute_Epilogue_BMM(sgx_enclave_id_t eid, int src_id, int bmm_param_id, int* ret_id);
sgx_status_t ecall_Sgx_ReLU(sgx_enclave_id_t eid, int src_id, int* ret_id);
sgx_status_t ecall_Sgx_Softmax(sgx_enclave_id_t eid, int src_id, int* ret_id);
sgx_status_t ecall_Sgx_Quantize_Post_Softmax(sgx_enclave_id_t eid, int src_id, int* ret_id);
sgx_status_t ecall_Sgx_Cast_From_Float_To_Int8(sgx_enclave_id_t eid, int src_id, int* ret_id);
sgx_status_t ecall_Sgx_Cast_From_Float_To_Int32(sgx_enclave_id_t eid, int src_id, int* ret_id);
sgx_status_t ecall_Sgx_Cast_From_Int8_To_Int32(sgx_enclave_id_t eid, int src_id, int* ret_id);
sgx_status_t ecall_Sgx_Set_Bmm_Param(sgx_enclave_id_t eid, float alpha, int* ret_id);
sgx_status_t ecall_Sgx_Residual_Add(sgx_enclave_id_t eid, int residual, int hidden_states, int* ret_id);
sgx_status_t ecall_Sgx_CPU_Bmm(sgx_enclave_id_t eid, int src_id1, int src_id2, int* ret_id);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
