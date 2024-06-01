#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_Sgx_Set_Hidden_States_t {
	float* ms_hidden_states;
	int ms_B;
	int ms_M;
	int ms_N;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Hidden_States_t;

typedef struct ms_ecall_Sgx_Copy_Hidden_States_t {
	int ms_src_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Copy_Hidden_States_t;

typedef struct ms_ecall_Sgx_Set_Layer_Norm_Param_t {
	float* ms_gamma;
	float* ms_beta;
	int ms_N;
	float ms_eps;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Layer_Norm_Param_t;

typedef struct ms_ecall_Sgx_Layer_Norm_Q_t {
	int ms_src_id;
	int ms_layer_norm_param_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Layer_Norm_Q_t;

typedef struct ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t {
	char* ms_weight;
	char* ms_bias;
	int ms_M;
	int ms_N;
	float ms_alpha;
	float ms_beta;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t;

typedef struct ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t {
	char* ms_weight;
	float* ms_bias;
	int ms_M;
	int ms_N;
	float ms_alpha;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t;

typedef struct ms_ecall_Sgx_Get_Tensor_Dim_Int32_t {
	int ms_src_id;
	int* ms_dim;
} ms_ecall_Sgx_Get_Tensor_Dim_Int32_t;

typedef struct ms_ecall_Sgx_Get_Tensor_Int32_t {
	int ms_src_id;
	int* ms_out;
} ms_ecall_Sgx_Get_Tensor_Int32_t;

typedef struct ms_ecall_Sgx_Set_Tensor_Int32_t {
	int* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Tensor_Int32_t;

typedef struct ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t {
	int ms_src_id;
	int ms_linear_param_id;
	int* ms_out;
} ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t;

typedef struct ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t {
	int ms_blind_factor_id;
	int ms_linear_param_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t;

typedef struct ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t {
	int* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int ms_linear_param_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t;

typedef struct ms_ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32_t {
	int ms_src_id1;
	int ms_src_id2;
	int* ms_out1;
	int* ms_out2;
	int* ms_blind_factor_ids;
} ms_ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32_t;

typedef struct ms_ecall_Sgx_Generate_Decryption_Key_Opr2_Int32_t {
	int ms_src_id1;
	int ms_src_id2;
	int ms_blind_factor_u_id;
	int ms_blind_factor_v_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Generate_Decryption_Key_Opr2_Int32_t;

typedef struct ms_ecall_Sgx_Set_Decrypted_Tensor_Opr2_Int32_t {
	int* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int ms_decryption_key_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Decrypted_Tensor_Opr2_Int32_t;

typedef struct ms_ecall_Sgx_Get_Tensor_Dim_Int8_t {
	int ms_src_id;
	int* ms_dim;
} ms_ecall_Sgx_Get_Tensor_Dim_Int8_t;

typedef struct ms_ecall_Sgx_Get_Tensor_Int8_t {
	int ms_src_id;
	char* ms_out;
} ms_ecall_Sgx_Get_Tensor_Int8_t;

typedef struct ms_ecall_Sgx_Set_Tensor_Int8_t {
	char* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Tensor_Int8_t;

typedef struct ms_ecall_Sgx_Get_Tensor_Dim_Float_t {
	int ms_src_id;
	int* ms_dim;
} ms_ecall_Sgx_Get_Tensor_Dim_Float_t;

typedef struct ms_ecall_Sgx_Get_Tensor_Float_t {
	int ms_src_id;
	float* ms_out;
} ms_ecall_Sgx_Get_Tensor_Float_t;

typedef struct ms_ecall_Sgx_Set_Tensor_Float_t {
	float* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Tensor_Float_t;

typedef struct ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t {
	int ms_src_id;
	int ms_linear_param_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t;

typedef struct ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t {
	int ms_src_id;
	int ms_linear_param_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t;

typedef struct ms_ecall_Sgx_Compute_Epilogue_BMM_t {
	int ms_src_id;
	int ms_bmm_param_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Compute_Epilogue_BMM_t;

typedef struct ms_ecall_Sgx_ReLU_t {
	int ms_src_id;
	int* ms_ret_id;
} ms_ecall_Sgx_ReLU_t;

typedef struct ms_ecall_Sgx_Softmax_t {
	int ms_src_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Softmax_t;

typedef struct ms_ecall_Sgx_Quantize_Post_Softmax_t {
	int ms_src_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Quantize_Post_Softmax_t;

typedef struct ms_ecall_Sgx_Cast_From_Float_To_Int8_t {
	int ms_src_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Cast_From_Float_To_Int8_t;

typedef struct ms_ecall_Sgx_Cast_From_Float_To_Int32_t {
	int ms_src_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Cast_From_Float_To_Int32_t;

typedef struct ms_ecall_Sgx_Cast_From_Int8_To_Int32_t {
	int ms_src_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Cast_From_Int8_To_Int32_t;

typedef struct ms_ecall_Sgx_Set_Bmm_Param_t {
	float ms_alpha;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Bmm_Param_t;

typedef struct ms_ecall_Sgx_Residual_Add_t {
	int ms_residual;
	int ms_hidden_states;
	int* ms_ret_id;
} ms_ecall_Sgx_Residual_Add_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_ocall_get_time_t {
	double ms_retval;
} ms_ocall_get_time_t;

typedef struct ms_ocall_end_clock_t {
	const char* ms_str;
} ms_ocall_end_clock_t;

static sgx_status_t SGX_CDECL Enclave_ocall_print_string(void* pms)
{
	ms_ocall_print_string_t* ms = SGX_CAST(ms_ocall_print_string_t*, pms);
	ocall_print_string(ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_start_clock(void* pms)
{
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	ocall_start_clock();
	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_get_time(void* pms)
{
	ms_ocall_get_time_t* ms = SGX_CAST(ms_ocall_get_time_t*, pms);
	ms->ms_retval = ocall_get_time();

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_end_clock(void* pms)
{
	ms_ocall_end_clock_t* ms = SGX_CAST(ms_ocall_end_clock_t*, pms);
	ocall_end_clock(ms->ms_str);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[4];
} ocall_table_Enclave = {
	4,
	{
		(void*)Enclave_ocall_print_string,
		(void*)Enclave_ocall_start_clock,
		(void*)Enclave_ocall_get_time,
		(void*)Enclave_ocall_end_clock,
	}
};
sgx_status_t ecall_Sgx_Set_Hidden_States(sgx_enclave_id_t eid, float* hidden_states, int B, int M, int N, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Hidden_States_t ms;
	ms.ms_hidden_states = hidden_states;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Copy_Hidden_States(sgx_enclave_id_t eid, int src_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Copy_Hidden_States_t ms;
	ms.ms_src_id = src_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Layer_Norm_Param(sgx_enclave_id_t eid, float* gamma, float* beta, int N, float eps, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Layer_Norm_Param_t ms;
	ms.ms_gamma = gamma;
	ms.ms_beta = beta;
	ms.ms_N = N;
	ms.ms_eps = eps;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Layer_Norm_Q(sgx_enclave_id_t eid, int src_id, int layer_norm_param_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Layer_Norm_Q_t ms;
	ms.ms_src_id = src_id;
	ms.ms_layer_norm_param_id = layer_norm_param_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Linear_Param_WS8BS8(sgx_enclave_id_t eid, char* weight, char* bias, int M, int N, float alpha, float beta, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t ms;
	ms.ms_weight = weight;
	ms.ms_bias = bias;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_alpha = alpha;
	ms.ms_beta = beta;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 4, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Linear_Param_WS8BFP32(sgx_enclave_id_t eid, char* weight, float* bias, int M, int N, float alpha, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t ms;
	ms.ms_weight = weight;
	ms.ms_bias = bias;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_alpha = alpha;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 5, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Tensor_Dim_Int32(sgx_enclave_id_t eid, int src_id, int* dim)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Tensor_Dim_Int32_t ms;
	ms.ms_src_id = src_id;
	ms.ms_dim = dim;
	status = sgx_ecall(eid, 6, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Tensor_Int32(sgx_enclave_id_t eid, int src_id, int* out)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Tensor_Int32_t ms;
	ms.ms_src_id = src_id;
	ms.ms_out = out;
	status = sgx_ecall(eid, 7, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Tensor_Int32(sgx_enclave_id_t eid, int* data, int B, int M, int N, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Tensor_Int32_t ms;
	ms.ms_data = data;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 8, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(sgx_enclave_id_t eid, int src_id, int linear_param_id, int* out)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t ms;
	ms.ms_src_id = src_id;
	ms.ms_linear_param_id = linear_param_id;
	ms.ms_out = out;
	status = sgx_ecall(eid, 9, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Generate_Decryption_Key_Opr1_Int32(sgx_enclave_id_t eid, int blind_factor_id, int linear_param_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t ms;
	ms.ms_blind_factor_id = blind_factor_id;
	ms.ms_linear_param_id = linear_param_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 10, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(sgx_enclave_id_t eid, int* data, int B, int M, int N, int linear_param_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t ms;
	ms.ms_data = data;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_linear_param_id = linear_param_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 11, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32(sgx_enclave_id_t eid, int src_id1, int src_id2, int* out1, int* out2, int* blind_factor_ids)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32_t ms;
	ms.ms_src_id1 = src_id1;
	ms.ms_src_id2 = src_id2;
	ms.ms_out1 = out1;
	ms.ms_out2 = out2;
	ms.ms_blind_factor_ids = blind_factor_ids;
	status = sgx_ecall(eid, 12, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Generate_Decryption_Key_Opr2_Int32(sgx_enclave_id_t eid, int src_id1, int src_id2, int blind_factor_u_id, int blind_factor_v_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Generate_Decryption_Key_Opr2_Int32_t ms;
	ms.ms_src_id1 = src_id1;
	ms.ms_src_id2 = src_id2;
	ms.ms_blind_factor_u_id = blind_factor_u_id;
	ms.ms_blind_factor_v_id = blind_factor_v_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 13, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Decrypted_Tensor_Opr2_Int32(sgx_enclave_id_t eid, int* data, int B, int M, int N, int decryption_key_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Decrypted_Tensor_Opr2_Int32_t ms;
	ms.ms_data = data;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_decryption_key_id = decryption_key_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 14, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Tensor_Dim_Int8(sgx_enclave_id_t eid, int src_id, int* dim)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Tensor_Dim_Int8_t ms;
	ms.ms_src_id = src_id;
	ms.ms_dim = dim;
	status = sgx_ecall(eid, 15, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Tensor_Int8(sgx_enclave_id_t eid, int src_id, char* out)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Tensor_Int8_t ms;
	ms.ms_src_id = src_id;
	ms.ms_out = out;
	status = sgx_ecall(eid, 16, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Tensor_Int8(sgx_enclave_id_t eid, char* data, int B, int M, int N, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Tensor_Int8_t ms;
	ms.ms_data = data;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 17, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Tensor_Dim_Float(sgx_enclave_id_t eid, int src_id, int* dim)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Tensor_Dim_Float_t ms;
	ms.ms_src_id = src_id;
	ms.ms_dim = dim;
	status = sgx_ecall(eid, 18, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Get_Tensor_Float(sgx_enclave_id_t eid, int src_id, float* out)
{
	sgx_status_t status;
	ms_ecall_Sgx_Get_Tensor_Float_t ms;
	ms.ms_src_id = src_id;
	ms.ms_out = out;
	status = sgx_ecall(eid, 19, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Tensor_Float(sgx_enclave_id_t eid, float* data, int B, int M, int N, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Tensor_Float_t ms;
	ms.ms_data = data;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 20, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Compute_Epilogue_WS8BS8(sgx_enclave_id_t eid, int src_id, int linear_param_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t ms;
	ms.ms_src_id = src_id;
	ms.ms_linear_param_id = linear_param_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 21, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Compute_Epilogue_WS8BFP32(sgx_enclave_id_t eid, int src_id, int linear_param_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t ms;
	ms.ms_src_id = src_id;
	ms.ms_linear_param_id = linear_param_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 22, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Compute_Epilogue_BMM(sgx_enclave_id_t eid, int src_id, int bmm_param_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Compute_Epilogue_BMM_t ms;
	ms.ms_src_id = src_id;
	ms.ms_bmm_param_id = bmm_param_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 23, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_ReLU(sgx_enclave_id_t eid, int src_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_ReLU_t ms;
	ms.ms_src_id = src_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 24, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Softmax(sgx_enclave_id_t eid, int src_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Softmax_t ms;
	ms.ms_src_id = src_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 25, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Quantize_Post_Softmax(sgx_enclave_id_t eid, int src_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Quantize_Post_Softmax_t ms;
	ms.ms_src_id = src_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 26, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Cast_From_Float_To_Int8(sgx_enclave_id_t eid, int src_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Cast_From_Float_To_Int8_t ms;
	ms.ms_src_id = src_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 27, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Cast_From_Float_To_Int32(sgx_enclave_id_t eid, int src_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Cast_From_Float_To_Int32_t ms;
	ms.ms_src_id = src_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 28, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Cast_From_Int8_To_Int32(sgx_enclave_id_t eid, int src_id, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Cast_From_Int8_To_Int32_t ms;
	ms.ms_src_id = src_id;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 29, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Set_Bmm_Param(sgx_enclave_id_t eid, float alpha, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Set_Bmm_Param_t ms;
	ms.ms_alpha = alpha;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 30, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Sgx_Residual_Add(sgx_enclave_id_t eid, int residual, int hidden_states, int* ret_id)
{
	sgx_status_t status;
	ms_ecall_Sgx_Residual_Add_t ms;
	ms.ms_residual = residual;
	ms.ms_hidden_states = hidden_states;
	ms.ms_ret_id = ret_id;
	status = sgx_ecall(eid, 31, &ocall_table_Enclave, &ms);
	return status;
}

