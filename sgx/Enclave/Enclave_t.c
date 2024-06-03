#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


typedef struct ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t {
	int ms_src_id1;
	int ms_src_id2;
	int* ms_out1;
	int* ms_out2;
	int* ms_blind_factor_ids;
} ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t;

typedef struct ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t {
	int ms_src_id1;
	int ms_src_id2;
	int ms_blind_factor_u_id;
	int ms_blind_factor_v_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t;

typedef struct ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t {
	int* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int ms_decryption_key_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t;

typedef struct ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t {
	int ms_src_id1;
	int ms_src_id2;
	int* ms_out1;
	int* ms_out2;
	int* ms_blind_factor_ids;
} ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t;

typedef struct ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t {
	int ms_src_id1;
	int ms_src_id2;
	int ms_blind_factor_u_id;
	int ms_blind_factor_v_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t;

typedef struct ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t {
	int* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int ms_decryption_key_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t;

typedef struct ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t {
	int ms_src_id1;
	int ms_src_id2;
	int* ms_out1;
	int* ms_out2;
	int ms_layer_id;
} ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t;

typedef struct ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t {
	int ms_src_id1;
	int ms_src_id2;
	int ms_layer_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t;

typedef struct ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t {
	int* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int ms_decryption_key_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t;

typedef struct ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t {
	int ms_src_id1;
	int ms_src_id2;
	int* ms_out1;
	int* ms_out2;
	int ms_layer_id;
} ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t;

typedef struct ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t {
	int ms_src_id1;
	int ms_src_id2;
	int ms_layer_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t;

typedef struct ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t {
	int* ms_data;
	int ms_B;
	int ms_M;
	int ms_N;
	int ms_decryption_key_id;
	int* ms_ret_id;
} ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t;

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

typedef struct ms_ecall_Sgx_CPU_Bmm_t {
	int ms_src_id1;
	int ms_src_id2;
	int* ms_ret_id;
} ms_ecall_Sgx_CPU_Bmm_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_ocall_get_time_t {
	double ms_retval;
} ms_ocall_get_time_t;

typedef struct ms_ocall_end_clock_t {
	const char* ms_str;
} ms_ocall_end_clock_t;

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t*, pms);
	ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t), ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_out1 = __in_ms.ms_out1;
	int* _tmp_out2 = __in_ms.ms_out2;
	int* _tmp_blind_factor_ids = __in_ms.ms_blind_factor_ids;


	ecall_Sgx_Get_Encrypted_Tensor_QK_Int32(__in_ms.ms_src_id1, __in_ms.ms_src_id2, _tmp_out1, _tmp_out2, _tmp_blind_factor_ids);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Generate_Decryption_Key_QK_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t*, pms);
	ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t), ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Generate_Decryption_Key_QK_Int32(__in_ms.ms_src_id1, __in_ms.ms_src_id2, __in_ms.ms_blind_factor_u_id, __in_ms.ms_blind_factor_v_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t*, pms);
	ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t), ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Decrypted_Tensor_QK_Int32(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, __in_ms.ms_decryption_key_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t*, pms);
	ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t), ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_out1 = __in_ms.ms_out1;
	int* _tmp_out2 = __in_ms.ms_out2;
	int* _tmp_blind_factor_ids = __in_ms.ms_blind_factor_ids;


	ecall_Sgx_Get_Encrypted_Tensor_PV_Int32(__in_ms.ms_src_id1, __in_ms.ms_src_id2, _tmp_out1, _tmp_out2, _tmp_blind_factor_ids);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Generate_Decryption_Key_PV_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t*, pms);
	ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t), ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Generate_Decryption_Key_PV_Int32(__in_ms.ms_src_id1, __in_ms.ms_src_id2, __in_ms.ms_blind_factor_u_id, __in_ms.ms_blind_factor_v_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t*, pms);
	ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t), ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Decrypted_Tensor_PV_Int32(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, __in_ms.ms_decryption_key_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t*, pms);
	ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t), ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_out1 = __in_ms.ms_out1;
	int* _tmp_out2 = __in_ms.ms_out2;


	ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(__in_ms.ms_src_id1, __in_ms.ms_src_id2, _tmp_out1, _tmp_out2, __in_ms.ms_layer_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t* ms = SGX_CAST(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t*, pms);
	ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t), ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(__in_ms.ms_src_id1, __in_ms.ms_src_id2, __in_ms.ms_layer_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t*, pms);
	ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t), ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, __in_ms.ms_decryption_key_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t*, pms);
	ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t), ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_out1 = __in_ms.ms_out1;
	int* _tmp_out2 = __in_ms.ms_out2;


	ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(__in_ms.ms_src_id1, __in_ms.ms_src_id2, _tmp_out1, _tmp_out2, __in_ms.ms_layer_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t* ms = SGX_CAST(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t*, pms);
	ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t), ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(__in_ms.ms_src_id1, __in_ms.ms_src_id2, __in_ms.ms_layer_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t*, pms);
	ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t), ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, __in_ms.ms_decryption_key_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Pre_Init(void* pms)
{
	sgx_status_t status = SGX_SUCCESS;
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	ecall_Sgx_Pre_Init();
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Hidden_States(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Hidden_States_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Hidden_States_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Hidden_States_t*, pms);
	ms_ecall_Sgx_Set_Hidden_States_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Hidden_States_t), ms, sizeof(ms_ecall_Sgx_Set_Hidden_States_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_hidden_states = __in_ms.ms_hidden_states;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Hidden_States(_tmp_hidden_states, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Copy_Hidden_States(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Copy_Hidden_States_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Copy_Hidden_States_t* ms = SGX_CAST(ms_ecall_Sgx_Copy_Hidden_States_t*, pms);
	ms_ecall_Sgx_Copy_Hidden_States_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Copy_Hidden_States_t), ms, sizeof(ms_ecall_Sgx_Copy_Hidden_States_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Copy_Hidden_States(__in_ms.ms_src_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Layer_Norm_Param(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Layer_Norm_Param_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Layer_Norm_Param_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Layer_Norm_Param_t*, pms);
	ms_ecall_Sgx_Set_Layer_Norm_Param_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Layer_Norm_Param_t), ms, sizeof(ms_ecall_Sgx_Set_Layer_Norm_Param_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_gamma = __in_ms.ms_gamma;
	float* _tmp_beta = __in_ms.ms_beta;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Layer_Norm_Param(_tmp_gamma, _tmp_beta, __in_ms.ms_N, __in_ms.ms_eps, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Layer_Norm_Q(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Layer_Norm_Q_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Layer_Norm_Q_t* ms = SGX_CAST(ms_ecall_Sgx_Layer_Norm_Q_t*, pms);
	ms_ecall_Sgx_Layer_Norm_Q_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Layer_Norm_Q_t), ms, sizeof(ms_ecall_Sgx_Layer_Norm_Q_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Layer_Norm_Q(__in_ms.ms_src_id, __in_ms.ms_layer_norm_param_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Linear_Param_WS8BS8(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t*, pms);
	ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t), ms, sizeof(ms_ecall_Sgx_Set_Linear_Param_WS8BS8_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_weight = __in_ms.ms_weight;
	char* _tmp_bias = __in_ms.ms_bias;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Linear_Param_WS8BS8(_tmp_weight, _tmp_bias, __in_ms.ms_M, __in_ms.ms_N, __in_ms.ms_alpha, __in_ms.ms_beta, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Linear_Param_WS8BFP32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t*, pms);
	ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t), ms, sizeof(ms_ecall_Sgx_Set_Linear_Param_WS8BFP32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_weight = __in_ms.ms_weight;
	float* _tmp_bias = __in_ms.ms_bias;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Linear_Param_WS8BFP32(_tmp_weight, _tmp_bias, __in_ms.ms_M, __in_ms.ms_N, __in_ms.ms_alpha, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Tensor_Dim_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Tensor_Dim_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Tensor_Dim_Int32_t*, pms);
	ms_ecall_Sgx_Get_Tensor_Dim_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Int32_t), ms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_dim = __in_ms.ms_dim;


	ecall_Sgx_Get_Tensor_Dim_Int32(__in_ms.ms_src_id, _tmp_dim);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Tensor_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Tensor_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Tensor_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Tensor_Int32_t*, pms);
	ms_ecall_Sgx_Get_Tensor_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Tensor_Int32_t), ms, sizeof(ms_ecall_Sgx_Get_Tensor_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_out = __in_ms.ms_out;


	ecall_Sgx_Get_Tensor_Int32(__in_ms.ms_src_id, _tmp_out);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Tensor_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Tensor_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Tensor_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Tensor_Int32_t*, pms);
	ms_ecall_Sgx_Set_Tensor_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Tensor_Int32_t), ms, sizeof(ms_ecall_Sgx_Set_Tensor_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Tensor_Int32(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t*, pms);
	ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t), ms, sizeof(ms_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_out = __in_ms.ms_out;


	ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(__in_ms.ms_src_id, __in_ms.ms_linear_param_id, _tmp_out);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t*, pms);
	ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t), ms, sizeof(ms_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Generate_Decryption_Key_Opr1_Int32(__in_ms.ms_blind_factor_id, __in_ms.ms_linear_param_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t*, pms);
	ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t), ms, sizeof(ms_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, __in_ms.ms_linear_param_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Tensor_Dim_Int8(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Int8_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Tensor_Dim_Int8_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Tensor_Dim_Int8_t*, pms);
	ms_ecall_Sgx_Get_Tensor_Dim_Int8_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Int8_t), ms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Int8_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_dim = __in_ms.ms_dim;


	ecall_Sgx_Get_Tensor_Dim_Int8(__in_ms.ms_src_id, _tmp_dim);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Tensor_Int8(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Tensor_Int8_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Tensor_Int8_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Tensor_Int8_t*, pms);
	ms_ecall_Sgx_Get_Tensor_Int8_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Tensor_Int8_t), ms, sizeof(ms_ecall_Sgx_Get_Tensor_Int8_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_out = __in_ms.ms_out;


	ecall_Sgx_Get_Tensor_Int8(__in_ms.ms_src_id, _tmp_out);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Tensor_Int8(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Tensor_Int8_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Tensor_Int8_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Tensor_Int8_t*, pms);
	ms_ecall_Sgx_Set_Tensor_Int8_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Tensor_Int8_t), ms, sizeof(ms_ecall_Sgx_Set_Tensor_Int8_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Tensor_Int8(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Tensor_Dim_Float(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Float_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Tensor_Dim_Float_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Tensor_Dim_Float_t*, pms);
	ms_ecall_Sgx_Get_Tensor_Dim_Float_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Float_t), ms, sizeof(ms_ecall_Sgx_Get_Tensor_Dim_Float_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_dim = __in_ms.ms_dim;


	ecall_Sgx_Get_Tensor_Dim_Float(__in_ms.ms_src_id, _tmp_dim);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Get_Tensor_Float(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Get_Tensor_Float_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Get_Tensor_Float_t* ms = SGX_CAST(ms_ecall_Sgx_Get_Tensor_Float_t*, pms);
	ms_ecall_Sgx_Get_Tensor_Float_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Get_Tensor_Float_t), ms, sizeof(ms_ecall_Sgx_Get_Tensor_Float_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_out = __in_ms.ms_out;


	ecall_Sgx_Get_Tensor_Float(__in_ms.ms_src_id, _tmp_out);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Tensor_Float(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Tensor_Float_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Tensor_Float_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Tensor_Float_t*, pms);
	ms_ecall_Sgx_Set_Tensor_Float_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Tensor_Float_t), ms, sizeof(ms_ecall_Sgx_Set_Tensor_Float_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_data = __in_ms.ms_data;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Tensor_Float(_tmp_data, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Compute_Epilogue_WS8BS8(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t* ms = SGX_CAST(ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t*, pms);
	ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t), ms, sizeof(ms_ecall_Sgx_Compute_Epilogue_WS8BS8_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Compute_Epilogue_WS8BS8(__in_ms.ms_src_id, __in_ms.ms_linear_param_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Compute_Epilogue_WS8BFP32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t* ms = SGX_CAST(ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t*, pms);
	ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t), ms, sizeof(ms_ecall_Sgx_Compute_Epilogue_WS8BFP32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Compute_Epilogue_WS8BFP32(__in_ms.ms_src_id, __in_ms.ms_linear_param_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Compute_Epilogue_BMM(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Compute_Epilogue_BMM_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Compute_Epilogue_BMM_t* ms = SGX_CAST(ms_ecall_Sgx_Compute_Epilogue_BMM_t*, pms);
	ms_ecall_Sgx_Compute_Epilogue_BMM_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Compute_Epilogue_BMM_t), ms, sizeof(ms_ecall_Sgx_Compute_Epilogue_BMM_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Compute_Epilogue_BMM(__in_ms.ms_src_id, __in_ms.ms_bmm_param_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_ReLU(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_ReLU_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_ReLU_t* ms = SGX_CAST(ms_ecall_Sgx_ReLU_t*, pms);
	ms_ecall_Sgx_ReLU_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_ReLU_t), ms, sizeof(ms_ecall_Sgx_ReLU_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_ReLU(__in_ms.ms_src_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Softmax(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Softmax_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Softmax_t* ms = SGX_CAST(ms_ecall_Sgx_Softmax_t*, pms);
	ms_ecall_Sgx_Softmax_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Softmax_t), ms, sizeof(ms_ecall_Sgx_Softmax_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Softmax(__in_ms.ms_src_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Quantize_Post_Softmax(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Quantize_Post_Softmax_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Quantize_Post_Softmax_t* ms = SGX_CAST(ms_ecall_Sgx_Quantize_Post_Softmax_t*, pms);
	ms_ecall_Sgx_Quantize_Post_Softmax_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Quantize_Post_Softmax_t), ms, sizeof(ms_ecall_Sgx_Quantize_Post_Softmax_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Quantize_Post_Softmax(__in_ms.ms_src_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Cast_From_Float_To_Int8(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Cast_From_Float_To_Int8_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Cast_From_Float_To_Int8_t* ms = SGX_CAST(ms_ecall_Sgx_Cast_From_Float_To_Int8_t*, pms);
	ms_ecall_Sgx_Cast_From_Float_To_Int8_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Cast_From_Float_To_Int8_t), ms, sizeof(ms_ecall_Sgx_Cast_From_Float_To_Int8_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Cast_From_Float_To_Int8(__in_ms.ms_src_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Cast_From_Float_To_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Cast_From_Float_To_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Cast_From_Float_To_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Cast_From_Float_To_Int32_t*, pms);
	ms_ecall_Sgx_Cast_From_Float_To_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Cast_From_Float_To_Int32_t), ms, sizeof(ms_ecall_Sgx_Cast_From_Float_To_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Cast_From_Float_To_Int32(__in_ms.ms_src_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Cast_From_Int8_To_Int32(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Cast_From_Int8_To_Int32_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Cast_From_Int8_To_Int32_t* ms = SGX_CAST(ms_ecall_Sgx_Cast_From_Int8_To_Int32_t*, pms);
	ms_ecall_Sgx_Cast_From_Int8_To_Int32_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Cast_From_Int8_To_Int32_t), ms, sizeof(ms_ecall_Sgx_Cast_From_Int8_To_Int32_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Cast_From_Int8_To_Int32(__in_ms.ms_src_id, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Set_Bmm_Param(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Set_Bmm_Param_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Set_Bmm_Param_t* ms = SGX_CAST(ms_ecall_Sgx_Set_Bmm_Param_t*, pms);
	ms_ecall_Sgx_Set_Bmm_Param_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Set_Bmm_Param_t), ms, sizeof(ms_ecall_Sgx_Set_Bmm_Param_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Set_Bmm_Param(__in_ms.ms_alpha, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_Residual_Add(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_Residual_Add_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_Residual_Add_t* ms = SGX_CAST(ms_ecall_Sgx_Residual_Add_t*, pms);
	ms_ecall_Sgx_Residual_Add_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_Residual_Add_t), ms, sizeof(ms_ecall_Sgx_Residual_Add_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_Residual_Add(__in_ms.ms_residual, __in_ms.ms_hidden_states, _tmp_ret_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Sgx_CPU_Bmm(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Sgx_CPU_Bmm_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Sgx_CPU_Bmm_t* ms = SGX_CAST(ms_ecall_Sgx_CPU_Bmm_t*, pms);
	ms_ecall_Sgx_CPU_Bmm_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Sgx_CPU_Bmm_t), ms, sizeof(ms_ecall_Sgx_CPU_Bmm_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	int* _tmp_ret_id = __in_ms.ms_ret_id;


	ecall_Sgx_CPU_Bmm(__in_ms.ms_src_id1, __in_ms.ms_src_id2, _tmp_ret_id);


	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[43];
} g_ecall_table = {
	43,
	{
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Generate_Decryption_Key_QK_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Generate_Decryption_Key_PV_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Pre_Init, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Hidden_States, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Copy_Hidden_States, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Layer_Norm_Param, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Layer_Norm_Q, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Linear_Param_WS8BS8, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Linear_Param_WS8BFP32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Tensor_Dim_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Tensor_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Tensor_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Generate_Decryption_Key_Opr1_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Tensor_Dim_Int8, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Tensor_Int8, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Tensor_Int8, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Tensor_Dim_Float, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Get_Tensor_Float, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Tensor_Float, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Compute_Epilogue_WS8BS8, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Compute_Epilogue_WS8BFP32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Compute_Epilogue_BMM, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_ReLU, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Softmax, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Quantize_Post_Softmax, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Cast_From_Float_To_Int8, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Cast_From_Float_To_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Cast_From_Int8_To_Int32, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Set_Bmm_Param, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_Residual_Add, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Sgx_CPU_Bmm, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[4][43];
} g_dyn_entry_table = {
	4,
	{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print_string(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_string_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_string_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_string_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_string_t));
	ocalloc_size -= sizeof(ms_ocall_print_string_t);

	if (str != NULL) {
		if (memcpy_verw_s(&ms->ms_str, sizeof(const char*), &__tmp, sizeof(const char*))) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_verw_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}

	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_start_clock(void)
{
	sgx_status_t status = SGX_SUCCESS;
	status = sgx_ocall(1, NULL);

	return status;
}
sgx_status_t SGX_CDECL ocall_get_time(double* retval)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_get_time_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_get_time_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_get_time_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_get_time_t));
	ocalloc_size -= sizeof(ms_ocall_get_time_t);

	status = sgx_ocall(2, ms);

	if (status == SGX_SUCCESS) {
		if (retval) {
			if (memcpy_s((void*)retval, sizeof(*retval), &ms->ms_retval, sizeof(ms->ms_retval))) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_end_clock(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_end_clock_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_end_clock_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_end_clock_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_end_clock_t));
	ocalloc_size -= sizeof(ms_ocall_end_clock_t);

	if (str != NULL) {
		if (memcpy_verw_s(&ms->ms_str, sizeof(const char*), &__tmp, sizeof(const char*))) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_verw_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}

	status = sgx_ocall(3, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

