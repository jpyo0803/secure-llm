#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_layernorm_t {
	float* ms_x;
	float* ms_gamma;
	float* ms_beta;
	float ms_eps;
	int ms_B;
	int ms_M;
	int ms_N;
} ms_ecall_layernorm_t;

typedef struct ms_ecall_ReLU_t {
	float* ms_x;
	int ms_B;
	int ms_M;
	int ms_N;
} ms_ecall_ReLU_t;

typedef struct ms_ecall_Softmax_t {
	float* ms_x;
	int ms_B;
	int ms_M;
	int ms_N;
} ms_ecall_Softmax_t;

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
sgx_status_t ecall_layernorm(sgx_enclave_id_t eid, float* x, float* gamma, float* beta, float eps, int B, int M, int N)
{
	sgx_status_t status;
	ms_ecall_layernorm_t ms;
	ms.ms_x = x;
	ms.ms_gamma = gamma;
	ms.ms_beta = beta;
	ms.ms_eps = eps;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_ReLU(sgx_enclave_id_t eid, float* x, int B, int M, int N)
{
	sgx_status_t status;
	ms_ecall_ReLU_t ms;
	ms.ms_x = x;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_Softmax(sgx_enclave_id_t eid, float* x, int B, int M, int N)
{
	sgx_status_t status;
	ms_ecall_Softmax_t ms;
	ms.ms_x = x;
	ms.ms_B = B;
	ms.ms_M = M;
	ms.ms_N = N;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

