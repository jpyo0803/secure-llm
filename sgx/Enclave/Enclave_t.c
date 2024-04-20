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

static sgx_status_t SGX_CDECL sgx_ecall_layernorm(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_layernorm_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_layernorm_t* ms = SGX_CAST(ms_ecall_layernorm_t*, pms);
	ms_ecall_layernorm_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_layernorm_t), ms, sizeof(ms_ecall_layernorm_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_x = __in_ms.ms_x;
	float* _tmp_gamma = __in_ms.ms_gamma;
	float* _tmp_beta = __in_ms.ms_beta;


	ecall_layernorm(_tmp_x, _tmp_gamma, _tmp_beta, __in_ms.ms_eps, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_ReLU(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_ReLU_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_ReLU_t* ms = SGX_CAST(ms_ecall_ReLU_t*, pms);
	ms_ecall_ReLU_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_ReLU_t), ms, sizeof(ms_ecall_ReLU_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_x = __in_ms.ms_x;


	ecall_ReLU(_tmp_x, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_Softmax(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_Softmax_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_Softmax_t* ms = SGX_CAST(ms_ecall_Softmax_t*, pms);
	ms_ecall_Softmax_t __in_ms;
	if (memcpy_s(&__in_ms, sizeof(ms_ecall_Softmax_t), ms, sizeof(ms_ecall_Softmax_t))) {
		return SGX_ERROR_UNEXPECTED;
	}
	sgx_status_t status = SGX_SUCCESS;
	float* _tmp_x = __in_ms.ms_x;


	ecall_Softmax(_tmp_x, __in_ms.ms_B, __in_ms.ms_M, __in_ms.ms_N);


	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[3];
} g_ecall_table = {
	3,
	{
		{(void*)(uintptr_t)sgx_ecall_layernorm, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_ReLU, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_Softmax, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[4][3];
} g_dyn_entry_table = {
	4,
	{
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
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

