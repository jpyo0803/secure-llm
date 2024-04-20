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

void ecall_layernorm(float* x, float* gamma, float* beta, float eps, int B, int M, int N);
void ecall_ReLU(float* x, int B, int M, int N);
void ecall_Softmax(float* x, int B, int M, int N);

sgx_status_t SGX_CDECL ocall_print_string(const char* str);
sgx_status_t SGX_CDECL ocall_start_clock(void);
sgx_status_t SGX_CDECL ocall_get_time(double* retval);
sgx_status_t SGX_CDECL ocall_end_clock(const char* str);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
