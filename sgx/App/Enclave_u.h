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

sgx_status_t ecall_layernorm(sgx_enclave_id_t eid, float* x, float* gamma, float* beta, float eps, int B, int M, int N);
sgx_status_t ecall_ReLU(sgx_enclave_id_t eid, float* x, int B, int M, int N);
sgx_status_t ecall_Softmax(sgx_enclave_id_t eid, float* x, int B, int M, int N);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
