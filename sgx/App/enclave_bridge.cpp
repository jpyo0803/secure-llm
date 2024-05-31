
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

#include "sgx_urts.h"
#include "Enclave_u.h"

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define TOKEN_FILENAME   "enclave.token"
#define ENCLAVE_FILENAME "enclave.signed.so"

using namespace std::chrono;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
        printf("Error: Unexpected error occurred.\n");
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

thread_local std::chrono::time_point<std::chrono::high_resolution_clock> start;

void ocall_start_clock()
{
	start = std::chrono::high_resolution_clock::now();
}

void ocall_end_clock(const char * str)
{
	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf(str, elapsed.count());
}

double ocall_get_time()
{
    auto now = std::chrono::high_resolution_clock::now();
	return time_point_cast<microseconds>(now).time_since_epoch().count();
}


extern "C"
{

    /*
     * Initialize the enclave
     */
    unsigned long int initialize_enclave(void)
    {

        std::cout << "Initializing Enclave..." << std::endl;

        sgx_enclave_id_t eid = 0;
        sgx_launch_token_t token = {0};
        sgx_status_t ret = SGX_ERROR_UNEXPECTED;
        int updated = 0;

        /* call sgx_create_enclave to initialize an enclave instance */
        /* Debug Support: set 2nd parameter to 1 */
        ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &eid, NULL);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }

        std::cout << "Enclave id: " << eid << std::endl;

        return eid;
    }

    /*
     * Destroy the enclave
     */
    void destroy_enclave(unsigned long int eid)
    {
        std::cout << "Destroying Enclave with id: " << eid << std::endl;
        sgx_destroy_enclave(eid);
    }

    // // Deprecated
    // void LayerNorm(unsigned long eid, float* x, float* gamma, float* beta, float eps, int B, int M, int N) {
    // 	sgx_status_t ret = ecall_Sgx_layernorm(eid,x,gamma,beta,eps,B,M,N);
	// 	if (ret != SGX_SUCCESS) {
	// 		print_error_message(ret);
	// 		throw ret;
	// 	}
    // }

    // // Deprecated
    // void ReLU(unsigned long eid, float* x, int B, int M, int N) {
    // 	sgx_status_t ret = ecall_Sgx_ReLU(eid,x,B,M,N);
	// 	if (ret != SGX_SUCCESS) {
	// 		print_error_message(ret);
	// 		throw ret;
	// 	}
    // }

    // // Deprecated
    // void Softmax(unsigned long eid, float* x, int B, int M, int N) {
    // 	sgx_status_t ret = ecall_Sgx_Softmax(eid,x,B,M,N);
	// 	if (ret != SGX_SUCCESS) {
	// 		print_error_message(ret);
	// 		throw ret;
	// 	}
    // }

    
    int Sgx_Set_Hidden_States(unsigned long eid, float* hidden_states, int B, int M, int N) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Hidden_States(eid,hidden_states,B,M,N,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Copy_Hidden_States(unsigned long eid, int src_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Copy_Hidden_States(eid,src_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Set_Layer_Norm_Param(unsigned long eid, float* gamma, float* beta, int N, float eps) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Layer_Norm_Param(eid,gamma,beta,N,eps,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Layer_Norm_Q(unsigned long eid, int src_id, int layer_norm_param_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Layer_Norm_Q(eid,src_id,layer_norm_param_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Set_Linear_Param_WS8BS8(unsigned long eid, char* weight, char* bias, int M, int N, float alpha, float beta) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Linear_Param_WS8BS8(eid,weight,bias,M,N,alpha,beta,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Set_Linear_Param_WS8BFP32(unsigned long eid, char* weight, float* bias, int M, int N, float alpha) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Linear_Param_WS8BFP32(eid,weight,bias,M,N,alpha,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    void Sgx_Get_Tensor_Dim_Int32(unsigned long eid, int src_id, int* dim) {
        sgx_status_t ret = ecall_Sgx_Get_Tensor_Dim_Int32(eid,src_id,dim);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Sgx_Get_Tensor_Int32(unsigned long eid, int src_id, int* out) {
        sgx_status_t ret = ecall_Sgx_Get_Tensor_Int32(eid,src_id,out);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    int Sgx_Set_Tensor_Int32(unsigned long eid, int* data, int B, int M, int N) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Tensor_Int32(eid,data,B,M,N,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Get_Encrypted_Tensor_Opr1_Int32(unsigned long eid, int src_id, int* out) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Get_Encrypted_Tensor_Opr1_Int32(eid,src_id,out,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Generate_Decryption_Key_Opr1_Int32(unsigned long eid, int blind_factor_id, int linear_param_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Generate_Decryption_Key_Opr1_Int32(eid,blind_factor_id,linear_param_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Set_Decrypted_Tensor_Opr1_Int32(unsigned long eid, int* data, int B, int M, int N, int decryption_key_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Decrypted_Tensor_Opr1_Int32(eid,data,B,M,N,decryption_key_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    void Sgx_Get_Encrypted_Tensor_Opr2_Int32(unsigned long eid, int src_id1, int src_id2, int* out1, int* out2, int* blind_factor_ids) {
        sgx_status_t ret = ecall_Sgx_Get_Encrypted_Tensor_Opr2_Int32(eid,src_id1,src_id2,out1,out2, blind_factor_ids);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    int Sgx_Generate_Decryption_Key_Opr2_Int32(unsigned long eid, int src_id1, int src_id2, int blind_factor_u_id, int blind_factor_v_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Generate_Decryption_Key_Opr2_Int32(eid,src_id1,src_id2,blind_factor_u_id,blind_factor_v_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Set_Decrypted_Tensor_Opr2_Int32(unsigned long eid, int* data, int B, int M, int N, int unblind_factor_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Decrypted_Tensor_Opr2_Int32(eid,data,B,M,N,unblind_factor_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    void Sgx_Get_Tensor_Dim_Int8(unsigned long eid, int src_id, int* dim) {
        sgx_status_t ret = ecall_Sgx_Get_Tensor_Dim_Int8(eid,src_id,dim);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Sgx_Get_Tensor_Int8(unsigned long eid, int src_id, char* out) {
        sgx_status_t ret = ecall_Sgx_Get_Tensor_Int8(eid,src_id,out);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    int Sgx_Set_Tensor_Int8(unsigned long eid, char* data, int B, int M, int N) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Tensor_Int8(eid,data,B,M,N,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    void Sgx_Get_Tensor_Dim_Float(unsigned long eid, int src_id, int* dim) {
        sgx_status_t ret = ecall_Sgx_Get_Tensor_Dim_Float(eid,src_id,dim);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    void Sgx_Get_Tensor_Float(unsigned long eid, int src_id, float* out) {
        sgx_status_t ret = ecall_Sgx_Get_Tensor_Float(eid,src_id,out);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
    }

    int Sgx_Set_Tensor_Float(unsigned long eid, float* data, int B, int M, int N) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Tensor_Float(eid,data,B,M,N,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Compute_Epilogue_WS8BS8(unsigned long eid, int src_id, int linear_param_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Compute_Epilogue_WS8BS8(eid,src_id,linear_param_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Compute_Epilogue_WS8BFP32(unsigned long eid, int src_id, int linear_param_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Compute_Epilogue_WS8BFP32(eid,src_id,linear_param_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Compute_Epilogue_BMM(unsigned long eid, int src_id, int bmm_param_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Compute_Epilogue_BMM(eid,src_id,bmm_param_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_ReLU(unsigned long eid, int src_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_ReLU(eid,src_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Softmax(unsigned long eid, int src_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Softmax(eid,src_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Quantize_Post_Softmax(unsigned long eid, int src_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Quantize_Post_Softmax(eid,src_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Cast_From_Float_To_Int8(unsigned long eid, int src_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Cast_From_Float_To_Int8(eid,src_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Cast_From_Float_To_Int32(unsigned long eid, int src_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Cast_From_Float_To_Int32(eid,src_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Cast_From_Int8_To_Int32(unsigned long eid, int src_id) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Cast_From_Int8_To_Int32(eid,src_id,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Set_Bmm_Param(unsigned long eid, float alpha) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Set_Bmm_Param(eid,alpha,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }

    int Sgx_Residual_Add(unsigned long eid, int residual, int hidden_states) {
        int ret_id;
        sgx_status_t ret = ecall_Sgx_Residual_Add(eid,residual,hidden_states,&ret_id);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        return ret_id;
    }
}

/* Application entry */
int main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    try {
        sgx_enclave_id_t eid = initialize_enclave();

        std::cout << "Enclave id: " << eid << std::endl;


        printf("Enter a character to destroy enclave ...\n");
        getchar();

        // Destroy the enclave
        sgx_destroy_enclave(eid);

        printf("Info: Enclave Launcher successfully returned.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return 0;
    }
    catch (int e)
    {
        printf("Info: Enclave Launch failed!.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return -1;
    }
}
