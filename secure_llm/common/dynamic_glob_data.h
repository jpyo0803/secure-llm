#ifndef SECURE_LLM_SMOOTHQUANT_C_DYNAMIC_GLOB_DATA_H
#define SECURE_LLM_SMOOTHQUANT_C_DYNAMIC_GLOB_DATA_H

#include "tensor.h"

#define DYNAMIC_LIST_LEN 100

struct TensorFloat* tensor_float_list[DYNAMIC_LIST_LEN];
int tensor_float_id = 0;

struct TensorInt32* tensor_int32_list[DYNAMIC_LIST_LEN];
int tensor_int32_id = 0;

struct TensorUint32* tensor_uint32_list[DYNAMIC_LIST_LEN];
int tensor_uint32_id = 0;

struct TensorInt8* tensor_int8_list[DYNAMIC_LIST_LEN];
int tensor_int8_id = 0;


#endif