#ifndef SECURE_LLM_SMOOTHQUANT_C_STATIC_GLOB_DATA_H
#define SECURE_LLM_SMOOTHQUANT_C_STATIC_GLOB_DATA_H

#define STATIC_LIST_LEN 1000

#include "tensor.h"

struct LayerNormParam {
  struct TensorFloat* gamma;
  struct TensorFloat* beta;
  float eps;
}* layer_norm_param_list[STATIC_LIST_LEN];
int layer_norm_param_id = 0;

struct LinearParamWS8BS8 {
  struct TensorInt8* weight;
  struct TensorInt8* bias;
  float alpha;
  float beta;
}* linear_param_wb8bs8_list[STATIC_LIST_LEN];
int linear_param_ws8bs8_id = 0;

struct LinearParamWS8BFP32 {
  struct TensorInt8* weight;
  struct TensorFloat* bias;
  float alpha;
  float beta;  // should be 1.0
}* linear_param_ws8bfp32_list[STATIC_LIST_LEN];
int linear_param_ws8bfp32_id = 0;

float bmm_param_list[STATIC_LIST_LEN];
int bmm_param_id = 0;

#endif