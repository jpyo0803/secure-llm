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

struct LinearParam {
  struct TensorInt8* weight;
  struct TensorInt8* bias_int8;
  struct TensorFloat* bias_float;
  float alpha;
  float beta;

  int is_bias_fp32;
}* linear_param_list[STATIC_LIST_LEN];
int linear_param_id = 0;

float bmm_param_list[STATIC_LIST_LEN];
int bmm_param_id = 0;

#endif