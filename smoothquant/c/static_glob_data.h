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

  int obfuscation_ratio;
  struct TensorInt32*
      blind_factors_set;  // Sample from here, row-wise blind factor
  struct TensorInt32* precomputed_unblind_factors;

  struct TensorInt32* chosen_keys;
}* linear_param_list[STATIC_LIST_LEN];
int linear_param_id = 0;

float bmm_param_list[STATIC_LIST_LEN];
int bmm_param_id = 0;

struct TensorInt32* qk_blind_factor_u = NULL; // does not change its size
struct TensorInt32* qk_blind_factor_v = NULL; // does not change its size
struct TensorInt32* qk_uy_unblind_factor_accum = NULL;
struct TensorInt32* qk_uv_dot = NULL;

struct TensorInt32* pv_blind_factor_u = NULL; // Change during generation
struct TensorInt32* pv_blind_factor_v = NULL; // does not change its size
struct TensorInt32* pv_uy_unblind_factor_accum = NULL;
struct TensorInt32* pv_uv_unblind_factor_accum = NULL;
#endif