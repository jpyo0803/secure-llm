#ifndef SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H
#define SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H

struct LayerNormParams {
  float* gamma;
  float* beta;
  int N;
  float eps;
};

struct LayerNormParams* layer_norm_params_list[300];
int g_layer_id = 0;

void LS_SetLayerNormParams(float* gamma, float* beta, int N, float eps);

void LS_LayerNorm(float* x, int B, int M,
                  int N, int layer_id);

void LS_ReLU(float* x, int B, int M, int N);

void LS_Softmax(float* x, int B, int M, int N);

void LS_ResidualAdd(float* x, float* y, int B, int M, int N);

#endif