#ifndef SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H
#define SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H

struct LayerNormParams {
  float* gamma;
  float* beta;
  int N;
  float eps;
};

struct TensorFloat {
  float* data;
  int B;
  int M;
  int N;
  unsigned int num_bytes;
};

struct TensorInt32 {
  int* data;
  int B;
  int M;
  int N;
  unsigned int num_bytes;
};

struct TensorUint32 {
  unsigned int* data;
  int B;
  int M;
  int N;
  unsigned int num_bytes;
};

struct TensorInt8 {
  char* data;
  int B;
  int M;
  int N;
  unsigned int num_bytes;
};

struct LayerNormParams* layer_norm_params_list[300];
int g_layer_id = 0;

void LS_SetLayerNormParams(float* gamma, float* beta, int N, float eps);

void LS_LayerNorm(float* x, int B, int M, int N, int layer_id);

void LS_ReLU(float* x, int B, int M, int N);

void LS_Softmax(float* x, int B, int M, int N);

void LS_ResidualAdd(float* x, float* y, int B, int M, int N);

void LS_SetHiddenStatesInternal(float* hidden_states, int B, int M,
                                int N);  // Set data 1

void LS_CopyResidual1Internal();

void LS_SelfAttnLayerNormQInternal(int layer_id);

void LS_FinalLayerNormQInternal(int layer_id);

void LS_GetResidual1Internal(float* out, int B, int M, int N);

void LS_GetSelfAttnLayerNormQInternal(char* q, int B, int M, int N);

void LS_EncryptHiddenStates(float* x, int B, int M, int N);

#endif