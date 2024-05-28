#ifndef SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H
#define SECURE_LLM_SMOOTHQUANT_CPP_LAYER_STRUCT_H

struct LayerNormParams {
  float* gamma;
  float* beta;
  int N;
  float eps;
};

struct LinearParamsI8I8 {
  char* weight;  // M by N
  char* bias;    // 1 by N
  int M;
  int N;
  float alpha;
  float beta;
};

struct LinearParamsI8FP32 {
  char* weight;  // M by N
  float* bias;   // 1 by N
  int M;
  int N;
  float alpha;
  float beta;  // should be 1.0
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

struct LinearParamsI8I8*
    linear_params_i8i8i8_list[600];  // i8i8i8i8 + i8i8i8fp32
int g_linear_i8i8i8_id = 0;

struct LinearParamsI8FP32* linear_params_i8i8fp32fp32_list[300];
int g_linear_i8i8fp32fp32_id = 0;

struct LayerNormParams* layer_norm_params_list[300];
int g_layer_id = 0;

struct TensorInt32* blind_factor_list[500];

struct TensorInt32* unblind_factor_xv_list[500];
struct TensorInt32* unblind_factor_uy_list[500];
struct TensorInt32* unblind_factor_uv_list[500];

void LS_SetLinearParams_I8I8I8(char* weight, char* bias, int M, int N,
                               float alpha, float beta);

void LS_SetLinearParams_I8I8FP32FP32(char* weight, float* bias, int M, int N,
                                     float alpha, float beta);

void LS_SetLayerNormParams(float* gamma, float* beta, int N, float eps);

void LS_LayerNorm(float* x, int B, int M, int N, int layer_id);

void LS_ReLU(float* x, int B, int M, int N);

void LS_Softmax(float* x, int B, int M, int N);

void LS_ResidualAdd(float* x, float* y, int B, int M, int N);

void LS_Blind_Input_Op1_I8I8I8(int* x, int B, int M, int N,
                               int blind_factor_id);

void LS_Unblind_Output_Op1_I8I8I8(int* x, int B, int M, int N,
                                  int blind_factor_id, int linear_id);

void LS_Blind_Input_Op1_I8FP32FP32(int* x, int B, int M, int N,
                                   int blind_factor_id);

void LS_Unblind_Output_Op1_I8FP32FP32(int* x, int B, int M, int N,
                                      int blind_factor_id, int linear_id);

void LS_Blind_Input_Op2_I8I8(int* x, int* y, int B, int M, int K, int N,
                             int blind_factor_id_u, int blind_factor_id_v);

void LS_Unblind_Output_Op2_I8I8(int* x, int B, int M, int N,
                                int blind_factor_id_u, int blind_factor_id_v);

void LS_ComputeEpilogue_I8I8I8(float* x, int B, int M, int N, int linear_id);

void LS_ComputeEpilogue_I8FP32FP32(float* x, int B, int M, int N,
                                   int linear_id);

void LS_SetHiddenStatesInternal(float* hidden_states, int B, int M,
                                int N);  // Set data 1

void LS_CopyResidual1Internal();

void LS_SelfAttnLayerNormQInternal(int layer_id);

void LS_FinalLayerNormQInternal(int layer_id);

void LS_GetResidual1Internal(float* out, int B, int M, int N);

void LS_GetSelfAttnLayerNormQInternal(char* q, int B, int M, int N);

void LS_EncryptHiddenStates(float* x, int B, int M, int N);

#endif