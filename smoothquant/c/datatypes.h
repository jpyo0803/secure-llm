#ifndef SECURE_LLM_SMOOTHQUANT_C_DATATYPES_H
#define SECURE_LLM_SMOOTHQUANT_C_DATATYPES_H

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

#endif