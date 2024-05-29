#ifndef SECURE_LLM_SGX_ENCLAVE_SGX_LAYER_STRUCT_H
#define SECURE_LLM_SGX_ENCLAVE_SGX_LAYER_STRUCT_H

#include <vector>
#include <iostream>


extern "C" {
void LayerNorm(float* x, float* gamma, float* beta, float eps, int B, int M, int N);

void ReLU(float* x, int B, int M, int N);

void Softmax(float* x, int B, int M, int N);
}

#endif 