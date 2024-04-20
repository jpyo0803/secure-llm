#ifndef SGX_SMOOTHQUANT_SMOOTHQUANT_CPP_LAYER_H
#define SGX_SMOOTHQUANT_SMOOTHQUANT_CPP_LAYER_H

#include <vector>
#include <iostream>


extern "C" {
void LayerNorm(float* x, float* gamma, float* beta, float eps, int B, int M, int N);

void ReLU(float* x, int B, int M, int N);

void Softmax(float* x, int B, int M, int N);
}

#endif 