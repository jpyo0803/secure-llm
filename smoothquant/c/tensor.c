#include "tensor.h"
#include <stdlib.h>

struct TensorInt32* CreateTensorInt32(int B, int M, int N) {
  struct TensorInt32* tensor =
      (struct TensorInt32*)malloc(sizeof(struct TensorInt32));
  tensor->num_bytes = B * M * N * sizeof(int);
  tensor->data = (int*)malloc(tensor->num_bytes);
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorInt32(struct TensorInt32* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorUint32* CreateTensorUint32(int B, int M, int N) {
  struct TensorUint32* tensor =
      (struct TensorUint32*)malloc(sizeof(struct TensorUint32));
  tensor->num_bytes = B * M * N * sizeof(unsigned int);
  tensor->data = (unsigned int*)malloc(tensor->num_bytes);
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorUint32(struct TensorUint32* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorFloat* CreateTensorFloat(int B, int M, int N) {
  struct TensorFloat* tensor =
      (struct TensorFloat*)malloc(sizeof(struct TensorFloat));
  tensor->num_bytes = B * M * N * sizeof(float);
  tensor->data = (float*)malloc(tensor->num_bytes);
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorFloat(struct TensorFloat* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorInt8* CreateTensorInt8(int B, int M, int N) {
  struct TensorInt8* tensor =
      (struct TensorInt8*)malloc(sizeof(struct TensorInt8));
  tensor->num_bytes = B * M * N * sizeof(char);
  tensor->data = (char*)malloc(tensor->num_bytes);
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorInt8(struct TensorInt8* tensor) {
  free(tensor->data);
  free(tensor);
}

