#include "tensor.h"

#include <stdlib.h>

extern "C" {
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

struct TensorInt32* CreateTensorInt32FromData(int* data, int B, int M, int N) {
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);
  for (int i = 0; i < B * M * N; i++) {
    tensor->data[i] = data[i];
  }
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

struct TensorFloat* CreateTensorFloatFromData(float* data, int B, int M,
                                              int N) {
  struct TensorFloat* tensor = CreateTensorFloat(B, M, N);
  for (int i = 0; i < B * M * N; i++) {
    tensor->data[i] = data[i];
  }
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

struct TensorInt8* CreateTensorInt8FromData(char* data, int B, int M, int N) {
  struct TensorInt8* tensor = CreateTensorInt8(B, M, N);
  for (int i = 0; i < B * M * N; i++) {
    tensor->data[i] = data[i];
  }
  return tensor;
}

void DeleteTensorInt8(struct TensorInt8* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorInt32* MatmulS32S32S32(struct TensorInt32* X,
                                    struct TensorInt32* Y) {
  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->N;

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
          sum +=
              X->data[b * M * K + m * K + k] * Y->data[b * K * N + k * N + n];
        }
        Z->data[b * M * N + m * N + n] = sum;
      }
    }
  }
  return Z;
}

struct TensorInt32* MatmulS32S8S32(struct TensorInt32* X,
                                   struct TensorInt8* Y) {
  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->N;

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
          sum += X->data[b * M * K + m * K + k] *
                 (int)Y->data[b * K * N + k * N + n];
        }
        Z->data[b * M * N + m * N + n] = sum;
      }
    }
  }
  return Z;
}

struct TensorInt32* TransposeLastTwoDimsInt32(struct TensorInt32* X) {
  int B = X->B;
  int M = X->M;
  int N = X->N;

  struct TensorInt32* Y = CreateTensorInt32(B, N, M);
  // transpose last two dimension for example (B, M, N) -> (B, N, M)
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        Y->data[b * N * M + n * M + m] = X->data[b * M * N + m * N + n];
      }
    }
  }
  return Y;
}

}