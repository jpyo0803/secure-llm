#include "tensor.h"

#include <immintrin.h>
#include <sgx_trts.h>
#include <stdlib.h>

extern "C" {

struct TensorInt32* CreateTensorInt32(int B, int M, int N) {
  struct TensorInt32* tensor =
      (struct TensorInt32*)malloc(sizeof(struct TensorInt32));
  tensor->num_bytes = B * M * N * sizeof(int);
  // tensor->data = (int*)malloc(tensor->num_bytes);
  tensor->data = (int*)aligned_alloc(64, B * M * N * sizeof(int));
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

struct TensorInt32* CreateTensorInt32FromData(int* data, int B, int M, int N) {
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);
  int total_elements = B * M * N;

  int i;
  for (i = 0; i <= total_elements - 16; i += 16) {
    // Load 16 int32 elements from data
    __m512i src_vec = _mm512_loadu_si512((__m512i*)&data[i]);

    // Store 16 int32 elements to tensor->data
    _mm512_storeu_si512((__m512i*)&tensor->data[i], src_vec);
  }

  // Copy any remaining elements (if total_elements is not a multiple of 16)
  for (; i < total_elements; ++i) {
    tensor->data[i] = data[i];
  }

  return tensor;
}

struct TensorInt32* CreateTensorInt32FromRandom(int low, int high, int B, int M,
                                                int N) {
  struct TensorInt32* tensor = CreateTensorInt32(B, M, N);
  int total_elements = B * M * N;

  // Generate random int32 elements in the range [low, high]
  for (int i = 0; i < total_elements; ++i) {
    uint32_t rand_val;
    sgx_read_rand((unsigned char*)&rand_val, sizeof(rand_val));
    tensor->data[i] = low + (rand_val % (high - low + 1));
  }

  return tensor;
}

void DeleteTensorInt32(struct TensorInt32* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorFloat* CreateTensorFloat(int B, int M, int N) {
  struct TensorFloat* tensor =
      (struct TensorFloat*)malloc(sizeof(struct TensorFloat));
  tensor->num_bytes = B * M * N * sizeof(float);
  tensor->data = (float*)aligned_alloc(
      64, tensor->num_bytes);  // Use aligned_alloc for alignment
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

struct TensorFloat* CreateTensorFloatFromData(float* data, int B, int M,
                                              int N) {
  struct TensorFloat* tensor = CreateTensorFloat(B, M, N);
  int total_elements = B * M * N;

  int i;
  for (i = 0; i <= total_elements - 16; i += 16) {
    // Load 16 float elements from data
    __m512 src_vec = _mm512_loadu_ps(&data[i]);

    // Store 16 float elements to tensor->data
    _mm512_storeu_ps(&tensor->data[i], src_vec);
  }

  // Copy any remaining elements (if total_elements is not a multiple of 16)
  for (; i < total_elements; ++i) {
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
  tensor->data = (char*)aligned_alloc(64, B * M * N * sizeof(char));
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

struct TensorInt8* CreateTensorInt8FromData(char* data, int B, int M, int N) {
  struct TensorInt8* tensor = CreateTensorInt8(B, M, N);
  int total_elements = B * M * N;

  int i;
  for (i = 0; i <= total_elements - 64; i += 64) {
    // Load 64 int8 elements from data
    __m512i src_vec = _mm512_loadu_si512((__m512i*)&data[i]);

    // Store 64 int8 elements to tensor->data
    _mm512_storeu_si512((__m512i*)&tensor->data[i], src_vec);
  }

  // Copy any remaining elements (if total_elements is not a multiple of 64)
  for (; i < total_elements; ++i) {
    tensor->data[i] = data[i];
  }

  return tensor;
}

void DeleteTensorInt8(struct TensorInt8* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorInt32* MatmulS32S32S32_Naive(struct TensorInt32* X,
                                          struct TensorInt32* Y){
  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->M;  // Y's second dimension should be N since Y has dimensions (B,

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int sum = 0;
        for (int k = 0; k < K; ++k) {
          sum +=
              X->data[b * M * K + m * K + k] * Y->data[b * N * K + n * K + k];
        }
        Z->data[b * M * N + m * N + n] = sum;
      }
    }
  }
  return Z;
}


struct TensorInt64* CreateTensorInt64(int B, int M, int N) {
  struct TensorInt64* tensor =
      (struct TensorInt64*)malloc(sizeof(struct TensorInt64));
  tensor->num_bytes = B * M * N * sizeof(int64_t);
  tensor->data = (int64_t*)aligned_alloc(64, B * M * N * sizeof(int64_t));
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

void DeleteTensorInt64(struct TensorInt64* tensor) {
  free(tensor->data);
  free(tensor);
}


struct TensorInt32* MatmulS32S32S32_ModP_Naive(struct TensorInt32* X,
                                               struct TensorInt32* Y) {
  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->M;  // Y's second dimension should be N since Y has dimensions (B,

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int64_t sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += (int64_t)X->data[b * M * K + m * K + k] *
                 (int64_t)Y->data[b * N * K + n * K + k];
        }
        Z->data[b * M * N + m * N + n] = ModP(sum);
      }
    }
  }
  return Z;
}

struct TensorInt32* MatmulS32S8S32_ModP_Naive(struct TensorInt32* X,
                                              struct TensorInt8* Y) {
  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->M;  // Y's second dimension should be N since Y has dimensions (B,

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int64_t sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += (int64_t)X->data[b * M * K + m * K + k] *
                 (int64_t)Y->data[b * N * K + n * K + k];
        }
        Z->data[b * M * N + m * N + n] = ModP(sum);
      }
    }
  }

  return Z;
}

struct TensorInt32* MatmulS32S8S32_Naive(struct TensorInt32* X,
                                         struct TensorInt8* Y) {
  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->M;

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
          sum += (int)X->data[b * M * K + m * K + k] * (int)Y->data[n * K + k];
        }
        Z->data[b * M * N + m * N + n] = sum;
      }
    }
  }

  return Z;
}


}