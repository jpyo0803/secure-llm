#include "tensor.h"

#include <stdlib.h>
#include <immintrin.h>

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

void DeleteTensorInt32(struct TensorInt32* tensor) {
  free(tensor->data);
  free(tensor);
}

struct TensorFloat* CreateTensorFloat(int B, int M, int N) {
  struct TensorFloat* tensor =
      (struct TensorFloat*)malloc(sizeof(struct TensorFloat));
  tensor->num_bytes = B * M * N * sizeof(float);
  tensor->data = (float*)aligned_alloc(64, tensor->num_bytes); // Use aligned_alloc for alignment
  tensor->B = B;
  tensor->M = M;
  tensor->N = N;
  return tensor;
}

struct TensorFloat* CreateTensorFloatFromData(float* data, int B, int M, int N) {
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

struct TensorInt32* MatmulS32S32S32(struct TensorInt32* X,
                                    struct TensorInt32* Y) {
  int B = X->B;
  int M = X->M;
  int K = X->N;
  int N = Y->M;

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        __m512i sum_vec = _mm512_setzero_si512();
        for (int k = 0; k < K; k += 16) {
            // Load 16 int32 elements from X
            __m512i x_vec = _mm512_loadu_si512(&X->data[b * M * K + m * K + k]);
            
            // Load 16 int32 elements from Y, treating it as if it were transposed
            __m512i y_vec = _mm512_loadu_si512(&Y->data[b * N * K + n * K + k]);

            // Multiply and accumulate
            __m512i mul_vec = _mm512_mullo_epi32(x_vec, y_vec);
            sum_vec = _mm512_add_epi32(sum_vec, mul_vec);
        }
        
        // Horizontally add the elements in sum_vec to get the final sum for this element
        int temp[16];
        _mm512_storeu_si512((__m512i*)temp, sum_vec);
        int sum = 0;
        for (int i = 0; i < 16; ++i) {
            sum += temp[i];
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
  int N = Y->M;

  struct TensorInt32* Z = CreateTensorInt32(B, M, N);

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m512i sum_vec = _mm512_setzero_si512();
            for (int k = 0; k < K; k += 16) {
                // Load 16 int32 elements from X
                __m512i x_vec = _mm512_loadu_si512(&X->data[b * M * K + m * K + k]);
                
                // Load 16 int8 elements from Y and expand to int32
                __m128i y_vec_8 = _mm_loadu_si128((__m128i*)&Y->data[n * K + k]);
                __m512i y_vec = _mm512_cvtepi8_epi32(y_vec_8);
                
                // Multiply and accumulate
                __m512i mul_vec = _mm512_mullo_epi32(x_vec, y_vec);
                sum_vec = _mm512_add_epi32(sum_vec, mul_vec);
            }
            
            // Horizontally add the elements in sum_vec to get the final sum for this element
            int temp[16];
            _mm512_storeu_si512((__m512i*)temp, sum_vec);
            int sum = 0;
            for (int i = 0; i < 16; ++i) {
                sum += temp[i];
            }

            Z->data[b * M * N + m * N + n] = sum;
        }
    }
  }
  return Z;
}

}