#ifndef SECURE_LLM_SMOOTHQUANT_C_TENSOR_H
#define SECURE_LLM_SMOOTHQUANT_C_TENSOR_H

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

struct TensorInt32* CreateTensorInt32(int B, int M, int N);
void DeleteTensorInt32(struct TensorInt32* tensor);

struct TensorUint32* CreateTensorUint32(int B, int M, int N);
void DeleteTensorUint32(struct TensorUint32* tensor);

struct TensorFloat* CreateTensorFloat(int B, int M, int N);
void DeleteTensorFloat(struct TensorFloat* tensor);

struct TensorInt8* CreateTensorInt8(int B, int M, int N);
void DeleteTensorInt8(struct TensorInt8* tensor);




#endif