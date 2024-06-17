#include "tools_sgx.h"
#include <sgx_trts.h>

uint32_t GenerateRandomNumber_Uint32() {
  uint32_t rand_val;
  sgx_read_rand((unsigned char*)&rand_val, sizeof(rand_val));
  return rand_val;
}