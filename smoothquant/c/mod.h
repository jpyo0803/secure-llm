#ifndef SECURE_LLM_SMOOTHQUANT_C_MOD_H
#define SECURE_LLM_SMOOTHQUANT_C_MOD_H

#define P ((1<<21) - 3)

#include <stdint.h>


int32_t ModP(int64_t x);


#endif