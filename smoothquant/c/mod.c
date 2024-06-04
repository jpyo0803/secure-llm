#include "mod.h"

int32_t ModP(int64_t x) {
    int64_t mod = ((x + P) % (2 * P)) - P;
    if (mod < -P) {
        mod += 2 * P;
    }
    return (int32_t)mod;
}

