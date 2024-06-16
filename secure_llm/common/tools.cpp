#include "tools.h"
#include <random>

uint32_t GenerateRandomNumber_Int32() {
    // Initialize static objects once
    static std::random_device rd;  // Seed
    static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    static std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
    
    // Generate and return the random number
    return dis(gen);
}