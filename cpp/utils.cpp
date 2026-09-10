#include "utils.h"

#include <random>

float random_float(float min, float max) {
    static std::random_device random_device;
    static std::mt19937 generator(random_device());
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}
