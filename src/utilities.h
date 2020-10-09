#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <random>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}

template <typename Rand> inline std::vector<glm::vec2> generateStratifiedSamples2D(std::size_t sqrtCount, Rand &rand) {
    std::vector<glm::vec2> samples;
    samples.reserve(sqrtCount * sqrtCount);
    float rng = 1.0f / sqrtCount;
    for (std::size_t y = 0; y < sqrtCount; ++y) {
        for (std::size_t x = 0; x < sqrtCount; ++x) {
            samples.emplace_back(x * rng, y * rng);
        }
    }
    std::shuffle(samples.begin(), samples.end(), rand);
    return samples;
}
template <typename Rand> inline std::vector<int> generateStratifiedSamplesChoice(std::size_t count, Rand &rand) {
    std::vector<int> samples;
    samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        samples.emplace_back(static_cast<int>(i));
    }
    std::shuffle(samples.begin(), samples.end(), rand);
    return samples;
}
