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

struct StratifiedSampler {
public:
    void resize(std::size_t size) {
        _sqrtSize = size;
        _samples.resize(size * size);
        _current = size * size;
    }
    void restart() {
        _samples.clear();
        float rng = range();
        for (std::size_t y = 0; y < _sqrtSize; ++y) {
            for (std::size_t x = 0; x < _sqrtSize; ++x) {
                _samples.emplace_back(x * rng, y * rng);
            }
        }

        std::default_random_engine rand(std::random_device{}());
        std::shuffle(_samples.begin(), _samples.end(), rand);
        _current = 0;
    }

    const std::vector<glm::vec2> &pool() const {
        return _samples;
    }
    float range() const {
        return 1.0f / _sqrtSize;
    }

    glm::vec2 next() {
        if (_current >= _samples.size()) {
            restart();
        }
        return _samples[_current++];
    }
protected:
    std::size_t _current = std::numeric_limits<std::size_t>::max();
    std::size_t _sqrtSize = 0;
    std::vector<glm::vec2> _samples;
};
