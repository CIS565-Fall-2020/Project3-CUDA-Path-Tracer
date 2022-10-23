#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <cuda.h>

#include "sceneStructs.h"

typedef float Float;
typedef glm::vec2 vc2;
typedef glm::vec3 vc3;
typedef glm::vec4 vc4;

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define InvPI             1.f / PI
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
    // Jack12 add
    // ref https://github.com/syoyo/tinygltf/blob/master/examples/glview/glview.cc
    //extern std::string GetFilePathExtension(const std::string& FileName);
   // __host__ __device__ glm::mat4 utilityCore::device_buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
}

struct aabbBounds;
struct Geom;
struct Triangle;
namespace geometry {
    extern aabbBounds bbUnion(const aabbBounds& a, const aabbBounds& b);
    extern aabbBounds bbUnion(const aabbBounds& a, const glm::vec3& b);
    extern void aabbForImplicit(aabbBounds& aabb, const Geom& geom);
    extern void aabbForTriangle(aabbBounds& aabb, const Triangle& geom);
    extern aabbBounds aabbForVertex(glm::vec3* verts, int num);
}

namespace Math {
    __host__ __device__ inline Float SphericalTheta(const vc3& v) {
        return glm::acos(glm::clamp(v.z, (Float)-1, (Float)1));
    }

    __host__ __device__ inline Float SphericalPhi(const vc3& v) {
        Float p = glm::atan(v.y, v.x);
        return (p < 0) ? (p + 2 * PI) : p;
    }

    __host__ __device__ inline float luminance(const glm::vec3& color) {
        // For the sRGB colorspace, the relative luminance of a color is defined as L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        const glm::vec3 T(.2126f, .7152f, .0722f);
        return glm::dot(color, T);
    }
}