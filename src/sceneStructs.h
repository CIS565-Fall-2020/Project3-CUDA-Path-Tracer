#ifndef SCENESTRUCTS_H
#define SCENESTRUCTS_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    int triangleStart;
    int triangleCount;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    inline void getBoundingBox(glm::vec3& minCorner, glm::vec3& maxCorner) const;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
};

// Boolean function for path termination
struct path_terminated {
    __host__ __device__ bool operator()(const PathSegment& segment) {
        return segment.remainingBounces <= 0;
    }
};

// Boolean function for path continuing
struct path_continue {
    __host__ __device__ bool operator()(const PathSegment& segment) {
        return segment.remainingBounces > 0;
    }
};

// Sort function for different materials of intersection
struct path_sort {
    __host__ __device__ bool operator()(const ShadeableIntersection& a,
        const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};

inline void Geom::getBoundingBox(glm::vec3& minCorner, glm::vec3& maxCorner) const {
    if (type == SPHERE) {
        float s = std::max(scale.x, std::max(scale.y, scale.z));
        glm::vec3 halfSide(s * 0.5f);
        minCorner = translation - halfSide;
        maxCorner = translation + halfSide;
    }
    else if (type == CUBE) {
        minCorner = glm::vec3(FLT_MAX);
        maxCorner = glm::vec3(-FLT_MAX);
        std::vector<glm::vec3> corners = {
            glm::vec3(-0.5, -0.5, -0.5),
            glm::vec3(-0.5, -0.5, 0.5),
            glm::vec3(-0.5, 0.5, -0.5),
            glm::vec3(-0.5, 0.5, 0.5),
            glm::vec3(0.5, -0.5, -0.5),
            glm::vec3(0.5, -0.5, 0.5),
            glm::vec3(0.5, 0.5, -0.5),
            glm::vec3(0.5, 0.5, 0.5)
        };
        for (int i = 0; i < 8; i++) {
            corners[i] = glm::vec3(transform * glm::vec4(corners[i], 1));
            minCorner.x = std::min(corners[i].x, minCorner.x);
            minCorner.y = std::min(corners[i].y, minCorner.y);
            minCorner.z = std::min(corners[i].z, minCorner.z);
            maxCorner.x = std::max(corners[i].x, maxCorner.x);
            maxCorner.y = std::max(corners[i].y, maxCorner.y);
            maxCorner.z = std::max(corners[i].z, maxCorner.z);
        }
    }
    else {
        std::cout << "Geom::getBoundingBox: do not support type " << type << std::endl;
    }
}

#endif // !SCENESTRUCTS_H
