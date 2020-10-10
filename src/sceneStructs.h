#pragma once

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

struct Triangle {
    int idx;
    glm::vec3 vert[3];
    glm::vec3 nor;
    glm::vec2 uv[3];
public:
    Triangle() : idx(-1) {}
    Triangle(int index) : idx(index) {}
};

struct OctreeNode {
    int idx;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int triangleIdx;
    int childStartIndex;
    OctreeNode(int index, glm::vec3 minV, glm::vec3 maxV, int triangleIdx) : 
        idx(index), bboxMin(minV), bboxMax(maxV), triangleIdx(triangleIdx)
    {}
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int startTriangleIndex;
    int endTriangleIndex;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;

    int startOctreeIndex;
    int endOctreeIndex;
    //Triangle* triangles;
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
  glm::vec2 uv;
};
