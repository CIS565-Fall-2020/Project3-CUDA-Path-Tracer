#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
    TRIANGLE,
    TANGLECUBE,
    BOUND_BOX,
};

enum MaterialType {
    DIFFUSE,
    MIRROR,
    GLOSSY,
    DIELECTRIC,
    GLASS,
    EMISSIVE,
};

enum TextureType {
    NO_TEXTURE,
    FBM,
    NOISE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int geomId;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // Triangles only
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;

    // Bounding box
    glm::vec3 max_point;
    glm::vec3 min_point;

    // Mesh only
    int triangleStart;
    int numTriangles;
};


struct Material {
    enum MaterialType type;
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    float hasTexture;
    enum TextureType texture;
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
    float lensRadius;
    float focalDist;
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

// Hierarchical spatial datastructe: Octree
struct OctNode {
    int id;
    glm::vec3 maxCorner;
    glm::vec3 minCorner;
    int numGeoms;
    int geomStartIdx;

    // children ids
    int upFarLeft;
    int upFarRight;
    int upNearLeft;
    int upNearRight;
    int downFarLeft;
    int downFarRight;
    int downNearLeft;
    int downNearRight;
};

struct keep_path
{
    __host__ __device__
        bool operator()(const PathSegment path)
    {
        return path.remainingBounces > 0;
    }
};

struct material_sort {
    __host__ __device__
        bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2) {
        return i1.materialId < i2.materialId;
    }
};