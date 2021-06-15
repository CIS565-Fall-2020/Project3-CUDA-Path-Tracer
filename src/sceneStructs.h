#pragma once

#include <string>
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
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
    glm::vec3 geomMinCorner, geomMaxCorner;
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
    float lensRadius; 
    float focalDistance; 
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

//struct Triangle {
//    std::array<glm::vec3, 3> vertices{};
//    std::array<glm::vec3, 3> normals{};
//    std::array<glm::vec2, 3> uvs{};
//};

struct Triangle {
    glm::vec3 vert1, vert2, vert3;
    glm::vec3 norm1, norm2, norm3; 
};

struct Mesh {
    std::vector<Triangle> triangles; 
    int num_triangles = 0;
    std::string filename; 
    int start_triangle, end_triangle; 
    glm::vec3 minCorner, maxCorner; 
};

//struct OctTreeNode
//{
//    int children[8]; 
//    glm::vec3 minCorner;
//    glm::vec3 maxCorner;
//    int startID;
//    int endID; 
//};
//
//struct OctTree
//{
//    OctTreeNode* root; 
//};