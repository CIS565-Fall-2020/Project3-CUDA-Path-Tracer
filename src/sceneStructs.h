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

struct Geom {
    enum GeomType type;
    int materialid;
    int indices_num;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::vec3 bounding_box_down_corner;
    glm::vec3 bounding_box_upper_corner;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    float* dev_mesh_positions;
    float* dev_mesh_normals;
    unsigned char* dev_texture;
    float* dev_uvs;
    int texture_width;
    int texture_height;
    bool hasTexture = false;
    unsigned int* dev_mesh_indices;
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
    float focal_length = 5.f;
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
    int ori_id;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec2 uv;
  bool hasTexture;
  GeomType hit_type;
  glm::vec3 surfaceNormal;
  glm::vec3 tangent, bitangent;
  int materialId;
  int geomId;
  bool outside;
};

struct OctreeNode {
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
    int childern_indices[8];
    bool hasChildern = false;
};
