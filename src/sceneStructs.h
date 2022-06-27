#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

typedef float Float;
typedef glm::vec3 vc3;
typedef glm::vec4 vc4;

enum GeomType {
    DELTA, // TODO tmp measure for delta light during direct light 
    SPHERE,
    CUBE,
    GLTF_MESH,
    TRIANGLE,
    BBOX,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    float time;
};

struct Geom {
    // geom index starting from 0
    int geom_idx;
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    //motion blur 
    glm::vec3 velocity;
    // for gltf model index
    int mesh_idx = -1;
};

struct Triangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;

    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;

    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;

    glm::vec3 norm;
};

struct GLTF_Model {
    // do not waste memory on all the same
    Geom self_geom;
    // to index from triangleslist
    int triangle_idx;
    int triangle_count;
};

//ref: https://github.com/mmerchante/CUDA-Path-tracer
struct TextureDescriptor
{
    int valid;
    int type; // 0 bitmap, 1 procedural TODO
    int index;
    int width;
    int height;
    glm::vec2 repeat;
    TextureDescriptor() : valid(-1), type(0), index(-1), width(0), height(0), repeat(glm::vec2(1.f)){};
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
    bool isSurface = true;

    TextureDescriptor diffuseTexture;
    TextureDescriptor specularTexture;
    TextureDescriptor normalTexture;
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

    float apertureRadius;
    float focusDist; // focalDistance
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
	glm::vec3 colorSum;
    glm::vec3 colorThroughput;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 pos;
  glm::vec3 surfaceNormal;
  int materialId;
  int geom_idx = -1;
  glm::vec2 uv;
};

enum BxDFType {
    BSDF_REFLECTION = 1 << 0,   // This BxDF handles rays that are reflected off surfaces
    BSDF_TRANSMISSION = 1 << 1, // This BxDF handles rays that are transmitted through surfaces
    BSDF_DIFFUSE = 1 << 2,      // This BxDF represents diffuse energy scattering, which is uniformly random
    BSDF_GLOSSY = 1 << 3,       // This BxDF represents glossy energy scattering, which is biased toward certain directions
    BSDF_SPECULAR = 1 << 4,     // This BxDF handles specular energy scattering, which has no element of randomness
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION
};
