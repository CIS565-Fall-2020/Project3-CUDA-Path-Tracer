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
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
	glm::vec3 vertices[3];
	glm::vec3 normal;
	int materialId;
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

	// triangle index of a mesh
	int tIndexStart;
	int tIndexEnd;
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

// Octree
struct OctreeNode {
	std::vector<int> triangleIdx;
	bool hasTriangle = false;
	float xmin, xmax, ymin, ymax, zmin, zmax;
	OctreeNode *tlf, *tlb, *trf, *trb, *blf, *blb, *brf, *brb; // top/bottom, left/right, front/back

	OctreeNode(
		float xmin, float xmax,
		float ymin, float ymax,
		float zmin, float zmax)
		:xmin(xmin), xmax(xmax),
		ymin(ymin), ymax(ymax),
		zmin(zmin), zmax(zmax),
		tlf(NULL), tlb(NULL),
		trf(NULL), trb(NULL),
		blf(NULL), blb(NULL),
		brf(NULL), brb(NULL) {
	}
};
struct OctreeNode_cuda {
	int triangleStart = -1;
	int triangleEnd = -1;
	float xmin, xmax, ymin, ymax, zmin, zmax;
	int tlf, tlb, trf, trb, blf, blb, brf, brb; // top/bottom, left/right, front/back

	OctreeNode_cuda(
		float xmin, float xmax,
		float ymin, float ymax,
		float zmin, float zmax)

		:xmin(xmin), xmax(xmax),
		ymin(ymin), ymax(ymax),
		zmin(zmin), zmax(zmax),
		tlf(tlf), tlb(tlb),
		trf(trf), trb(trb),
		blf(blf), blb(blb),
		brf(brf), brb(brb) {
	}
};

void constructOctree(OctreeNode *&root,
	int maxdepth,
	float xmin, float xmax,
	float ymin, float ymax,
	float zmin, float zmax);

void traverseOctree(OctreeNode *&root, Triangle &t, int Idx);

int traverseOctreeToArray(OctreeNode *root
	, std::vector<int> &sortTriangles
	, std::vector<OctreeNode_cuda> &octreeVector);