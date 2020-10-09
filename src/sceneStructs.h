#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum class GeomType : unsigned char {
	INVALID,
	SPHERE,
	CUBE,
	TRIANGLE
};

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

struct GeomTransform {
	glm::mat4x3 transform;
	glm::mat4x3 inverseTransform;
};
struct GeomTriangle {
	glm::vec3 vertices[3];
	glm::vec3 normals[3];
	glm::vec2 uvs[3];
};
struct Geom {
	Geom() {
	}
	Geom(const Geom &src) {
		std::memcpy(this, &src, sizeof(Geom));
	}
	~Geom() {
	}

	union {
		GeomTransform implicit;
		GeomTriangle triangle;
	};
	int materialid = -1;
	GeomType type = GeomType::INVALID;
};

enum class MaterialType : unsigned char {
	invalid,
	emissive,
	diffuse,
	specularReflection,
	specularTransmission,
	disney
};

struct SpecularTransmissionMaterial {
	float indexOfRefraction;
};
struct DisneyMaterial {
	float
		roughness,
		metallic,
		specular,
		specularTint,
		sheen,
		sheenTint,
		clearcoat,
		clearcoatGloss;
};

struct Material {
	union {
		bool emitterNoMis;
		SpecularTransmissionMaterial specularTransmission;
		DisneyMaterial disney;
	};
	glm::vec3 baseColorLinear;
	MaterialType type;
};

struct Camera {
	glm::ivec2 resolution;
	glm::vec3 position;
	glm::vec3 lookAt;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec2 pixelLength;
	float fovy;
	float aperture = 0.3f, focalDistance = 20.0f;
};

struct RenderState {
	Camera camera;
	unsigned int iterations;
	int traceDepth;
	std::vector<glm::vec3> image;
	std::string imageName;
};

struct PathSegment {
	glm::vec3 colorThroughput, colorAccum;
	Ray ray;
	int pixelIndex;
	int lastGeom;
	int16_t remainingBounces;
	bool prevBounceNoMis;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
	glm::vec3 geometricNormal, shadingNormal;
	float t;
	int materialId;
};

struct AABBTreeNode {
	glm::vec3 leftAABBMin, leftAABBMax, rightAABBMin, rightAABBMax;
	int leftChild, rightChild;
};

struct IntersectionSample {
	glm::vec2 mis1, mis2, out;
};
struct CameraSample {
	glm::vec2 pixel, dof;
};
