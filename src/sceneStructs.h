#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "utilities.h"

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
	int meshid = -1;  // id of the corresponding mesh
	int num_faces ;   // number of faces in this mesh = gltf::Mesh::faces.size() / 3

	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;
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

	float focalLength = 0;
    float lensRadius = 0;
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

struct GltfMesh
{

};

struct Test
{
	int* x;
	int num;
};

struct material_comp
{
	__host__ __device__ bool operator()(const ShadeableIntersection& isect1,
		const ShadeableIntersection& isect2)
	{
		return isect1.materialId > isect2.materialId;
	}
};

struct raytracing_continuing
{
	__host__ __device__ bool operator()(const PathSegment& segment)
	{
		return segment.remainingBounces > 0;
	}
};

__host__ __device__ inline void setGeomTransform(Geom* geom, const glm::mat4& trans)
{
	geom->transform = trans;
	geom->inverseTransform = glm::inverse(trans);
	geom->invTranspose = glm::inverseTranspose(trans);
}


__host__ __device__ inline glm::vec2 ConcentricSampleDisk(const glm::vec2& u)
{
	// Map uniform random numbers to [-1, 1]^2
	glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0)
		return glm::vec2(0);

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y))
	{
		r = uOffset.x;
		theta = PI_OVER_4 * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = PI_OVER_2 - PI_OVER_4 * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cos(theta), std::sin(theta));
}