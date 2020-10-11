#pragma once

#include "intersections.h"

#define ENHANCEDSAMPLING 1

	// CHECKITOUT
	/**
	 * Computes a cosine-weighted random direction in a hemisphere.
	 * Used for diffuse lighting.
	 */
	__host__ __device__
	glm::vec3 calculateRandomDirectionInHemisphere(
		glm::vec3 normal, thrust::default_random_engine & rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	
	float jitterX = u01(rng);
	float jitterY = u01(rng);

#ifdef ENHANCEDSAMPLING
	float grid1DCount = 10.f; // square root of the total number of grids
	float gridSize = 1.f / (float)grid1DCount;
	// Choose grid then jitter within grid
	jitterX = (float)((int)(u01(rng) * grid1DCount)) / grid1DCount + u01(rng) * gridSize;
	jitterY = (float)((int)(u01(rng) * grid1DCount)) / grid1DCount + u01(rng) * gridSize;
#endif

	float up = sqrt(jitterX); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = jitterY * TWO_PI;

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}
	// ----------------------------------------------------------------------------------------------
	// ---------------------------- MAIN ------------------------------------------------------------
	// ----------------------------------------------------------------------------------------------

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
	PathSegment& pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material& m,
	thrust::default_random_engine& rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.
	
	if (m.hasRefractive == 1.f) {
		// figure out which ior is incident or transmitted
		float cosTheta = glm::dot(normal, pathSegment.ray.direction);
		bool entering = cosTheta > 0;
		glm::vec3 correctedNormal = entering ? -normal : normal;
		float iorI = entering ? m.ior0 : m.ior1;
		float iorT = entering ? m.ior1 : m.ior0;
		float iorRatio = iorI / iorT;

		// Compute ray direction for specular transmission
		glm::vec3 rayDir = entering ?
			glm::refract(pathSegment.ray.direction, normal, iorRatio) :
			glm::refract(pathSegment.ray.direction, -normal, iorRatio);

		float threshold = glm::asin(iorRatio);
		float theta = acos(glm::dot(pathSegment.ray.direction, correctedNormal));

		if (theta < threshold) { // total internal reflection
			pathSegment.color = glm::vec3(0.f);
			pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, correctedNormal);
			pathSegment.ray.origin = intersect + 0.01f * correctedNormal;
			return;
		}

		cosTheta = glm::dot(correctedNormal, pathSegment.ray.direction);
		float r0sqrt = (iorI - iorT) / (iorI + iorT);
		float r0 = r0sqrt * r0sqrt;
		float oneMinusCosTheta = 1.f - cosTheta;
		float pow5 = (oneMinusCosTheta * oneMinusCosTheta) * (oneMinusCosTheta * oneMinusCosTheta) * oneMinusCosTheta;
		float rTheta = r0 + (1 - r0) * pow5;

		thrust::uniform_real_distribution<float> u01(0, 1);
		float epsilon = 0.01f;
		if (m.hasReflective == 1.f && 1.5f * u01(rng) > rTheta) {
			pathSegment.color *= m.specular.color;
			pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, correctedNormal);
			pathSegment.ray.origin = intersect + epsilon * correctedNormal;
			return;
		}
		else {
			pathSegment.ray.direction =
				glm::normalize(glm::refract(pathSegment.ray.direction, correctedNormal, iorRatio));
			pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
			pathSegment.color *= m.specular.color;
			return;
		}
	}
	pathSegment.color *= m.color;
	pathSegment.ray.origin = intersect + 0.0001f * normal;
	if (m.hasRefractive == 0.f && m.hasReflective == 0.f) {
		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	}
	else if (m.hasRefractive == 0.f) {
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
	}
}
