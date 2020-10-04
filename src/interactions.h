#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal, thrust::default_random_engine &rng
) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	} else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	} else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 = glm::cross(normal, perpendicularDirection1);

	return
		up * normal + over * (
			cos(around) * perpendicularDirection1 +
			sin(around) * perpendicularDirection2
		);

}

__host__ __device__ glm::vec2 sampleUnitDiskUniform(thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> angleDist(0.0f, 2.0f * glm::pi<float>()), radiusDist(0.0f, 1.0f);
	float angle = angleDist(rng), radius = glm::sqrt(radiusDist(rng));
	return glm::vec2(glm::cos(angle), glm::sin(angle)) * radius;
}

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
__host__ __device__ void scatterRay(
	PathSegment &path, glm::vec3 intersect, glm::vec3 normal, Material m, thrust::default_random_engine &rng
) {
	path.ray.origin = intersect;
	path.color *= m.color;
	if (m.emittance > 0.0f) { // terminate path
		path.color *= m.emittance;
		path.remainingBounces = -1;
	} else {
		--path.remainingBounces;
		if (m.hasReflective) {
			path.ray.direction = glm::reflect(path.ray.direction, normal);
		} else if (m.hasRefractive) {
			float ior = m.indexOfRefraction, cosOut = -glm::dot(normal, path.ray.direction);
			if (cosOut < 0.0f) {
				normal = -normal;
				cosOut = -cosOut;
			} else {
				ior = 1.0f / ior;
			}

			float fresnel = 1.0f;
			float sinOut = glm::sqrt(1.0f - cosOut * cosOut);
			float sinIn = sinOut * ior;
			if (sinIn <= 1.0f) {
				float cosIn = glm::sqrt(1.0f - sinIn * sinIn);
				float rParl = (cosOut - ior * cosIn) / (cosOut + ior * cosIn);
				float rPerp = (ior * cosOut - cosIn) / (ior * cosOut + cosIn);
				fresnel = 0.5f * (rPerp * rPerp + rParl * rParl);
			}

			thrust::uniform_real_distribution<float> fresnelDist(0.0f, 1.0f);
			if (fresnelDist(rng) < fresnel) { // reflect
				path.ray.direction = glm::reflect(path.ray.direction, normal);
			} else { // refract
				path.ray.direction = glm::normalize(glm::refract(path.ray.direction, normal, ior));
			}
		} else {
			if (glm::dot(normal, path.ray.direction) >= 0.0f) {
				normal = -normal;
			}
			path.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		}
		path.ray.origin += 0.01f * path.ray.direction;
	}
}
