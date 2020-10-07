#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal, thrust::default_random_engine &rng) {
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

// Reference: https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/refraction
__host__ __device__
float schlickApproximation(float cosTheta, float IOR)
{
	float r0 = (1 - IOR) / (1 + IOR);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow(1.0f - abs(cosTheta), 5.0f);
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
__host__ __device__
void scatterRay(
	PathSegment & pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	float xi = u01(rng);

	glm::vec3 wo = pathSegment.ray.direction;
	glm::vec3 wi;
	float pdf = 1.0;

	if (xi < m.hasReflective)	// Specular
	{
		pdf = m.hasReflective;
		wi = glm::normalize(glm::reflect(wo, normal));
		pathSegment.color *= m.specular.color / pdf;
	}
	else if (xi < m.hasReflective + m.hasRefractive)	// Refraction
	{
		pdf = m.hasRefractive;
		bool entering = glm::dot(wo, normal) < 0;
		float etaI = entering ? 1 : m.indexOfRefraction;
		float etaT = entering ? m.indexOfRefraction : 1;
		float cosTheta = glm::min(glm::dot(-wo, normal), 1.0f);
		float sinTheta = glm::sqrt(1.0f - cosTheta);

		float R = schlickApproximation(cosTheta, m.indexOfRefraction);
		float sample = u01(rng);
		if (m.indexOfRefraction * sinTheta > 1.0f || R > sample)	// Cannot refract
		{
			wi = glm::normalize(glm::reflect(wo, normal));
			pathSegment.color *= m.specular.color / pdf;
		}
		else
		{
			wi = glm::normalize(glm::refract(wo, normal, etaI / etaT));
			pathSegment.color *= m.color / pdf;
			normal = -normal;
		}
	}
	else    // Diffuse
	{
		//float absCosTheta = glm::abs(glm::dot(normal, wi));
		//float pdf = absCosTheta * INV_PI;
		//glm::vec3 f = m.color * INV_PI;
		//pathSegment.color *= (f * absCosTheta / pdf) / (1 - m.hasReflective);
		pdf = 1 - m.hasReflective - m.hasRefractive;
		wi = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= m.color / pdf;
	}
	pathSegment.ray.direction = wi;
	pathSegment.ray.origin = intersect + normal * 0.001f;
}
