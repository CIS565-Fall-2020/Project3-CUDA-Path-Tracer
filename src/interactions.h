#pragma once

#include "intersections.h"

#define STRATIFIED_SAMPLEING 0

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

#if STRATIFIED_SAMPLEING == 1

    int numSamples = 100;
    // The square root of the number of samples 
    int sqrtVal = (int)(std::sqrt((float)numSamples) + 0.5);
    // A number useful for scaling a square of size sqrtVal x sqrtVal to 1 x 1
    float invSqrtVal = 1.f / sqrtVal;
    
    // Ensure that the number of samples we use fits evenly within a square grid
    numSamples = sqrtVal * sqrtVal;

    int y = (int)(u01(rng) * sqrtVal);
    int x = (int)(u01(rng) * numSamples) % sqrtVal;

    glm::vec2 sample = glm::vec2((float)x + u01(rng), (float)y + u01(rng)) * invSqrtVal;

	up = sqrt(sample.x); // cos(theta)
	over = sqrt(1 - up * up); // sin(theta)
	around = sample.y * TWO_PI;
    
#endif //STRATIFIED_SAMPLEING

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
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
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
 * - This way is inefficient, but serves as a good starting point - it
 *   converges slowly, especially for pure-diffuse or pure-specular.
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

    glm::vec3 newDir = glm::vec3(0.f, 0.f, 0.f);

    if (m.hasReflective == 0.f && m.hasRefractive == 0.f)  // ideal diffuse 
    {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
		float pdf = glm::abs(glm::dot(normal, newDir)) / PI;
		glm::vec3 f = m.color / PI;
		if (pdf == 0.f)
		{
			pathSegment.remainingBounces = -1;
			pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
		}
		else
		{
			pathSegment.color *= (f * glm::abs(glm::dot(normal, newDir)) / pdf);
            pathSegment.remainingBounces--;
		}
    }
    else if (m.hasReflective == 1.f && m.hasRefractive == 0.f) // perfectly specular-reflective
    {
        newDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
        pathSegment.remainingBounces--;
    }
    else if (m.hasReflective == 1.f && m.hasRefractive == 1.f) // Refraction
    {
        // entering: true- from low eta to high eta
        bool entering = glm::dot(normal, pathSegment.ray.direction) < 0;
        float eta = entering ? 1.f / m.indexOfRefraction : m.indexOfRefraction;

        glm::vec3 unitDir = glm::normalize(pathSegment.ray.direction);
        unitDir = entering ? -1.f * unitDir : unitDir;
        float cosThetaI = glm::dot(unitDir, normal);
        float sinThetaI = glm::sqrt(1.f - cosThetaI * cosThetaI);

        bool cannotRefract = eta * sinThetaI > 1.f;

        // Schlick's approximation for reflectance
        float r0 = (1 - eta) / (1 + eta);
        r0 = r0 * r0;
        float reflectance = r0 + (1 - r0) * glm::pow((1 - cosThetaI), 5);

        thrust::uniform_real_distribution<float> u01(0, 1);

        if (cannotRefract || reflectance > u01(rng))   // reflection  
        {
			newDir = glm::reflect(pathSegment.ray.direction, normal);
			pathSegment.color *= m.specular.color;
			pathSegment.remainingBounces--;
        }
        else   // refraction
        {
            newDir = glm::refract(pathSegment.ray.direction, normal, eta);
            pathSegment.color *= m.specular.color;
            pathSegment.remainingBounces--;
        }      
    }

	pathSegment.ray.direction = newDir;
	pathSegment.ray.origin = intersect + newDir * 0.001f;  // to make the intersection point outside of the primitive
}
