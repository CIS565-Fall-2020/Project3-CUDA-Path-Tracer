#pragma once

#include "intersections.h"
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define SOBOL_SAMPLE 0

// Evaluate fresnel coefficient for reflection and refraction
__host__ __device__ float fresnelEvaluate(
    float etaI, float etaT,
    float cosThetaI) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    float eI = etaI;
    float eT = etaT;

    if (cosThetaI <= 0.f) { // Leaving the medium
        eI = etaT;
        eT = etaI;
        cosThetaI = -cosThetaI;
    }

    float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = eI / eT * sinThetaI;

    if (sinThetaT >= 1.f) {
        return 1.f;
    }

    float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));

    float parl = ((eT * cosThetaI) - (eI * cosThetaT)) /
        ((eT * cosThetaI) + (eI * cosThetaT));
    float perp = ((eI * cosThetaI) - (eT * cosThetaT)) /
        ((eI * cosThetaI) + (eT * cosThetaT));

    return (parl * parl + perp * perp) / 2.f;
}

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

__device__ __host__ glm::vec3 sampleSobolHemishpere(
    glm::vec3 normal, int ith) {
    
    ith = ith % 1024;
    int vec[10][2] = {
        0x8680u, 0x4c80u, 0xf240u, 0x9240u, 0x8220u, 0x0e20u,
        0x4110u, 0x1610u, 0xa608u, 0x7608u, 0x8a02u, 0x280au,
        0xe204u, 0x9e04u, 0xa400u, 0x4682u, 0xe300u, 0xa74du,
        0xb700u, 0x9817u
    };
    glm::ivec2 sample(0);
    for (int k = 0; ith > 0; ith >>= 1, k++) {
        sample[0] ^= (ith & 1) ? vec[k][0] : 0;
        sample[1] ^= (ith & 1) ? vec[k][1] : 0;
    }

    float inverseRange = 1.0 / 0x10000;
    float up = sqrt((float)sample.x * inverseRange);
    float over = sqrt(1 - up * up);
    float around = (float)sample.y * inverseRange * TWO_PI;

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
        thrust::default_random_engine &rng,
        int ith) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    
    glm::vec3 newDir;
    float pdf;
    glm::vec3 f;
    bool entering = glm::dot(normal, pathSegment.ray.direction) < 0;

    if (m.hasReflective == 1 && m.hasRefractive == 1) { // Refraction
        float eta = entering ? 1.f / m.indexOfRefraction : m.indexOfRefraction;
        float cosThetaI = glm::dot(normal, pathSegment.ray.direction);
        if (eta * eta * (1.f - cosThetaI * cosThetaI) > 1.f) { // Total internel reflection
            newDir = glm::reflect(pathSegment.ray.direction, normal);
            f = m.specular.color;
            pdf = 1.f;
            float fresnel = 1.f;
            pathSegment.color *= f * fresnel / pdf;
            pathSegment.remainingBounces--;
            pathSegment.ray.origin = intersect;
        }
        else {
            newDir = glm::refract(pathSegment.ray.direction, normal, eta);
            float cosThetaT = glm::dot(newDir, normal);
            float fresnel = fresnelEvaluate(1.f, m.indexOfRefraction, cosThetaT);
            thrust::uniform_real_distribution<float> u01(0, 1);
            if (u01(rng) > fresnel) { // Refract
                f = m.color;
                pdf = 1.f - fresnel;
                pathSegment.color *= f * (1.f - fresnel) / pdf;
                pathSegment.remainingBounces--;
                pathSegment.ray.origin = intersect +
                    glm::normalize(pathSegment.ray.direction) * 0.0002f;
            }
            else { // Reflect
                newDir = glm::reflect(pathSegment.ray.direction, normal);
                f = m.specular.color;
                pdf = fresnel;
                pathSegment.color *= f * fresnel / pdf;
                pathSegment.remainingBounces--;
                pathSegment.ray.origin = intersect;
            }
        }
        pathSegment.ray.origin = intersect + glm::normalize(pathSegment.ray.direction) * 0.0002f;
    }
    else if (m.hasReflective == 1) { // Reflection
        newDir = glm::reflect(pathSegment.ray.direction, normal);
        pdf = 1.f;
        f = m.specular.color;
        float fresnel = 1.f; // Perfect Mirror
        pathSegment.color *= f * fresnel / pdf;
        pathSegment.remainingBounces--;
        pathSegment.ray.origin = intersect;
    }
    else { // Diffuse
#if SOBOL_SAMPLE == 1
        newDir = sampleSobolHemishpere(normal, ith);
#else
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
#endif
        float absCosTheta = glm::abs(glm::dot(normal, newDir));
        pdf = INV_PI * absCosTheta;
        f = m.color * INV_PI;
        if (pdf == 0.f) {
            pathSegment.color = glm::vec3(0.f);
            pathSegment.remainingBounces = 0;
        }
        else {
            pathSegment.color *= f * absCosTheta / pdf;
            pathSegment.remainingBounces--;
        }
        pathSegment.ray.origin = intersect;
    }
    
    pathSegment.ray.direction = newDir;
}
