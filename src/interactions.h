#pragma once

#include "intersections.h"
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

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

    int BxDFs[3] = { 0, 0, 0 }; // [0]: Diffuse [1]: Reflection [2]: Refraction
    int matCt = 0;

    // debug
    int deb = pathSegment.pixelIndex;
    
    if (m.hasReflective) {
        BxDFs[1] = 1;
        matCt++;
    }
    if (m.hasRefractive) {
        BxDFs[2] = 1;
        matCt++;
    }
    if (matCt == 0) {
        BxDFs[0] = 1;
        matCt++;
    }
    
    thrust::uniform_int_distribution<int> uMat(0, matCt - 1);
    int comp = uMat(rng);
    int bxdf = 0;
    while (bxdf < 3) {
        if (BxDFs[bxdf] == 1 && (comp-- == 0)) {
            break;
        }
        bxdf++;
    }
    
    glm::vec3 newDir;
    float pdf;
    glm::vec3 f;

    if (bxdf == 0) { // Diffuse
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        float z = glm::abs(glm::dot(normal, newDir));
        pdf = INV_PI * z / matCt;
        f = m.color * INV_PI;
        if (pdf == 0.f) {
            pathSegment.color = glm::vec3(0.f);
            pathSegment.remainingBounces = 0;
        }
        else {
            pathSegment.color *= f * z / pdf;
            pathSegment.remainingBounces--;
        }
        pathSegment.ray.origin = intersect;
    }
    else if (bxdf == 1) { // Reflection
        newDir = glm::reflect(pathSegment.ray.direction, normal);
        float z = glm::dot(normal, newDir);
        pdf = 1.f / matCt;
        f = m.specular.color;

        float fresnel;
        if (m.indexOfRefraction == 0) {
            fresnel = 1.f;
        }
        else {
            fresnel = fresnelEvaluate(1.f, m.indexOfRefraction, z);
        }

        pathSegment.color *= f * fresnel / pdf; // Fresnel take care of lambert
        pathSegment.remainingBounces--;
        pathSegment.ray.origin = intersect;
    }
    else { // Refraction
        float z = -glm::dot(normal, pathSegment.ray.direction);
        float eta = z > 0 ? 1.f / m.indexOfRefraction : m.indexOfRefraction;
        
        newDir = glm::refract(pathSegment.ray.direction, normal, eta);
        if (isnan(newDir.x) || isnan(newDir.y) || isnan(newDir.z)) {
            pathSegment.color = glm::vec3(0.f);
            pathSegment.remainingBounces = 0;
        }
        else {
            float cosT = glm::dot(newDir, normal);
            float fresnel = 1.f - fresnelEvaluate(1.f, m.indexOfRefraction, cosT);
            pdf = 1.f / matCt;
            f = m.color;
            pathSegment.color *= f * fresnel / pdf;
            pathSegment.remainingBounces--;
        }
        pathSegment.ray.origin = intersect + glm::normalize(pathSegment.ray.direction) * 0.0002f;
    }
    
    pathSegment.ray.direction = newDir;
}
