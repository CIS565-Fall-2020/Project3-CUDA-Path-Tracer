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

__host__ __device__ float fresnel(
    double cosThetaI, double ref_idx
    )
{
    float etaI_ = 1;
    float etaT_ = ref_idx;
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float temp = etaI_;
        etaI_ = etaT_;
        etaT_ = temp;
        cosThetaI = abs(cosThetaI);
    }
    // Compute cosThetaT using Snell’s law
    float sinThetaI =sqrt(max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI_ / etaT_ * sinThetaI;
    // Handle total internal reflection
    if (sinThetaT >= 1) {
        return 1;
    }
    float cosThetaT = sqrt(max(0.f, 1 - sinThetaT * sinThetaT));

    float Rparl = ((etaT_ * cosThetaI) - (etaI_ * cosThetaT)) /
        ((etaT_ * cosThetaI) + (etaI_ * cosThetaT));
    float Rperp = ((etaI_ * cosThetaI) - (etaT_ * cosThetaT)) /
        ((etaI_ * cosThetaI) + (etaT_ * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__host__ __device__ double schlick(double cosine, double ref_idx) {
    double r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
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
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    thrust::uniform_real_distribution<float> u01(0, 1);
    float p0 = u01(rng);

    // specular
    if (p0 <= m.hasReflective) {
        if (m.specular.exponent > 0.0f) {
            float p1 = u01(rng);
            float p2 = u01(rng);
            pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal) + m.specular.exponent * glm::vec3(p0, p1, p2));
        }
        else {
            pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        }
        float scale = m.hasReflective <= 0.0 ? 0.0 : 1.0 / m.hasReflective;
        pathSegment.color *= m.color * scale;
        pathSegment.ray.origin = intersect + EPSILON * normal;
    }
    // specular
    else if (p0 <= m.hasReflective + m.hasRefractive) {
        float costheta = glm::dot(-pathSegment.ray.direction, normal);
        costheta = costheta > 1.0 ? 1.0 : costheta;
        costheta = costheta < -1.0 ? -1.0 : costheta;
        // if the ray is entering the surface
        float eta = costheta > 0 ? (1.0 / m.indexOfRefraction) : m.indexOfRefraction;
        glm::vec3 _normal = costheta > 0 ? normal : -normal;
        float scale = m.hasRefractive <= 0.0 ? 0.0 : 1.0 / m.hasRefractive;

        glm::vec3 refract = glm::refract(glm::normalize(pathSegment.ray.direction), glm::normalize(_normal), eta);
        glm::vec3 reflect = glm::reflect(pathSegment.ray.direction, _normal);
        if (glm::length(refract) < EPSILON) {
            pathSegment.ray.direction = glm::normalize(reflect);
            pathSegment.color *= m.specular.color * scale;
            pathSegment.ray.origin = intersect + (pathSegment.ray.direction * 0.001f);
            return;
        }
        float f = fresnel(costheta, m.indexOfRefraction);
        // reflect or refract?
        if (u01(rng) < f) {
            pathSegment.ray.direction = glm::normalize(reflect);
            pathSegment.color *= m.specular.color * scale;
        }
        else {
            pathSegment.ray.direction = glm::normalize(refract);
            pathSegment.color *= m.color * scale;
        }

        pathSegment.ray.origin = intersect + (pathSegment.ray.direction * 0.001f);
    }
    // diffuse
    else {
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        float scale = m.hasReflective >= 1.0 ? 0.0 : 1.0 / (1.0 - m.hasReflective);
        pathSegment.color *= m.color * scale;
        pathSegment.ray.origin = intersect + EPSILON * normal;
    }
}
