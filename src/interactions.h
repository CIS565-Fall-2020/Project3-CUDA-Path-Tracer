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

__host__ __device__ 
glm::vec3 calculateRefractionDirection(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m) {
    
    // assume we are outside of the sphere
    float normal_dot = glm::dot(pathSegment.ray.direction, normal);
    glm::vec3 n = normal;
    float factor = 1.0f / m.indexOfRefraction;

    // are we inside of the sphere?
    if (normal_dot > 0.f) {
        n *= -1.0f;
        factor = m.indexOfRefraction;
    }
    glm::vec3 ray_direction = glm::refract(pathSegment.ray.direction, n, factor);

    // critical angle
    if (ray_direction == glm::vec3(0.0f)) {
        ray_direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= 0.0f;
    }
    
    pathSegment.color *= m.specular.color;
    pathSegment.ray.direction = ray_direction;
    pathSegment.ray.origin = intersect + (.001f) * pathSegment.ray.direction;

    return ray_direction;
}

// based on: http://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission.html#FrDielectric
__host__ __device__ 
float fresnel_dielectric(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // check if we are entering or exiting the sphere
    bool entering = cosThetaI > 0.f;
    float etaICopy = etaI;
    float etaTCopy = etaT;
    if (!entering) {
        etaICopy = etaT;
        etaTCopy = etaI;
        cosThetaI = glm::abs(cosThetaI);
    }

    // computer cosThetaT using Snell's Law
    float sinThetaI = glm::sqrt(glm::max(0.0f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaICopy / etaTCopy * sinThetaI;

    // handle total internal refection
    float cosThetaT = glm::sqrt(glm::max(0.0f, 1 - sinThetaT * sinThetaT));

    float Rparl = ((etaTCopy * cosThetaI) - (etaICopy * cosThetaT)) /
        ((etaTCopy * cosThetaI) + (etaICopy * cosThetaT));
    float Rperp = ((etaICopy * cosThetaI) - (etaTCopy * cosThetaT)) /
        ((etaICopy * cosThetaI) + (etaTCopy * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
}

__host__ __device__
void calculateFresnelDirection(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, thrust::default_random_engine& rng) {
    float normal_dot = glm::dot(-pathSegment.ray.direction, normal);
    float etaI = m.indexOfRefraction;
    float etaT = 1.f; 
    if (normal_dot > 0.f) {
        etaI = 1.f;
        etaT = m.indexOfRefraction;
    }
    float fresnel_factor = fresnel_dielectric(normal_dot, etaI, etaT) / glm::abs(normal_dot);

    thrust::uniform_real_distribution<float> u01(0, 1);
    if (u01(rng) > fresnel_factor) {
        // refraction
        calculateRefractionDirection(pathSegment, intersect, normal, m);
        return;
    }

    // reflection
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.color *= m.specular.color;
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.0001f;
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
		PathSegment& pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

    // TO DO: Need to figure out specular power later

    // change the ray's origin to be at the intersection point
    // make sure to shift it along the normal a bit to avoid intersecting itself (floating point error)
    pathSegment.ray.origin = intersect + normal * 0.001f;

    if (m.hasReflective && m.hasRefractive) {
        // fresnel material
        calculateFresnelDirection(pathSegment, intersect, normal, m, rng);
    }
    else if (m.hasReflective) {
        // reflective
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.0001f;
    }
    else if (m.hasRefractive) {
        // refraction
        calculateRefractionDirection(pathSegment, intersect, normal, m);
    }
    else {
        // default is diffuse
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.0001f;
    }

}
