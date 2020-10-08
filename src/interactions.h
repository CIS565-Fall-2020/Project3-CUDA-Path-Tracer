#pragma once

#include "intersections.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include<iostream>
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

__host__ __device__ void reflect(PathSegment& pathSegment, 
    glm::vec3 intersection, glm::vec3 normal, const Material& material) {
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.ray.origin = intersection + 0.01f * normal;
    pathSegment.color *= material.specular.color;
}

__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersection,
        glm::vec3 normal,
        const Material &material,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    //pathSegment.color *= material.color;
    // reflective
    pathSegment.remainingBounces--;
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand_num = u01(rng);
    if (material.hasRefractive == 1.f) {
        glm::vec3 dir(0.f);
        glm::vec3 real_norm(0.f);
        float ior_ratio = 0.f;
        if (glm::dot(pathSegment.ray.direction, normal) > 0.f) {
            real_norm = -1.f * normal;
            ior_ratio = material.indexOfRefraction;
        }
        else {
            real_norm = normal;
            ior_ratio = 1.f / material.indexOfRefraction;
        }

        dir = glm::refract(pathSegment.ray.direction, real_norm, ior_ratio);
        float threshold = asin(ior_ratio);
        float theta = acos(glm::dot(pathSegment.ray.direction, real_norm));
        
        if (theta < threshold) {
            // total internal reflection
            reflect(pathSegment, intersection, real_norm, material);
            pathSegment.color = glm::vec3(0.f);
            return;
        }
        
        float R0 = powf((1.f - ior_ratio)/(1.f + ior_ratio), 2.f);
        float R_theta = R0 + (1.f - R0) * powf(1.f - fmax(0.f, glm::dot(pathSegment.ray.direction, real_norm)), 5.f);
        
        if (material.hasReflective == 1.f && rand_num > R_theta / 1.5f) {
            // reflection
            reflect(pathSegment, intersection, real_norm, material);
            return;
        }
        else {
            // refraction
            pathSegment.ray.direction = glm::normalize(dir);
            pathSegment.ray.origin = intersection + dir * 0.01f;
            pathSegment.color *= material.specular.color;
            return;
        }
    } else if (material.hasReflective == 1.f) {
        reflect(pathSegment, intersection, normal, material);
        return;
    }
    else {
        glm::vec3 bounce_dir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.direction = bounce_dir;
        pathSegment.ray.origin = intersection + 0.01f * normal;
        pathSegment.color *= material.color;
        
        return;
    }
}
