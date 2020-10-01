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
glm::vec3 squareToDiskConcentric(glm::vec2 sample) {
    float phi, r, u, v;
    float a = 2.f * sample[0] - 1;
    float b = 2.f * sample[1] - 1;

    if (a > -b) {
        if (a > b) {
            r = a;
            phi = (PI / 4.f) * (b / a);
        }
        else {
            r = b;
            phi = (PI / 4.f) * (2.f - (a / b));
        }
    }
    else {
        if (a < b) {
            r = -a;
            phi = (PI / 4.f) * (4.f + (b / a));
        }
        else {
            r = -b;
            if (b != 0) {
                phi = (PI / 4.f) * (6.f - (a / b));
            }
            else {
                phi = 0.f;
            }
        }
    }
    u = r * glm::cos(phi);
    v = r * glm::sin(phi);
    return glm::vec3(u, v, 0.f);
}

__host__ __device__ 
glm::vec3 squareToHemisphereCosine(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 sampleDisk = squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
    sampleDisk[2] = glm::sqrt(glm::max(1.f - sampleDisk[0] * sampleDisk[0] - sampleDisk[1] * sampleDisk[1], 0.f));
    return sampleDisk;
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
    if (m.hasReflective > 0.f) {
        // Update "color" parameter in place
        pathSegment.color *= m.color;
        // Update "ray" parameter in place
        pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        pathSegment.ray.origin = intersect;
    }
    else if (m.hasRefractive > 0.f) {

    }
    else {
            // Calculate new ray direction
        /*
        glm::vec3 newRayDir = squareToHemisphereCosine(rng);
        if (-pathSegment.ray.direction.z < 0.f) {
            newRayDir.z *= -1.f;
        }

        // Calculate the pdf of the bsdf
        float pdf;
        if (-pathSegment.ray.direction.z * newRayDir.z > 0.f) {  // means both the old ray and the new bounce are within the same hemisphere
            pdf = newRayDir.z / PI;
        }
        else {
            pdf = 0.f;
        }

        // Update color parameter in place
        if (pdf != 0.f) {
            pathSegment.color *= m.color / PI * glm::abs(glm::dot(newRayDir, normal)) / pdf;
        }
        else {
            pathSegment.color = glm::vec3(0.f);
        }

        // pathSegment.color = (normal + glm::vec3(1.f)) / 2.f;

        // Update "ray" parameter in place
        pathSegment.ray.direction = newRayDir;
        pathSegment.ray.origin = intersect;
        */

        // Update "color" parameter in place
        pathSegment.color *= m.color;
        // Update "ray" parameter in place
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.ray.origin = intersect;
    }
    
}
