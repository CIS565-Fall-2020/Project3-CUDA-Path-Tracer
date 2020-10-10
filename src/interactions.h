#pragma once

#include "intersections.h"
#define SCHLICK 0

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
float reflectance(float cosine, float ref_idx) {
      // Use Schlick's approximation for reflectance.
      auto r0 = (1.f - ref_idx) / (1.f + ref_idx);
      r0 = r0 * r0;
      return r0 + (1.f - r0) * pow(glm::abs(1.f - cosine), 5.f);
}

// Algorithm from PBRT book
__host__ __device__
glm::vec2 ConcentricSampleDisk(const glm::vec2& u) {
      float PiOver2 = 1.57079632679489661923;
      float PiOver4 = 0.78539816339744830961;

      // Map uniform random numbers to (-1, 1)
      glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

      // Handle degeneracy at the origin
      if (uOffset.x == 0 && uOffset.y == 0)
            return glm::vec2(0, 0);

      // Apply concentric mapping to point
      float theta, r;
      if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
            r = uOffset.x;
            theta = PiOver4 * (uOffset.y / uOffset.x);
      } else {
            r = uOffset.y;
            theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
      }
      return r * glm::vec2(std::cos(theta), std::sin(theta));
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
      float prob = u01(rng);
      glm::vec3 rayDir = glm::normalize(pathSegment.ray.direction);
      normal = glm::normalize(normal);

      if (prob < m.hasRefractive) { //  with references to raytracing-in-one-weekend
            float eta = m.indexOfRefraction;
            bool outside = false;
            glm::vec3 refractNorm = -1.f * normal;
            if (glm::dot(rayDir, normal) < 0) {
                  eta = 1.f / eta;
                  outside = true;
                  refractNorm *= -1.f;
            }
            // check for total internal reflection
            float cos_theta = glm::min(glm::dot(-1.f * rayDir, normal), 1.f);
            float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            bool cannot_refract = eta * sin_theta > 1.0;

            pathSegment.ray.origin = intersect + 0.001f * rayDir; // toggle between 0.5 and 0.001
            if (cannot_refract) {
                  pathSegment.ray.direction = glm::normalize(glm::reflect(rayDir, normal));
            } else {
                  pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, refractNorm, eta));
            }
#if SCHLICK
            if (reflectance(cos_theta, eta) > u01(rng)) {
                  pathSegment.ray.direction = glm::normalize(glm::reflect(rayDir, normal));
            }
#endif
            pathSegment.color *= m.specular.color;
      } 
      else if (prob < (m.hasRefractive + m.hasReflective)) {
             // specular reflection
             pathSegment.ray.origin = intersect + 0.001f * normal;
             pathSegment.color *= m.specular.color;
             pathSegment.ray.direction = glm::normalize(glm::reflect(rayDir, normal)); // need to change to sampling based later
            
      } 
      else {
            // ideal diffuse
            pathSegment.ray.origin = intersect + 0.001f * normal;
            pathSegment.color *= m.color;
            pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
      }
      

}
