#pragma once

#include "intersections.h"

#define STRATIFIED 0

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

#if STRATIFIED
    //int nSamples = 25;
    glm::vec2 sample[256];
    int nx = 16, ny = 16;
    float dx = (float)1 / (int)nx, dy = (float)1 / (int)ny;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            float jx = u01(rng);
            float jy = u01(rng);
            sample[y * ny + x].x = glm::min((x + jx) * dx, 1 - EPSILON);
            sample[y * ny + x].y = glm::min((y + jy) * dy, 1 - EPSILON);
        }
    }
    int x = min((int)round(u01(rng) * 256), 255);
    float u = sample[x].x;
    float v = sample[x].y;
    float up = sqrt(u); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = v * TWO_PI;
#else
    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;
#endif // STRATIFIED



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

__host__ __device__
glm::vec3 calculatePerfectSpecular(glm::vec3 normal, glm::vec3 ray) {
    return glm::reflect(ray, normal);
    //return -ray + 2 * glm::dot(normal, ray) * normal;
}

__host__ __device__
float cosTheta(glm::vec3 v1, glm::vec3 v2) {
    return glm::dot(glm::normalize(v1), glm::normalize(v2));
}

__host__ __device__
glm::vec3 reflect(const glm::vec3& r, const glm::vec3& normal) {
    return glm::normalize(glm::reflect(r, normal));
    //return glm::normalize(r - 2 * glm::dot(r, normal) * normal);
}

__host__ __device__
glm::vec3 refract(const glm::vec3& r, const glm::vec3& normal, const float &reflect_ratio) {
    return glm::normalize(glm::refract(r, normal, reflect_ratio));
    /*
    float cos_theta = glm::dot(-r, normal);
    glm::vec3 perp = reflect_ratio * (r + cos_theta * normal);
    glm::vec3 parallel = -glm::sqrt(glm::abs(1.0f - glm::length(perp) * glm::length(perp))) * normal;
    return perp + parallel;*/
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
    const Material& m,
    thrust::default_random_engine& rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Specular
    thrust::uniform_real_distribution<float> u01(0, 1);
    Ray r = pathSegment.ray;
    if (m.hasRefractive) {
        Ray r = pathSegment.ray;
        bool outside = glm::dot(r.direction, glm::normalize(normal)) < 0;
        float eta2 = m.indexOfRefraction;

        float refraction_ratio = (outside ? (1.0f / eta2) : eta2);
        

        glm::vec3 unit_direction = glm::normalize(r.direction);
        float cos_theta = glm::min(glm::dot(-unit_direction, normal), 1.0f);

        float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

        float r0 = (1 - refraction_ratio) / (1 + refraction_ratio);
        r0 = r0 * r0;
        float fresnel = r0 + (1 - r0) * glm::pow((1 - glm::abs(cos_theta)), 5.0f);

        glm::vec3 direction;
        if (cannot_refract || fresnel > u01(rng)) {
            pathSegment.ray.direction = reflect(unit_direction, normal);
            //pathSegment.color *= m.specular.color;
            if (outside) {
                pathSegment.ray.origin = intersect + 0.001f * normal;
            }
            else {
                pathSegment.ray.origin = intersect + 0.001f * -normal;
            }
        }
        else {
            pathSegment.ray.direction = refract(unit_direction, normal, refraction_ratio);
            if (outside) {
                pathSegment.ray.origin = intersect + 0.001f * -normal;
            }
            else {
                pathSegment.ray.origin = intersect + 0.001f * normal;
            }
        }
        
    }
    else if (m.hasReflective) {
        // pure reflect
        pathSegment.ray.direction = reflect(glm::normalize(pathSegment.ray.direction), normal);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect;

    }
    else {
        // diffuse
        glm::vec3 diffuseRayDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.direction = glm::normalize(diffuseRayDirection);
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect;

    }



}
