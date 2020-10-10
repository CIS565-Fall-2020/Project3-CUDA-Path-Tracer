#pragma once

#include "intersections.h"

#define RAY_EPSILON 0.0005f


__host__ __device__
glm::vec2 calculateStratifiedSample(
    int iter, int totalIters, thrust::default_random_engine& rng, glm::vec2 pixelLength) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    // Split the pixel into totalIters grids (if possible)
    int grid = (int)(glm::sqrt((float)totalIters) + 0.5f);
    float invGrid = 1.f / grid;

    // Find the grid where current iteration is at
    glm::vec2 topLeft((iter - 1) % grid * invGrid, (iter - 1) / grid * invGrid);
    return glm::vec2(topLeft.x + invGrid * u01(rng), topLeft.y + invGrid * u01(rng));
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng, int iter, int totalIters, glm::vec2 pixelLength) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    //glm::vec2 sample(calculateStratifiedSample(iter, totalIters, rng, pixelLength));
    glm::vec2 sample(u01(rng), u01(rng));
    float sx = sample.x;
    float sy = sample.y;

    float up = sqrt(sx); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = sy * TWO_PI;

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

/*
* Computes sampled glossy reflection direction.
*/
__host__ __device__
glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, float spec_exp, thrust::default_random_engine& rng, int iter, int totalIters, glm::vec2 pixelLength) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 sample(u01(rng), u01(rng));//calculateStratifiedSample(iter, totalIters, rng, pixelLength);
    float s1 = sample.x;
    float s2 = sample.y;
    float theta = glm::acos(glm::pow(s1, 1.f / (spec_exp + 1.f)));
    float phi = TWO_PI * s2;
    glm::vec3 sample_dir(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

    // Compute tangent space
    glm::vec3 tangent;
    glm::vec3 bitangent;

    if (glm::abs(normal.x) > glm::abs(normal.y)) {
        tangent = glm::vec3(-normal.z, 0.f, normal.x) / glm::sqrt(normal.x * normal.x + normal.z * normal.z);
    }
    else {
        tangent = glm::vec3(0, normal.z, -normal.y) / glm::vec3(normal.y * normal.y + normal.z * normal.z);
    }
    bitangent = glm::cross(normal, tangent);

    // Transform sample from specular space to tangent space
    glm::mat3 transform(tangent, bitangent, normal);
    return glm::normalize(transform * sample_dir);
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
void diffuseScatter(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    int iter,
    int totalIters,
    glm::vec2 pixelLength) {

    glm::vec3 diffuseDir = calculateRandomDirectionInHemisphere(normal, rng, iter, totalIters, pixelLength);

    // uniform diffuse
    pathSegment.ray.direction = diffuseDir;
    pathSegment.color *= m.color;

    glm::vec3 originOffset = RAY_EPSILON * normal;
    originOffset = (glm::dot(pathSegment.ray.direction, normal) > 0) ? originOffset : -originOffset;
    pathSegment.ray.origin = intersect + originOffset; // avoid shadow acne
}

__host__ __device__
void mirrorScatter(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);

    // perfect specular
    pathSegment.ray.direction = reflectDir;
    pathSegment.color *= m.specular.color;

    glm::vec3 originOffset = RAY_EPSILON * normal;
    originOffset = (glm::dot(pathSegment.ray.direction, normal) > 0) ? originOffset : -originOffset;
    pathSegment.ray.origin = intersect + originOffset; // avoid shadow acne
}

__host__ __device__
void glossyScatter(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    int iter,
    int totalIters,
    glm::vec2 pixelLength) {

    glm::vec3 reflectDir = calculateImperfectSpecularDirection(normal, m.specular.exponent, rng, iter, totalIters, pixelLength);

    // imperfect specular
    pathSegment.ray.direction = reflectDir;
    pathSegment.color *= m.specular.color;

    glm::vec3 originOffset = RAY_EPSILON * normal;
    originOffset = (glm::dot(pathSegment.ray.direction, normal) > 0) ? originOffset : -originOffset;
    pathSegment.ray.origin = intersect + originOffset; // avoid shadow acne
}

__host__ __device__
void dielectricScatter(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float ior1, float ior2,
    thrust::default_random_engine& rng) {

    float cosine;
    float reflect_prob;
    bool entering = glm::dot(pathSegment.ray.direction, normal) < 0;
    float etaI = entering ? ior1 : ior2;
    float etaT = entering ? ior2 : ior1;
    float eta = etaI / etaT;
    cosine = entering ? -glm::dot(pathSegment.ray.direction, normal) / glm::length(pathSegment.ray.direction) :
        m.indexOfRefraction * glm::dot(pathSegment.ray.direction, normal) / glm::length(pathSegment.ray.direction);
    glm::vec3 refractDir = glm::refract(pathSegment.ray.direction, normal, eta);
    if (glm::length(refractDir) == 0.f) {
        reflect_prob = 1.f;
    }
    else {
        float R0 = (etaI - etaT) / (etaI + etaT);
        R0 *= R0;
        reflect_prob = R0 + (1.f - R0) * glm::pow(1.f - cosine, 5.f);
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float prob = u01(rng);
    if (prob < reflect_prob) {
        refractDir = glm::reflect(pathSegment.ray.direction, normal);
    }
    pathSegment.ray.direction = refractDir;
    pathSegment.color *= m.specular.color;
    pathSegment.ray.origin = intersect + RAY_EPSILON * refractDir;
}

__host__ __device__
void glassScatter(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float ior1, float ior2,
    thrust::default_random_engine& rng) {

    bool entering = glm::dot(-pathSegment.ray.direction, normal) > 0;
    float etaI = entering ? ior1 : ior2;
    float etaT = entering ? ior2 : ior1;
    float eta = etaI / etaT;
    glm::vec3 refractDir = glm::refract(pathSegment.ray.direction, normal, eta);
    if (glm::length(refractDir) == 0.f) {
        refractDir = glm::reflect(pathSegment.ray.direction, normal);
    }

    pathSegment.ray.direction = glm::normalize(refractDir);
    pathSegment.color *= m.specular.color;
    pathSegment.ray.origin = intersect + RAY_EPSILON * refractDir; // avoid shadow acne
}

