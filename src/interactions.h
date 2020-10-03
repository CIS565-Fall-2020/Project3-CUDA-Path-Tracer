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
glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, float spec_exp, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float s1 = u01(rng);
    float s2 = u01(rng);
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
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

    glm::vec3 originOffset = 0.0005f * normal;
    glm::vec3 diffuseDir = calculateRandomDirectionInHemisphere(normal, rng);
    //glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);
    glm::vec3 reflectDir = calculateImperfectSpecularDirection(normal, m.specular.exponent, rng);

    float cosine;
    float reflect_prob;
    float idx_ref_other = 1.f; // Replace this later
    bool entering = glm::dot(pathSegment.ray.direction, normal) < 0;
    float etaI = entering ? idx_ref_other : m.indexOfRefraction;
    float etaT = entering ? m.indexOfRefraction : idx_ref_other;
    float eta = etaI / etaT;
    cosine = entering ? -glm::dot(pathSegment.ray.direction, normal) / pathSegment.ray.direction.length() :
        m.indexOfRefraction * glm::dot(pathSegment.ray.direction, normal) / pathSegment.ray.direction.length();
    glm::vec3 refractDir = glm::refract(pathSegment.ray.direction, normal, eta);
    if (glm::length(refractDir) == 0.f) {
        reflect_prob = 1.f;
    }
    else {
        float R0 = (etaI - etaT) / (etaI + etaT);
        R0 *= R0;
        reflect_prob = R0 + (1.f - R0) * glm::pow(1.f - cosine, 5.f);
    }

    if (m.hasReflective) {
        // reflective
        pathSegment.ray.direction = reflectDir;
        pathSegment.color *= (m.specular.color);
        originOffset = (glm::dot(pathSegment.ray.direction, normal) > 0) ? originOffset : -originOffset;
        pathSegment.ray.origin = intersect + originOffset; // avoid shadow acne
    }
    else if (m.hasRefractive) {
        // refractive
        thrust::uniform_real_distribution<float> u01(0, 1);
        float prob = u01(rng);
        if (prob < reflect_prob) {
            refractDir = glm::reflect(pathSegment.ray.direction, normal);
        }
        pathSegment.ray.direction = refractDir;
        pathSegment.color *= (m.specular.color);
        pathSegment.ray.origin = intersect + 0.0005f * refractDir; // avoid shadow acne
    }
    else {
        // uniform diffuse
        pathSegment.ray.direction = diffuseDir;
        pathSegment.color *= m.color;
        originOffset = (glm::dot(pathSegment.ray.direction, normal) > 0) ? originOffset : -originOffset;
        pathSegment.ray.origin = intersect + originOffset; // avoid shadow acne
    }
}
