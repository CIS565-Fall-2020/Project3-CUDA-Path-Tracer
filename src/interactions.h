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
glm::vec3 calculatePerfectSpecular(glm::vec3 normal, glm::vec3 ray) {
    return glm::reflect(ray, normal);
    //return -ray + 2 * glm::dot(normal, ray) * normal;
}

__host__ __device__
float cosTheta(glm::vec3 v1, glm::vec3 v2) {
    return glm::dot(glm::normalize(v1), glm::normalize(v2));
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


    if (m.hasReflective && m.hasRefractive) {
        glm::vec3 ray = pathSegment.ray.direction;

        float etaI = 1.0f, etaT = m.indexOfRefraction;
        float r0 = pow((etaI - etaT) / (etaI + etaT), 2);

        float etaRatio = 0;
        float costheta = glm::dot(ray, glm::normalize(normal));
        glm::vec3 newNormal = normal;

        bool enter = costheta < 0 ? true : false;

        if (!enter) {
            float tmp = etaI;
            etaI = etaT;
            etaT = tmp;
            costheta = glm::abs(costheta);
            newNormal *= -1;
        }

        etaRatio = etaI / etaT;
        float rTheta = r0 + (1 - r0) * glm::pow(1 - costheta, 5);

        if (u01(rng) > 0.5) {
            float a = 0;
        }

        if (u01(rng) < rTheta) {
            // refelction
            glm::vec3 reflectDirection = glm::normalize(glm::reflect(ray, glm::normalize(normal)));
            pathSegment.ray.origin = intersect;
            pathSegment.ray.direction = reflectDirection;
        }
        else {
            glm::vec3 refractDirection = glm::normalize(glm::refract(ray, glm::normalize(newNormal), etaRatio));
            pathSegment.ray.origin = intersect;
            pathSegment.ray.direction = refractDirection;
        }
    }
    else if (m.hasReflective) {
        // pure reflect
        glm::vec3 ray = intersect - pathSegment.ray.origin;
        glm::vec3 reflectRayDirection = calculatePerfectSpecular(normal, ray);
        glm::vec3 specularColor = m.specular.color;
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = glm::normalize(reflectRayDirection);
        pathSegment.color *= specularColor;

    }
    else {
        // diffuse
        glm::vec3 diffuseRayDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = glm::normalize(diffuseRayDirection);
        pathSegment.color *= m.color;
    }

}
