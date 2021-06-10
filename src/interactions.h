#pragma once

#include "intersections.h"
#define USE_ALT_METHOD false

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
glm::vec3 altDirGenerator(
    glm::vec3 normal, thrust::default_random_engine& rng, int iter, int depth) {

    if (depth < 9) {
        return calculateRandomDirectionInHemisphere(normal, rng);
    }

    // pick a grid cell
    thrust::uniform_real_distribution<float> u(0, 1);
    int mod = 30;

    float posX = iter % mod;
    float posY = int((iter - posX) / mod);

    posX = (posX + u(rng)) / mod;
    posY = (posY + u(rng)) / mod;

    float up = sqrt(posX); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = posY * TWO_PI;

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

__host__ __device__ float power_5(float val) {
    return val * val * val * val * val;
}


__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    int iter, int depth) {

    pathSegment.remainingBounces -= 1;
    pathSegment.ray.origin = intersect + 0.001f * normal;
    thrust::uniform_real_distribution<float> u(0, 1);
    float randVar = u(rng);

    if (m.hasRefractive) {
        float indexOfRef = m.indexOfRefraction;
        float initialRef = (1 - indexOfRef) / (1 + indexOfRef);
        initialRef = initialRef * initialRef;

        glm::vec3 dir = pathSegment.ray.direction;
        float cosTheta = glm::dot(glm::normalize(normal), glm::normalize(dir));
        float thresh = initialRef + (1 - initialRef) * power_5(1.f - cosTheta);
        bool inObject = pathSegment.inObject;


        if (!inObject && randVar > thresh) {
            pathSegment.color = pathSegment.color * m.specular.color;
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            glm::vec3 randomDirection;

            if (inObject) {
                randomDirection = glm::refract(glm::normalize(dir), glm::normalize(normal), indexOfRef);
            }
            else {
                randomDirection = glm::refract(glm::normalize(dir), glm::normalize(normal), 1.f / indexOfRef);
            }

            if (glm::length(randomDirection) == 0) {
                pathSegment.color = pathSegment.color * m.specular.color;
                pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            }
            else {
                pathSegment.color = pathSegment.color * m.color;
                pathSegment.ray.direction = randomDirection;
                pathSegment.ray.origin = intersect + (0.1f * randomDirection);
                pathSegment.inObject = !inObject;
            }
        }
    }
    else if (m.hasReflective && randVar > 0.3) {
        pathSegment.color = pathSegment.color * m.specular.color;
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    }
    else {
        pathSegment.color = pathSegment.color * m.color;
        if (USE_ALT_METHOD) {
            pathSegment.ray.direction = altDirGenerator(normal, rng, iter, depth);
        }
        else {
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        }
    }
}

struct isBouncing {
    __host__ __device__
        bool operator()(const PathSegment& segment) {
        return segment.remainingBounces != 0;
    }
};
