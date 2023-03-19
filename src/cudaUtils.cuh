#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "Constants.h"
#include "utility"

__host__ __device__ __forceinline__ glm::vec3 crossDirection(glm::vec3 v) {
    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    if (abs(v.x) < SQRT_OF_ONE_THIRD) {
        return glm::vec3(1, 0, 0);
    }
    else if (abs(v.y) < SQRT_OF_ONE_THIRD) {
        return glm::vec3(0, 1, 0);
    }
    return glm::vec3(0, 0, 1);
}

__host__ __device__ __forceinline__
void CoordinateSys(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3) {
    glm::vec3 directionNotNormal = crossDirection(v1);

    // Use not-normal direction to generate two perpendicular directions
    v2 = glm::normalize(glm::cross(v1, directionNotNormal));
    v3 = glm::normalize(glm::cross(v1, v2));
}

