#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "utility"
#include "utilities.h"

__host__ __device__ glm::vec3 crossDirection(glm::vec3 v) {
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

__host__ __device__
void CoordinateSys(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3) {
    glm::vec3 directionNotNormal = crossDirection(v1);

    // Use not-normal direction to generate two perpendicular directions
    v2 = glm::normalize(glm::cross(v1, directionNotNormal));
    v3 = glm::normalize(glm::cross(v1, v2));
}

__forceinline__
__host__ __device__ glm::vec3 sampleTexture(glm::vec3* dev_textures, glm::vec2 uv, TextureDescriptor tex)
{
    //uv.y = 1.f - uv.y;
    uv = glm::mod(uv * tex.repeat, glm::vec2(1.f));
    if (tex.type == 0)
    {
        int x = glm::min((int)(uv.x * tex.width), tex.width - 1);
        int y = glm::min((int)(uv.y * tex.height), tex.height - 1);
        int index = y * tex.width + x;
        return dev_textures[tex.index + index];
    }
    else
    {
        int steps = 0;
        glm::vec2 z = glm::vec2(0.f);
        glm::vec2 c = (uv * 2.f - glm::vec2(1.f)) * 1.5f;
        c.x -= .5;

        for (steps = 0; steps < 100; steps++)
        {
            float x = z.x * z.x - z.y * z.y + c.x;
            float y = 2.f * z.x * z.y + c.y;

            z = glm::vec2(x, y);

            if (glm::dot(z, z) > 2.f)
                break;
        }

        float sn = float(steps) - log2(log2(dot(z, z))) + 4.0f; // http://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
        sn = glm::clamp(sn, 0.1f, 1.f);
        return glm::vec3(sn);
    }
}