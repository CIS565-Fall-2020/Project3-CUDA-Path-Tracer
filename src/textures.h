#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"

__host__ __device__ float noise(float i) {
    return glm::fract(glm::sin(glm::vec2(203.311f * float(i), float(i) * sin(0.324f + 140.0f * float(i))))).x;
}

__host__ __device__ float randomSeeded1D(float in, float seed) {
    return glm::fract(glm::cos(glm::length(glm::cross(glm::vec3(341.2f, 0, in),
                                  glm::vec3(seed, seed, 2.3f)))));
}

__host__ __device__ float randomSeeded2D(glm::vec2 p, float seed) {
    return glm::fract(glm::sin(glm::dot(p + glm::vec2(seed), glm::vec2(127.1f, 311.7f))) * 43758.5453f);
}

__host__ __device__ float randomSeeded3D(glm::vec3 p, float seed) {
    return glm::fract(glm::sin(glm::dot(p + glm::vec3(seed), glm::vec3(seed, 327.1f, 566.7f))) * 726.0f);
}

// Fractional Brownian Noise (1D)

__host__ __device__ float interpNoise1D(float x) {
    float intX = glm::floor(x);
    float fractX = glm::fract(x);

    float v1 = noise(intX);
    float v2 = noise(intX + 1.0f);
    return glm::mix(v1, v2, fractX);
}

__host__ __device__ float fbmNoise(float x) {
    float total = 0.0f;
    float persistence = 0.5f;
    int octaves = 8;

    for (int i = 0; i < octaves; i++) {
        float freq = pow(2.0f, float(i));
        float amp = pow(persistence, float(i));

        total += interpNoise1D(x * freq) * amp;
    }

    return total;
}

// Fractional Brownian Noise (2D)

__host__ __device__ float interpNoise2D(float x, float y) {
    float intX = glm::floor(x);
    float fractX = glm::fract(x);
    float intY = glm::floor(y);
    float fractY = glm::fract(y);

    float v1 = randomSeeded2D(glm::vec2(intX, intY), 0);
    float v2 = randomSeeded2D(glm::vec2(intX + 1.0f, intY), 0);
    float v3 = randomSeeded2D(glm::vec2(intX, intY + 1.0f), 0);
    float v4 = randomSeeded2D(glm::vec2(intX + 1.0f, intY + 1.0f), 0);

    float i1 = glm::mix(v1, v2, fractX);
    float i2 = glm::mix(v3, v4, fractX);
    return glm::mix(i1, i2, fractY);
}

__host__ __device__ float fbm2(glm::vec2 p) {
    float total = 0.0f;
    float persistence = 0.5f;
    int octaves = 8;

    for (int i = 0; i < octaves; i++) {
        float freq = pow(2.0f, float(i));
        float amp = pow(persistence, float(i));

        total += interpNoise2D(p.x * freq, p.y * freq) * amp;
    }

    return total;
}

// "Perturbed" FBM (FBM within FBM)

__host__ __device__ float perturbedFbm(glm::vec2 p)
{
    glm::vec2 q = glm::vec2(fbm2(p + glm::vec2(0.0, 0.0)),
        fbm2(p + glm::vec2(4.8, -1.3)));

    glm::vec2 r = glm::vec2(fbm2(p + 4.0f * q + glm::vec2(1.4, 9.2)),
        fbm2(p + 4.0f * q + glm::vec2(2.5, 7.8)));

    return fbm2(p + 4.0f * r);
}

// Worley Noise
__host__ __device__ glm::vec2 generate_point(glm::vec2 cell, float cell_size) {
    glm::vec2 p = glm::vec2(cell.x, cell.y);
    p += glm::fract(sin(glm::vec2(glm::dot(p, glm::vec2(127.1, 311.7)), glm::dot(p, glm::vec2(269.5, 183.3)) * 43758.5453)));
    return p * cell_size;
}

__host__ __device__ float worleyNoise(glm::vec2 p, float cell_size) {
    glm::vec2 cell = glm::floor(p / cell_size);

    glm::vec2 point = generate_point(cell, cell_size);

    float shortest_distance = glm::length(p - point);

    // compute shortest distance from cell + neighboring cell points

    for (float i = -1.0f; i <= 1.0f; i += 1.0f) {
        float ncell_x = cell.x + i;
        for (float j = -1.0f; j <= 1.0f; j += 1.0f) {
            float ncell_y = cell.y + j;

            // get the point for that cell
            glm::vec2 npoint = generate_point(glm::vec2(ncell_x, ncell_y), cell_size);

            // compare to previous distances
            float distance = glm::length(p - npoint);
            if (distance < shortest_distance) {
                shortest_distance = distance;
            }
        }
    }

    return shortest_distance / cell_size;
}

__host__ __device__ glm::vec3 proceduralTexture1(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal) {

    // return blue
    if (randomSeeded1D(intersect.y, pathSegment.remainingBounces) > 0.9) {
        return glm::vec3(0.25f, 0.25f, 0.85f);
    }

    // return red
    if (randomSeeded1D(intersect.y, pathSegment.remainingBounces) > 0.7) {
        return glm::vec3(0.85f, 0.2f, 0.3f);
    }

    // return yellow
	return glm::vec3(0.85, 0.67, 0.35);
}

__host__ __device__ glm::vec3 proceduralTexture2(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal) {

    glm::vec3 testVec = glm::vec3(0, 1, 0);
    float thetaY = glm::acos(glm::dot(testVec, normal));
    float worley = worleyNoise(glm::vec2(intersect.x, intersect.z), 0.25f);
    thetaY += worley;
    if (thetaY > 80 * PI / 180) {
        if (thetaY > 120 * PI / 180) {
            return glm::vec3(0.85);
        }

        glm::vec3 initialColor = worley * glm::vec3(0.85, 0.67, 0.35);
        initialColor += glm::vec3(0.35, 0, 0); // add a bit of red
        return glm::clamp(initialColor, 0.1f, 0.85f);
    }

    return worley * glm::vec3(0.35, 0.67, 0.85);
}

__host__ __device__ glm::vec3 proceduralTexture3(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal) {

    float worley = worleyNoise(glm::vec2(intersect.x, intersect.z), 1.0f);
    glm::vec3 green = perturbedFbm(glm::vec2(intersect.y, intersect.z)) * glm::vec3(0.45, 0.87, 0.45);
    
    return glm::mix(green, glm::vec3(0.35, 0.35, 0.85), 1.2 * worley);
    
}