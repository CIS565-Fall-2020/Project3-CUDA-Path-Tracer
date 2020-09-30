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

#define PI 3.141592
//TODO: try and implement stratified sampler
// Samples a disc concentrically; used for thin lens camera
// Taken from http://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations.html
__host__ __device__
glm::vec3 sampleDiskConcentric(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float x = u01(rng),
          y = u01(rng);

    float phi, r, u, v;
    float a = 2 * x - 1;
    float b = 2 * y - 1;

    if (a > -b) {
        if (a > b) {
            r = a;
            phi = (PI / 4) * (b / a);
        }
        else {
            r = b;
            phi = (PI / 4) * (2 - (a / b));
        }
    }
    else {
        if (a < b) {
            r = -a;
            phi = (PI / 4) * (4 + (b / a));
        }
        else {
            r = -b;
            if (b < 0 || b > 0) {
                phi = (PI / 4) * (6 - (a / b));
            }
            else {
                phi = 0;
            }
        }
    }
    return glm::vec3(r * cosf(phi), r * sinf(phi), 0);
}

// specular exponent
// material going to look
// reflection
// if ray bouncing off of material, what is chance that it's perfect reflection
// diffuse or refractive
// black sphere
// if specular exponent bigger,
// that dot would be larger
// lighting equation
// use specex as power

__host__ __device__ void calculateDiffuse(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    Ray& r = pathSegment.ray;
    glm::vec3 hemiDir = calculateRandomDirectionInHemisphere(normal, rng);
    r.origin = intersect + 0.0001f * normal;
    r.direction = hemiDir;
    pathSegment.color *= m.color;
}

__host__ __device__ void calculateReflect(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    Ray& r = pathSegment.ray;
    glm::vec3 reflectedDir = glm::reflect(r.direction,
        normal);
    // color specular
    glm::vec3 specularColor = m.specular.color;
    r.origin = intersect + 0.0001f * normal;
    r.direction = reflectedDir;

    if (m.specular.exponent > 0) {
        glm::vec3 specularColor = m.specular.color;
        r.direction = reflectedDir;
        pathSegment.color *= specularColor;
    }
}

__host__ __device__ glm::vec3 evaluateFresnel(float cosThetaI, float eI, float eT) {
    float sinThetaI = std::sqrt(std::max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = eI / eT * sinThetaI;
    if (sinThetaT >= 1) {
        return glm::vec3(1.);
    }
    float cosThetaT = std::sqrt(std::max(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((eT * cosThetaI) - (eI * cosThetaT)) /
        ((eT * cosThetaI) + (eI * cosThetaT));
    float Rperp = ((eI * cosThetaI) - (eT * cosThetaT)) /
        ((eI * cosThetaI) + (eT * cosThetaT));
    return glm::clamp(glm::vec3((Rparl * Rparl + Rperp * Rperp)) / 2.f, 0.f, 1.f);
}

__host__ __device__ void calculateRefract(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    Ray& r = pathSegment.ray;

    //thrust::uniform_real_distribution<float> u01(0, 1);
    //float random = u01(rng);

    // assumes that the other medium is air, whose IOR is approximately 1.
    float cosTheta = glm::dot(r.direction, normal);
    bool entering = cosTheta > 0;
    float etaI = entering ? 1.0f : m.indexOfRefraction;
    float etaT = entering ? m.indexOfRefraction : 1.0f;
    if (!entering) {
        cosTheta = fabsf(cosTheta);
    }

    glm::vec3 reflectDir = glm::reflect(r.direction, normal);
    glm::vec3 refractedDir = glm::refract(r.direction, normal, etaI / etaT);
    if (!entering) {
        refractedDir = glm::refract(r.direction, -normal, etaI / etaT);
    }
    r.direction = refractedDir;
    r.origin = intersect - 0.0001f * normal;

    pathSegment.color *= m.specular.color;
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

    float probThresholdDiffuse = -1.0f;
    float probThresholdReflect = -1.0f;
    float probThresholdRefract = -1.0f;

    const unsigned int bsdf_diffuse = 1 << 0;
    const unsigned int bsdf_reflection = 1 << 1;
    const unsigned int bsdf_refraction = 1 << 2;

    unsigned int flags = 0;
    int numFlags = 0;

    if (m.hasDiffuse) {
        flags = flags | bsdf_diffuse;
        numFlags++;
    }

    if (m.hasReflective) {
        flags = flags | bsdf_reflection;
        numFlags++;
    }

    if (m.hasRefractive) {
        flags = flags | bsdf_refraction;
        numFlags++;
    }

    switch (flags) {
    case bsdf_diffuse:
        probThresholdDiffuse = 1.0f;
        break;
    case bsdf_reflection:
        probThresholdReflect = 1.0f;
        break;
    case bsdf_refraction:
        probThresholdRefract = 1.0f;
        break;
    case bsdf_diffuse | bsdf_reflection:
        probThresholdDiffuse = 0.5f;
        probThresholdReflect = 1.0f;
        break;
    case bsdf_diffuse | bsdf_refraction:
        probThresholdDiffuse = 0.5f;
        probThresholdRefract = 1.0f;
        break;
    case bsdf_reflection | bsdf_refraction:
        probThresholdReflect = 0.5f;
        probThresholdRefract = 1.0f;
        break;
    case bsdf_diffuse | bsdf_reflection | bsdf_refraction:
        probThresholdDiffuse = 0.333f;
        probThresholdReflect = 0.667f;
        probThresholdReflect = 1.0f;
        break;

    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);

    if (random <= probThresholdDiffuse) {
        calculateDiffuse(pathSegment, intersect, normal, m, rng);
    }
    else if (random <= probThresholdReflect) {
        calculateReflect(pathSegment, intersect, normal, m, rng);
    }
    else if (random <= probThresholdRefract) {
        calculateRefract(pathSegment, intersect, normal, m, rng);
    }

    pathSegment.color *= numFlags;
}
