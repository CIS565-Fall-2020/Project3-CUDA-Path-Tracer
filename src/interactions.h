#pragma once

#include "intersections.h"

#define RAY_EPSILON 0.0005f
#define NUM_OCTAVES 12


__host__ __device__
glm::vec2 calculateStratifiedSample(
    int iter, int totalIters, thrust::default_random_engine& rng, int depth, int totalDepth) {
    int grid = (int)(glm::sqrt((float)totalIters) + 0.5f);
    float invGrid = 1.f / grid;
    thrust::uniform_real_distribution<float> u02(0, grid);
    int x0 = u02(rng);
    int y0 = u02(rng);
    float x = (float)x0 / (float)grid;
    float y = (float)y0 / (float)grid;

    thrust::uniform_real_distribution<float> u01(0, invGrid);
    return glm::vec2(x + u01(rng), y + u01(rng));
}

__host__ __device__
glm::vec2 calculateGaussianSampling(
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float s = u01(rng);
    float t = u01(rng);
    float a = 0.4f * glm::sqrt(-2.f * glm::log(s));
    float b = 2 * PI * t;
    return glm::vec2(0.5f + a * glm::sin(b), 0.5f + a * glm::cos(b));
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng, int iter, int totalIters, int depth, int totalDepth) {
    //thrust::uniform_real_distribution<float> u01(0, 1);
    //float sx = u01(rng);
    //float sy = u01(rng);
    glm::vec2 samples = calculateStratifiedSample(iter, totalIters, rng, depth, totalDepth);
    float sx = samples.x;
    float sy = samples.y;

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
    glm::vec3 normal, float spec_exp, thrust::default_random_engine& rng, int iter, int totalIters, int depth, int totalDepth) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 sample(calculateStratifiedSample(iter, totalIters, rng, depth, totalDepth));
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

// Helper functions for FBM

__host__ __device__
float random(glm::vec2 st) {
    return glm::fract(glm::sin(glm::dot(glm::vec2(st.x, st.y), glm::vec2(12.9898, 78.233))) * 43758.5453123);
}

__host__ __device__
glm::mat2 rotate2d(float angle) {
    return glm::mat2(cos(angle), -sin(angle),
        sin(angle), cos(angle));
}

__host__ __device__
float noise(glm::vec2 st) {
    glm::vec2 i = glm::floor(st);
    glm::vec2 f = glm::fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + glm::vec2(1.f, 0.f));
    float c = random(i + glm::vec2(0.f, 1.f));
    float d = random(i + glm::vec2(1.f, 1.f));

    glm::vec2 u = f * f * (3.f - 2.f * f);

    return glm::mix(a, b, u.x) +
        (c - a) * u.y * (1.f - u.x) +
        (d - b) * u.x * u.y;
}

__host__ __device__
float lines(glm::vec2 pos, float b) {
    float scale = 10.0;
    pos *= scale;
    return glm::smoothstep(0.0f,
        .5f + b * .5f,
        abs((sin(pos.x * 3.1415f) + b * 2.0f)) * .5f);
}

__host__ __device__
float fbm(glm::vec2 st) {
    float v = 0.f;
    float a = 0.5f;
    glm::vec2 shift = glm::vec2(100.f);
    // Rotate to reduce axial bias
    glm::mat2 rot = glm::mat2(glm::cos(0.5f), glm::sin(0.5f),
        -glm::sin(0.5f), glm::cos(0.5f));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(st);
        st = rot * st * 2.f + shift;
        a *= 0.5f;
    }
    return v;
}

__host__ __device__
glm::vec3 calculateFBMTexture(
    glm::vec3 normal,
    const Material& m)
{
    glm::vec2 st(normal.x, normal.y);
    st *= glm::abs(normal.z);
    glm::vec2 q = glm::vec2(0.f);
    q.x = fbm(st);
    q.y = fbm(st + glm::vec2(1.f));

    glm::vec2 r = glm::vec2(0.f);
    r.x = fbm(st + 1.f * q + glm::vec2(1.7f, 9.2f) + 0.15f);
    r.y = fbm(st + 1.f * q + glm::vec2(8.3f, 2.8f) + 0.126f);

    float f = fbm(st + r);

    return glm::vec3((f * f * f + .7f * f * f + .8f * f) * m.color);
}

__host__ __device__
glm::vec3 calculateNoiseTexture(
    glm::vec3 normal,
    const Material& m)
{
    glm::vec2 st(normal.x, normal.y);
    st *= glm::abs(normal.z);
    glm::vec2 pos = glm::vec2(st.x, st.y) * glm::vec2(10.f, 3.f);

    float pattern = pos.x;

    // Add noise
    pos = rotate2d(noise(pos)) * pos;

    // Draw lines
    pattern = lines(pos, .5f);
    return glm::vec3(pattern);
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
    int depth,
    int totalDepth) {

    glm::vec3 color = m.color;
    if (m.hasTexture) {
        if (m.texture == FBM) color = calculateFBMTexture(normal, m);
        if (m.texture == NOISE) color = calculateNoiseTexture(normal, m);
    }
    glm::vec3 diffuseDir = calculateRandomDirectionInHemisphere(normal, rng, iter, totalIters, depth, totalDepth);

    // uniform diffuse
    pathSegment.ray.direction = diffuseDir;
    pathSegment.color *= color;

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

    glm::vec3 color = m.specular.color;
    if (m.hasTexture) {
        if (m.texture == FBM) color = calculateFBMTexture(normal, m);
        if (m.texture == NOISE) color = calculateNoiseTexture(normal, m);
    }

    glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);

    // perfect specular
    pathSegment.ray.direction = reflectDir;
    pathSegment.color *= color;

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
    int depth,
    int totalDepth) {

    glm::vec3 color = m.specular.color;
    if (m.hasTexture) {
        if (m.texture == FBM) color = calculateFBMTexture(normal, m);
        if (m.texture == NOISE) color = calculateNoiseTexture(normal, m);
    }

    glm::vec3 reflectDir = calculateImperfectSpecularDirection(normal, m.specular.exponent, rng, iter, totalIters, depth, totalDepth);

    // imperfect specular
    pathSegment.ray.direction = reflectDir;
    pathSegment.color *= color;

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

    glm::vec3 color = m.specular.color;
    if (m.hasTexture) {
        if (m.texture == FBM) color = calculateFBMTexture(normal, m);
        if (m.texture == NOISE) color = calculateNoiseTexture(normal, m);
    }

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
    pathSegment.color *= color;
    pathSegment.ray.origin = intersect + RAY_EPSILON * refractDir;
}

__host__ __device__
void glassScatter(PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    float ior1, float ior2,
    thrust::default_random_engine& rng) {

    glm::vec3 color = m.specular.color;
    if (m.hasTexture) {
        if (m.texture == FBM) color = calculateFBMTexture(normal, m);
        if (m.texture == NOISE) color = calculateNoiseTexture(normal, m);
    }

    bool entering = glm::dot(-pathSegment.ray.direction, normal) > 0;
    float etaI = entering ? ior1 : ior2;
    float etaT = entering ? ior2 : ior1;
    float eta = etaI / etaT;
    glm::vec3 refractDir = glm::refract(pathSegment.ray.direction, normal, eta);
    if (glm::length(refractDir) == 0.f) {
        refractDir = glm::reflect(pathSegment.ray.direction, normal);
    }

    pathSegment.ray.direction = glm::normalize(refractDir);
    pathSegment.color *= color;
    pathSegment.ray.origin = intersect + RAY_EPSILON * refractDir; // avoid shadow acne
}

