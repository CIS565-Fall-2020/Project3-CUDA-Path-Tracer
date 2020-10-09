#pragma once
#include "intersections.h"
#include "fresnel.h"
#include "sceneStructs.h"

#define STRATIFIEDSAMPLING true
#define EPSILON 0.000618f
#define SAMPLESPERPIXEL 36

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r1 = u01(rng), r2 = u01(rng);

#if STRATIFIEDSAMPLING
    int samplesPerPixel = SAMPLESPERPIXEL;
    int sqrtVal = sqrt(float(samplesPerPixel));
    // A number useful for scaling a square of size sqrtVal x sqrtVal to 1 x 1
    float invSqrtVal = 1.f / sqrtVal;
    int i = u01(rng) * SAMPLESPERPIXEL;
    int y = i / sqrtVal;
    int x = i % sqrtVal;
    glm::vec2 sample = glm::vec2((x + r1) * invSqrtVal,
        (y + r2) * invSqrtVal);
    r1 = sample.x;
    r2 = sample.y;
#endif // STRATIFIEDSAMPLING

    float up = sqrt(r1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = r2 * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal +
        cos(around) * over * perpendicularDirection1 +
        sin(around) * over * perpendicularDirection2;
}

__host__ __device__
void sampleLi(const ShadeableIntersection& ref, const Geom& lightGeom, const glm::vec2& xi, glm::vec3* wi, float* pdf)
{
    ShadeableIntersection it;
    if (lightGeom.type == GeomType::CUBE)
        it = getSampleOnSquare(xi, pdf, lightGeom);
    else if (lightGeom.type == GeomType::SPHERE)
        it = getSampleOnSphere(xi, pdf, lightGeom);

    *wi = it.point - ref.point;
    float len2 = glm::length2(*wi);
    if (len2 > 0)
    {
        *wi = glm::normalize(*wi);
    }

    float cosTheta = glm::abs(glm::dot(it.surfaceNormal, *wi));
    if (cosTheta == 0)
    {
        *pdf = 0.f;
    }

    // Convert the PDF with respect to surface area to a PDF with respect to solid angle
    *pdf *= len2 / cosTheta;
}


/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 */
__host__ __device__
void scatterIndirectRay(PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 intersectionPoint = getPointOnRay(pathSegment.ray, intersection.t);
    glm::vec3 normal = intersection.surfaceNormal;
    glm::vec3 wo = pathSegment.ray.direction;

    if (m.hasRefractive > 0.f)
    {
        thrust::uniform_real_distribution<float> u01(0, 1);
        pathSegment.ray.direction = FresnelDielectric::evaluate(wo, normal, m.hasRefractive, u01(rng));
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersectionPoint + EPSILON * pathSegment.ray.direction;
    }
    else if (m.hasReflective > 0.f)  // specular
    {
        pathSegment.ray.direction = glm::reflect(wo, normal);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersectionPoint;
    }
    else  // diffuse
    {
        glm::vec3 wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        // Pure diffuse use Lambertian BRDF
        float cosTheta = glm::dot(normal, wi);
        float pdf = cosTheta * INV_PI;
        glm::vec3 f = m.color * INV_PI;

        if (pdf == 0.f)
        {
            pathSegment.color = glm::vec3(0.f);
            pathSegment.remainingBounces = 0;
            return;
        }

        pathSegment.ray.direction = wi;
        pathSegment.color = f * pathSegment.color * std::abs(cosTheta) / pdf;
        pathSegment.ray.origin = intersectionPoint;
    }

    pathSegment.remainingBounces--;
}

__host__ __device__
void scatterDirectRay(PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    const Material& m,
    thrust::default_random_engine& rng,
    Geom* lightGeoms,
    int num_lights)
{
    if (num_lights == 0 || m.hasRefractive > 0.f || m.hasReflective > 0.f || pathSegment.remainingBounces == 1)
    {
        pathSegment.remainingBounces = 0;
        pathSegment.color = glm::vec3(0);
    }
    else
    {
        pathSegment.remainingBounces = 1;
        glm::vec3 intersectionPoint = getPointOnRay(pathSegment.ray, intersection.t);
        glm::vec3 normal = intersection.surfaceNormal;

        // Randomly choose a single light to sample
        thrust::uniform_real_distribution<float> u01(0, 1);
        int randLightIdx = min(int(u01(rng) * num_lights), num_lights - 1);
        const Geom& randLight = lightGeoms[randLightIdx];

        float lightPdf;
        glm::vec3 wi;

        float r1 = u01(rng), r2 = u01(rng);
#if STRATIFIEDSAMPLING
        int samplesPerPixel = SAMPLESPERPIXEL;
        int sqrtVal = sqrt(float(samplesPerPixel));
        float invSqrtVal = 1.f / sqrtVal;

        int i = u01(rng) * SAMPLESPERPIXEL;
        int y = i / sqrtVal;
        int x = i % sqrtVal;
        glm::vec2 sample = glm::vec2((x + r1) * invSqrtVal,
            (y + r2) * invSqrtVal);
        r1 = sample.x;
        r2 = sample.y;
#endif // STRATIFIEDSAMPLING

        sampleLi(intersection, randLight, glm::vec2(r1, r2), &wi, &lightPdf);
        if (lightPdf == 0.f)
        {
            pathSegment.color = glm::vec3(0);
        }
        else
        {
            // Do visibility test in the next recursion by calling computeIntersection()
            pathSegment.ray.origin = intersectionPoint;
            pathSegment.ray.direction = wi;

            float cosTheta = glm::dot(normal, wi);
            glm::vec3 f = m.color;
            pathSegment.color = float(num_lights) * f * abs(cosTheta) / lightPdf;
        }
    }
}
