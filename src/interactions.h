#pragma once
#include "intersections.h"
#include "fresnel.h"

#define STRATIFIEDSAMPLING true
#define SAMPLESPERPIXEL 100

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine &rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r1 = u01(rng), r2 = u01(rng);

#if STRATIFIEDSAMPLING
    int samplesPerPixel = SAMPLESPERPIXEL;
    int sqrtVal = (int)(sqrt((float)samplesPerPixel) + 0.5);
    // A number useful for scaling a square of size sqrtVal x sqrtVal to 1 x 1
    float invSqrtVal = 1.f / sqrtVal;

    samplesPerPixel = sqrtVal * sqrtVal;
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

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 */
__host__ __device__
void scatterRay(PathSegment& pathSegment,
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
        pathSegment.ray.origin = intersectionPoint + 0.000618f * pathSegment.ray.direction;
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
