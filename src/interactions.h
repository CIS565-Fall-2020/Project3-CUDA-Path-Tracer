#pragma once

#include "intersections.h"
#include "utilities.h"

#define XBASE 3
#define YBASE 5
#define XRES 100
#define YRES 100

__host__ __device__
float haltonSequence(float index, float base) 
{
    float f = 1.0f;
    float r = 0.0f;

    while (index > 0.0f) 
    {
        f = f / base;
        r = r + f * ((int)index % (int)base);
        index = index / base;
    }

    return r;
}

__host__ __device__ glm::vec3 squareLightSample(
    thrust::default_random_engine& rng,
    glm::vec3 interPos,
    glm::mat4 transform)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float xGrid = u01(rng);
    float yGrid = u01(rng);

    glm::vec3 pointOnLight = glm::vec3(transform * glm::vec4(xGrid - 0.5f, -0.500f, yGrid - 0.5f, 1.0f));

    return pointOnLight - interPos;
}


// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float xGrid = u01(rng);
    float yGrid = u01(rng);

    float xHalton = haltonSequence((int)(yGrid * YRES) * XRES + (int)(xGrid * XRES), XBASE);
    float yHalton = haltonSequence((int)(yGrid * YRES) * XRES + (int)(xGrid * XRES), YBASE);

    float up = sqrt(yHalton); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xHalton * TWO_PI;

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
glm::vec2 concentricSampling(const glm::vec2& u) 
{
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f, 1.0f);
    if (uOffset.x == 0 && uOffset.y == 0)
        return glm::vec2(0.0f);

    float theta, r;
    if (fabs(uOffset.x) > fabs(uOffset.y)) 
    {
        r = uOffset.x;
        theta = PI / 4 * (uOffset.y / uOffset.x);
    }
    else 
    {
        r = uOffset.y;
        theta = PI / 2 - PI / 4 * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(cos(theta), sin(theta));
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
void directRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    Geom* lights,
    int lightNum) 
{
    if (m.hasReflective == 0.0f && m.hasRefractive == 0.0f)
    {
        float Pi = 3.1415926f;
        thrust::uniform_real_distribution<float> u01(0, 1);
        int lightSample = glm::min((int)(u01(rng) * lightNum), lightNum - 1);
        glm::vec3 rayDir = squareLightSample(rng, intersect, lights[lightSample].transform);
        float area = fabs(lights[lightSample].scale.x) * fabs(lights[lightSample].scale.z);
        Ray scatterRay;
        scatterRay.origin = intersect + 0.001f * glm::normalize(rayDir);
        scatterRay.direction = glm::normalize(rayDir);
        float lightPdf = 1.0f / area;

        if (lightPdf == 0)
        {
            pathSegment.color = glm::vec3(0.0f, 0.0f, 0.0f);
            pathSegment.remainingBounces = 0;
        }
        else
        {
            glm::vec3 f = m.color / Pi;
            pathSegment.color += (f * glm::abs(glm::dot(rayDir, normal))) * (float)lightNum / lightPdf;
            pathSegment.ray = scatterRay;
        }
    }
    else
    {
        pathSegment.color = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    if (pathSegment.remainingBounces > 1)
    {
        pathSegment.remainingBounces = 1;
    }
    else
        pathSegment.remainingBounces--;
}


__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng,
        bool isMesh) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    if (isMesh) 
    {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float result = u01(rng);

        if (m.hasReflective > result) 
        {
            glm::vec3 specularDir = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
            Ray scatterRay;
            scatterRay.origin = intersect + 0.001f * specularDir;
            scatterRay.direction = specularDir;
            pathSegment.ray = scatterRay;
        }
        else 
        {
            // Diffuse Scattering
            float Pi = 3.1415926f;
            glm::vec3 scatteredDir = calculateRandomDirectionInHemisphere(normal, rng);
            Ray scatterRay;
            scatterRay.origin = intersect + 0.001f * scatteredDir;
            scatterRay.direction = scatteredDir;
            float diffusePdf = glm::dot(normal, scatteredDir) / Pi;
            if (diffusePdf == 0)
            {
                pathSegment.color = glm::vec3(0.0f, 0.0f, 0.0f);
                pathSegment.remainingBounces = -1;
            }
            else
            {
                glm::vec3 f = m.color / Pi;
                pathSegment.color *= (f * glm::abs(glm::dot(scatteredDir, normal))) / diffusePdf;
                pathSegment.ray = scatterRay;
            }
        }
    }
    else
    {
        if (m.hasReflective == 1.0f && m.hasRefractive == 0.0f)
        {
            glm::vec3 specularDir = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
            Ray scatterRay;
            scatterRay.origin = intersect + 0.001f * specularDir;
            scatterRay.direction = specularDir;
            pathSegment.ray = scatterRay;
        }
        else if (m.hasReflective == 0.0f && m.hasRefractive == 0.0f)
        {
            // Diffuse Scattering
            float Pi = 3.1415926f;
            glm::vec3 scatteredDir = calculateRandomDirectionInHemisphere(normal, rng);
            Ray scatterRay;
            scatterRay.origin = intersect + 0.001f * scatteredDir;
            scatterRay.direction = scatteredDir;
            float diffusePdf = glm::dot(normal, scatteredDir) / Pi;
            if (diffusePdf == 0)
            {
                pathSegment.color = glm::vec3(0.0f, 0.0f, 0.0f);
                pathSegment.remainingBounces = -1;
            }
            else
            {
                glm::vec3 f = m.color / Pi;
                pathSegment.color *= (f * glm::abs(glm::dot(scatteredDir, normal))) / diffusePdf;
                pathSegment.ray = scatterRay;
            }
        }
        else if (m.hasRefractive == 1.0f && m.hasReflective == 0.0f)
        {
            // Refractive
            float cosThetaI = glm::dot(normal, pathSegment.ray.direction);
            bool entering = cosThetaI > 0.0f;
            float etaI = entering ? m.indexOfRefraction : 1.0f;
            float etaT = entering ? 1.0f : m.indexOfRefraction;

            glm::vec3 transmissionDir = glm::refract(pathSegment.ray.direction, normal, etaI / etaT);
            float cosThetaT = glm::dot(normal, glm::normalize(transmissionDir));
            if (glm::l2Norm(transmissionDir) <= 0.0001f)
            {
                pathSegment.color = glm::vec3(0.0f);
            }
            else
            {
                pathSegment.color *= m.specular.color;
            }

            Ray scatterRay;
            scatterRay.origin = intersect + 0.001f * transmissionDir;
            scatterRay.direction = transmissionDir;
            pathSegment.ray = scatterRay;
        }
        else if (m.hasRefractive == 1.0f && m.hasReflective == 1.0f)
        {
            float cosThetaI = glm::dot(normal, pathSegment.ray.direction);
            bool entering = cosThetaI > 0.0f;
            float etaI = entering ? m.indexOfRefraction : 1.0f;
            float etaT = entering ? 1.0f : m.indexOfRefraction;
            float R0 = (1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction);
            R0 = R0 * R0;
            float R = R0 + (1 - R0) * powf((1 - glm::abs(cosThetaI)), 5);

            thrust::uniform_real_distribution<float> u01(0, 1);
            float result = u01(rng);

            glm::vec3 transmissionDir = glm::refract(pathSegment.ray.direction, normal, etaI / etaT);
            glm::vec3 specularDir = glm::reflect(pathSegment.ray.direction, normal);


            if (R < result)
            {
                Ray scatterRay;
                scatterRay.origin = intersect + 0.001f * transmissionDir;
                scatterRay.direction = transmissionDir;
                pathSegment.ray = scatterRay;

                if (glm::l2Norm(transmissionDir) <= 0.0001f)
                {
                    pathSegment.color = glm::vec3(0.0f);
                }
                else
                {
                    pathSegment.color *= m.specular.color;
                }
            }
            else
            {
                Ray scatterRay;
                scatterRay.origin = intersect + 0.001f * specularDir;
                scatterRay.direction = specularDir;
                pathSegment.ray = scatterRay;

                if (glm::l2Norm(specularDir) <= 0.0001f)
                {
                    pathSegment.color = glm::vec3(0.0f);
                }
                else
                {
                    pathSegment.color *= m.specular.color;
                }
            }


        }
    }
    
    pathSegment.remainingBounces--;
}
