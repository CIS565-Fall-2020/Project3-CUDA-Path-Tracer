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
glm::vec3 calculateJitteredRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    // Find a cell it below to. (100 x 100)
    int idx_x = (int)(u01(rng) * 1000);
    int idx_y = (int)(u01(rng) * 1000);
    // Find the (0.f - 1.f) point
    float s = ((float)idx_x + u01(rng)) / 1000.f;
    float t = ((float)idx_y + u01(rng)) / 1000.f;

    float up = sqrt(s); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = t * TWO_PI;

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
glm::vec3 fresnel_evaluate(float etaI, float etaT, float cosThetaI, float cosThetaT) {
// glm::vec3 fresnel_evaluate(float etaI, float etaT, float cosThetaI) {
    cosThetaI = glm::abs(glm::clamp(cosThetaI, -1.f, 1.f));
    cosThetaT = glm::abs(glm::clamp(cosThetaT, -1.f, 1.f));
    // cosThetaT = glm::abs(cosThetaT);

    float sinThetaI = glm::sqrt(glm::max(0.0f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    if (sinThetaT >= 1) {
        return glm::vec3(1.0f);
    }

    cosThetaT = glm::sqrt(glm::max(0.0f, 1 - sinThetaT * sinThetaT));


    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    float ele = (Rparl * Rparl + Rperp * Rperp) / 2;
    return glm::vec3(ele, ele, ele);
}


__host__ __device__ float myFresnel(float cosThetaI, float etaI, float etaT) {
    cosThetaI = glm::abs(glm::clamp(cosThetaI, -1.f, 1.f));
    float sinThetaI = glm::sqrt(glm::max(0.0f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1) {
        return 1.0f;
    }
    float cosThetaT = glm::sqrt(glm::max(0.0f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}


__host__ __device__ float Fresnel(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    bool entering = cosThetaI > 0.0f;
    float etaIb = etaI;
    float etaTb = etaT;
    if (!entering) {
        etaIb = etaT;
        etaTb = etaI;
        cosThetaI = glm::abs(cosThetaI);
    }

    float sinThetaI = glm::sqrt(glm::max(0.0f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaIb / etaTb * sinThetaI;

    if (sinThetaT >= 1) {
        return 1.0f;
    }

    float cosThetaT = glm::sqrt(glm::max(0.0f, 1 - sinThetaT * sinThetaT));

    float Rparl = ((etaTb * cosThetaI) - (etaIb * cosThetaT)) /
        ((etaTb * cosThetaI) + (etaIb * cosThetaT));
    float Rperp = ((etaIb * cosThetaI) - (etaTb * cosThetaT)) /
        ((etaIb * cosThetaI) + (etaTb * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

/*
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        bool outside,
        const Material &m,
        thrust::default_random_engine &rng) {
*/ 
__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const ShadeableIntersection& temp_intersect,
    Geom* geoms,
    const Material& m,
    thrust::default_random_engine& rng) {
    
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.hasReflective == 0.f && m.hasRefractive == 0.f) {

        // glm::vec3 diffu_dir = calculateRandomDirectionInHemisphere(normal, rng);
        glm::vec3 diffu_dir = calculateJitteredRandomDirectionInHemisphere(normal, rng);

        glm::vec3 temp_col = m.color;
        // Texture coloring:
        if (temp_intersect.hit_type == MESH && temp_intersect.hasTexture) {
            Geom& temp_geo_ref = geoms[temp_intersect.geomId];
            float temp_u = temp_intersect.uv[0];
            float temp_v = temp_intersect.uv[1];
            int coordU = (int)(temp_u * (temp_geo_ref.texture_width));
            int coordV = (int)(temp_v * (temp_geo_ref.texture_height));

            if (coordU >= 512) {
                printf("coordU >= 512: %d\n", coordU);
                coordU %= 512;
            }
            if (coordV >= 512) {
                printf("coordV >= 512: %d\n", coordV);
                coordV %= 512;
            }

            int pixel_idx = coordV * temp_geo_ref.texture_width + coordU;
            unsigned int colR = (unsigned int)temp_geo_ref.dev_texture[pixel_idx * 4];
            unsigned int colG = (unsigned int)temp_geo_ref.dev_texture[pixel_idx * 4 + 1];
            unsigned int colB = (unsigned int)temp_geo_ref.dev_texture[pixel_idx * 4 + 2];
            temp_col[0] = (float)colR / 255.f;
            temp_col[1] = (float)colG / 255.f;
            temp_col[2] = (float)colB / 255.f;
        }

        glm::vec3 f = temp_col / 3.1415926f;
        float pdf = glm::dot(normal, diffu_dir) / 3.1415926f;
        if (pdf == 0.f) {
            pathSegment.remainingBounces = 0;
            pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
        }
        else {
            pathSegment.color *= (f * glm::abs(glm::dot(diffu_dir, normal)) / pdf);
            pathSegment.ray.direction = diffu_dir;
            pathSegment.ray.origin = intersect + diffu_dir * 0.01f;
            pathSegment.remainingBounces--;
        }
    }
    else if(m.hasReflective == 1.f && m.hasRefractive == 0.f) {
        glm::vec3 ref_dir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.color;
        pathSegment.ray.direction = ref_dir;
        pathSegment.ray.origin = intersect + ref_dir * 0.0001f;
        pathSegment.remainingBounces--;
    }
    else if (m.hasReflective == 0.f && m.hasRefractive == 1.f) {
        // Fresnel Transmission (BTDF):
        bool entering = false;
        float etaI = 0.f;
        float etaT = 0.f;
        if (outside == false) {
            entering = false;
            etaI = m.indexOfRefraction; // Geo material
            etaT = 1.f; // Air
        }
        else {
            entering = true;
            etaI = 1.f; // Air
            etaT = m.indexOfRefraction; // Geo material
        }
        glm::vec3 refract_dir = glm::refract(pathSegment.ray.direction, normal, etaI / etaT);
        if (glm::l2Norm(refract_dir) <= 0.0001f) {
            pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
        }
        else {
            // Refraction
            float cosThetaI = glm::dot(normal, pathSegment.ray.direction);
            float cosThetaT = glm::dot(normal, refract_dir);
            pathSegment.color *= (m.color * (glm::vec3(1.f, 1.f, 1.f) - fresnel_evaluate(etaI, etaT, cosThetaI, cosThetaT)) * glm::abs(glm::dot(refract_dir, normal)) / glm::abs(cosThetaI));
            // pathSegment.color *= (m.color * (glm::vec3(1.f, 1.f, 1.f) - myFresnel(cosThetaI, etaI, etaT)) * glm::abs(glm::dot(refract_dir, normal)) / glm::abs(cosThetaT));
            pathSegment.ray.direction = refract_dir;
        }
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
        pathSegment.remainingBounces--;
    }
    else if (m.hasReflective == 1.f && m.hasRefractive == 1.f) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        bool entering = false;
        float etaI = 0.f;
        float etaT = 0.f;
        if (outside == false) {
            entering = false;
            etaI = m.indexOfRefraction; // Geo material
            etaT = 1.f; // Air
        }
        else {
            entering = true;
            etaI = 1.f; // Air
            etaT = m.indexOfRefraction; // Geo material
        }

        float tempcosThetaI = glm::dot(normal, pathSegment.ray.direction);
        if (tempcosThetaI > 0.f) {
            printf("cosThetaI > 0.f");
        }

        float cosThetaI = glm::dot(normal, pathSegment.ray.direction);
        float tempFresnel = myFresnel(cosThetaI, etaI, etaT);

        // Fresnel Reflection (BRDF) + Fresnel Transmission (BTDF):
        if (u01(rng) > tempFresnel) {
            // BTDF
            glm::vec3 refract_dir = glm::refract(pathSegment.ray.direction, normal, etaI / etaT);
            if (glm::l2Norm(refract_dir) <= 0.0001f) {
                pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
                // glm::vec3 ref_dir = glm::reflect(pathSegment.ray.direction, normal);
                // pathSegment.ray.direction = ref_dir;
            }
            else {
                // Refraction
                // float cosThetaI = glm::dot(normal, pathSegment.ray.direction);
                float cosThetaT = glm::dot(normal, refract_dir);
                // pathSegment.color *= ((1.f / (1.f - tempFresnel)) * m.color * (glm::vec3(1.f, 1.f, 1.f) - fresnel_evaluate(etaI, etaT, cosThetaI, cosThetaT)) * glm::abs(glm::dot(refract_dir, normal)) / glm::abs(cosThetaT));
                // pathSegment.color *= ((1.f / (1.f - tempFresnel)) * m.color * (glm::vec3(1.f, 1.f, 1.f) - fresnel_evaluate(etaI, etaT, cosThetaI, cosThetaT)));
                pathSegment.color *= m.color;
                // pathSegment.color *= (2.f * m.color * (glm::vec3(1.f, 1.f, 1.f) - fresnel * glm::abs(glm::dot(refract_dir, normal))) / glm::abs(cosThetaT));
                // pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
                pathSegment.ray.direction = refract_dir;
                // printf("Refraction");
            }
        }
        else {
            // BRDF
            // Compute perfect specular reflection direction:
            glm::vec3 ref_dir = glm::reflect(pathSegment.ray.direction, normal);
            // float cosThetaI = glm::dot(normal, pathSegment.ray.direction);
            float cosThetaT = glm::dot(normal, ref_dir);
            // pathSegment.color *= ((1.f / tempFresnel) * m.specular.exponent * m.specular.color * fresnel_evaluate(etaI, etaT, cosThetaI, cosThetaT) * glm::abs(glm::dot(ref_dir, normal)) / glm::abs(cosThetaI));
            // pathSegment.color *= (2.f * m.specular.exponent * m.specular.color * fresnel * glm::abs(glm::dot(ref_dir, normal)) / glm::abs(cosThetaT));
            pathSegment.color *= m.specular.exponent * m.specular.color;
            pathSegment.ray.direction = ref_dir;
        }
        /*if (glm::dot(normal, pathSegment.ray.direction) < 0.f) {
            normal = -normal;
        }*/
        // pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.0001f;
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
        pathSegment.remainingBounces--;
    }
}
