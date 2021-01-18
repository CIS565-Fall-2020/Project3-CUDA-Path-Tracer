#pragma once
#pragma hd_warning_disable
#include "sceneStructs.h"

class FresnelDielectric
{
private:
    __host__ __device__
    static float reflectance(float cosine, float refractIdx) 
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refractIdx) / (1 + refractIdx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }

public:
    __host__ __device__
    static glm::vec3 evaluate(const glm::vec3 wo, const glm::vec3 normal, const float refractIdx, const float rand)
    {
        bool entering = glm::dot(wo, normal) > 0.f;
        float refractRatio = entering ? refractIdx : (1.f / refractIdx);

        float cosTheta = min(glm::dot(-wo, normal), 1.f);
        float sinTheta = sqrt(1.f - cosTheta * cosTheta);

        bool cannotRefract = refractRatio * sinTheta > 1.f;
        glm::vec3 wi;

        if (cannotRefract || reflectance(cosTheta, refractRatio) > rand)
        {
            wi = glm::reflect(wo, normal);
        }
        else
        {
            wi = glm::refract(wo, normal, refractRatio);
        }

        return wi;
    }
};