#pragma once
#include "glm/glm.hpp"

struct ShadeableIntersection;

enum BxDFType {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR |
    BSDF_REFLECTION | BSDF_TRANSMISSION,
};


class BxDF {
public:
    virtual ~BxDF() { }
    BxDF(BxDFType type) : type(type) { }
    bool MatchesFlags(BxDFType t) const {
        return (type & t) == type;
    }
    virtual glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) const = 0;
    virtual glm::vec3 Sample_f(const glm::vec3& wo, glm::vec3* wi,
        const glm::vec2& sample, float* pdf,
        BxDFType* sampledType = nullptr) const;
    virtual glm::vec3 rho(const glm::vec3& wo, int nSamples,
        const glm::vec2* samples) const;
    virtual glm::vec3 rho(int nSamples, const glm::vec2* samples1,
        const glm::vec2* samples2) const;
    virtual float Pdf(const glm::vec3& wi, const glm::vec3& wo) const;

    const BxDFType type;

};


class BSDF
{
public:
    BSDF(const ShadeableIntersection& isect, float eta = 1);

    ~BSDF();

    void Add(BxDF* b) { if (numBxDFs < MaxBxDFs) { bxdfs[numBxDFs++] = b; } }

    glm::vec3 f(const glm::vec3& woW, const glm::vec3& wiW, BxDFType flags = BSDF_ALL) const;

    glm::vec3 Sample_f(const glm::vec3& woW, glm::vec3* wiW, const glm::vec2& xi,
        float* pdf, BxDFType type = BSDF_ALL,
        BxDFType* sampledType = nullptr) const;

    float Pdf(const glm::vec3& woW, const glm::vec3& wiW,
        BxDFType flags = BSDF_ALL) const;

    // Compute the number of BxDFs that match the input flags.
    int BxDFsMatchingFlags(BxDFType flags) const;

    void UpdateTangentSpaceMatrices(const glm::vec3& n, const glm::vec3& t, const glm::vec3 b);

    glm::mat3 worldToTangent; // Transforms rays from world space into tangent space,
                              // where the surface normal is always treated as (0, 0, 1)
    glm::mat3 tangentToWorld; // Transforms rays from tangent space into world space.
                              // This is the inverse of worldToTangent (incidentally, inverse(worldToTangent) = transpose(worldToTangent))

    glm::vec3 normal;          // May be the geometric normal OR the shading normal at the point of intersection.
                              // If the Material that created this BSDF had a normal map, then this will be the latter.

    float eta; // The ratio of indices of refraction at this surface point. Irrelevant for opaque surfaces.

private:
    int numBxDFs; // How many BxDFs this BSDF currently contains (init. 0)
    const static int MaxBxDFs = 8; // How many BxDFs a single BSDF can contain
    BxDF* bxdfs[MaxBxDFs]; // The collection of BxDFs contained in this BSDF
};