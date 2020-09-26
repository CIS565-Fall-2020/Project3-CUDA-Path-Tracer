#include "intersections.h"
#include "BSDF.h"


BSDF::BSDF(const ShadeableIntersection& isect, float eta)
    : worldToTangent(),
    tangentToWorld(),
    normal(isect.surfaceNormal),
    eta(eta),
    numBxDFs(0),
    bxdfs{ nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }
{
    tangentToWorld = glm::mat3(isect.tangent, isect.bitangent, isect.surfaceNormal);
    worldToTangent = glm::inverse(tangentToWorld);
}


