#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ 
float triArea(glm::vec3 pos0, glm::vec3 pos1, glm::vec3 pos2)
{
    float aSide = glm::length(pos1 - pos0);
    float bSide = glm::length(pos2 - pos0);
    float cSide = glm::length(pos1 - pos2);

    float p = (aSide + bSide + cSide) / 2.0f;
    float S = sqrtf(p * (p - aSide) * (p - bSide) * (p - cSide));

    return S;
}

__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside,
    glm::vec3& tangent, glm::vec3& bitangent,
    float* meshPos, float* meshNor, int* meshIdx, float* meshUV, int faceNum, int offset, int posOffset) 
{
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tMin = FLT_MAX;
    glm::vec3 curInter = glm::vec3(0.0f);
    glm::vec3 curNormal = glm::vec3(0.0f);
    glm::vec3 curTangent = glm::vec3(0.0f);
    glm::vec3 curBitangent = glm::vec3(0.0f);
    bool isInter = false;

    for (int i = 0; i < faceNum; i++) 
    {
        int index0 = meshIdx[i * 3 + 3 * offset] + 3 * offset;
        int index1 = meshIdx[i * 3 + 3 * offset + 1] + 3 * offset;
        int index2 = meshIdx[i * 3 + 3 * offset + 2] + 3 * offset;

        int posIndex0 = meshIdx[i * 3 + 3 * offset] + posOffset / 3;
        int posIndex1 = meshIdx[i * 3 + 3 * offset + 1] + posOffset / 3;
        int posIndex2 = meshIdx[i * 3 + 3 * offset + 2] + posOffset / 3;

        glm::vec3 pos0 = glm::vec3(meshPos[3 * posIndex0], meshPos[3 * posIndex0 + 1], meshPos[3 * posIndex0 + 2]);
        glm::vec3 pos1 = glm::vec3(meshPos[3 * posIndex1], meshPos[3 * posIndex1 + 1], meshPos[3 * posIndex1 + 2]);
        glm::vec3 pos2 = glm::vec3(meshPos[3 * posIndex2], meshPos[3 * posIndex2 + 1], meshPos[3 * posIndex2 + 2]);

        glm::vec3 nor0 = glm::vec3(meshNor[3 * index0], meshNor[3 * index0 + 1], meshNor[3 * index0 + 2]);
        glm::vec3 nor1 = glm::vec3(meshNor[3 * index1], meshNor[3 * index1 + 1], meshNor[3 * index1 + 2]);
        glm::vec3 nor2 = glm::vec3(meshNor[3 * index2], meshNor[3 * index2 + 1], meshNor[3 * index2 + 2]);        
        glm::vec3 interP = glm::vec3(0.0f);

        glm::vec2 uv0 = glm::vec2(meshUV[2 * index0], meshUV[2 * index0 + 1]);
        glm::vec2 uv1 = glm::vec2(meshUV[2 * index1], meshUV[2 * index1 + 1]);
        glm::vec2 uv2 = glm::vec2(meshUV[2 * index2], meshUV[2 * index2 + 1]);

        bool triIntersect = glm::intersectRayTriangle(ro, rd, pos0, pos1, pos2, interP);

        glm::vec3 deltaPos1 = pos1 - pos0;
        glm::vec3 deltaPos2 = pos2 - pos0;
        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;

        glm::vec3 tangent = (deltaUV2.y * deltaPos1 - deltaUV1.y * deltaPos2)
            / (deltaUV2.y * deltaUV1.x - deltaUV1.y * deltaUV2.x);

        glm::vec3 bitangent = (deltaPos2 - deltaUV2.x * tangent) / deltaUV2.y;

        if (!triIntersect)
            continue;

        float t = interP.z;
        interP = ro + t * rd;
        glm::vec3 nor = glm::normalize(glm::cross(pos2 - pos0, pos1 - pos0));

        float s2 = triArea(interP, pos0, pos1);
        float s1 = triArea(interP, pos0, pos2);
        float s0 = triArea(interP, pos1, pos2); 
        float s = triArea(pos0, pos1, pos2);

        

        glm::vec2 interUV = s0 / s * uv0 + s1 / s * uv1 + s2 / s * uv2;


        if (t < tMin && t > 0.0f) 
        {
            tMin = t;
            curInter = interP + 0.001f * nor;
            curNormal = nor;
            curTangent = glm::normalize(tangent);
            curBitangent = glm::normalize(bitangent);
            isInter = true;
            uv = interUV;
        }
    }

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(curInter, 1.f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(curNormal, 0.f)));

    if (glm::dot(normal, intersectionPoint - r.origin) > 0.0f) 
    {
        normal = -normal;
    }
        

    outside = true;

    if (isInter)
        return glm::length(r.origin - intersectionPoint);
    else
        return -1;
}

__host__ __device__ float Clamp(float val, float low, float high) {
    if (val < low) return low;
    else if (val > high) return high;
    else return val;
}

__host__ __device__ float frDielectric(float cosThetaI, float cosThetaT, float etaI, float etaT, bool entering)
{
    cosThetaI = Clamp(cosThetaI, -1.0, 1.0);
    if (!entering)
    {
        cosThetaI = -cosThetaI;
        cosThetaT = -cosThetaT;
    }


    //float cosThetaT = std::sqrt(std::fmax(0.0f, 1 - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));

    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

