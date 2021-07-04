#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

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

// CHECKITOUT
/**
 * Test intersection between a ray and a triangle.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float triangleIntersectionTest(Geom triangle, Ray r, 
        glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {

    glm::vec3 ro = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    glm::vec3 baryCoor(0.f);
    bool intersects = glm::intersectRayTriangle(rt.origin, rt.direction, triangle.v0, triangle.v1, triangle.v2, baryCoor);
    if (!intersects) {
        return -1;
    }
    glm::vec3 baryPos = (1.f - baryCoor.x - baryCoor.y) * triangle.v0 + baryCoor.x * triangle.v1 + baryCoor.y * triangle.v2;
    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(baryPos, 1.f));
    // compute smoothened normal
    bool hasNormalData = (glm::length(triangle.n0) != 0) && (glm::length(triangle.n1) != 0) && (glm::length(triangle.n2) != 0);
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
    if (hasNormalData) {
        n0 = triangle.n0;
        n1 = triangle.n1;
        n2 = triangle.n2;
    }
    else {
        // compute vertex normals with cross product
        n0 = glm::normalize(glm::cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        n1 = glm::normalize(glm::cross(triangle.v0 - triangle.v1, triangle.v2 - triangle.v1));
        n2 = glm::normalize(glm::cross(triangle.v0 - triangle.v2, triangle.v1 - triangle.v2));
    }
    float S = 0.5f * glm::length(glm::cross(triangle.v0 - triangle.v1, triangle.v2 - triangle.v1));
    float S0 = 0.5f * glm::length(glm::cross(triangle.v1 - baryPos, triangle.v2 - baryPos));
    float S1 = 0.5f * glm::length(glm::cross(triangle.v0 - baryPos, triangle.v2 - baryPos));
    float S2 = 0.5f * glm::length(glm::cross(triangle.v0 - baryPos, triangle.v1 - baryPos));
    glm::vec3 smoothNormal = glm::normalize(n0 * S0 / S + n1 * S1 / S + n2 * S2 / S);
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(smoothNormal, 0.f)));
    if (glm::dot(normal, r.direction) > 0) {
        normal = -normal;
        outside = false;
    }
    float t = glm::length(r.origin - intersectionPoint);
    return t;
}

__host__ __device__ float computeTanglecubeSDF(glm::vec3& currPos) {
    float x4 = currPos.x * currPos.x * currPos.x * currPos.x;
    float x2 = currPos.x * currPos.x;
    float y4 = currPos.y * currPos.y * currPos.y * currPos.y;
    float y2 = currPos.y * currPos.y;
    float z4 = currPos.z * currPos.z * currPos.z * currPos.z;
    float z2 = currPos.z * currPos.z;
    return x4 - 5.f * x2 + y4 - 5.f * y2 + z4 - 5.f * z2 + 11.8f;
}
__host__ __device__ void computeTanglecubeNormal(const glm::vec3& p, glm::vec3& nor)
{
    glm::vec3 pxl = p + glm::vec3(-1e-6, 0.0, 0.0);
    float distxl = computeTanglecubeSDF(pxl);

    glm::vec3 pxh = p + glm::vec3(1e-6, 0.0, 0.0);
    float distxh = computeTanglecubeSDF(pxh);

    glm::vec3 pyl = p + glm::vec3(0.0, -1e-6, 0.0);
    float distyl = computeTanglecubeSDF(pyl);

    glm::vec3 pyh = p + glm::vec3(0.0, 1e-6, 0.0);
    float distyh = computeTanglecubeSDF(pyh);

    glm::vec3 pzl = p + glm::vec3(0.0, 0.0, -1e-6);
    float distzl = computeTanglecubeSDF(pzl);

    glm::vec3 pzh = p + glm::vec3(0.0, 0.0, 1e-6);
    float distzh = computeTanglecubeSDF(pzh);

    nor = glm::normalize(glm::vec3(distxh - distxl, distyh - distyl, distzh - distzl));
}

__host__ __device__ float tanglecubeIntersectionTest(Geom tanglecube, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(tanglecube.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(tanglecube.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    glm::vec3 currPos = ro;
    bool intersected = false;
    float threshold = 0.01f;
    float t = 0;
    float s = tanglecube.scale.x;

    while (t < 20.f) {
        float d = s * computeTanglecubeSDF(currPos / s);
        if (fabs(d) < threshold) {
            intersected = true;
            // compute normal
            computeTanglecubeNormal(currPos / s, normal);
            intersectionPoint = multiplyMV(tanglecube.transform, glm::vec4(currPos, 1.f));
            normal = glm::normalize(multiplyMV(tanglecube.invTranspose, glm::vec4(normal, 0.f)));
            break;
        }
        currPos = ro + rd * t;
        t += 0.0005f;
    }
    if (!intersected) {
        t = -1.f;
    }
    else {
        t -= 0.0005f;
    }
    return t;
}

__host__ __device__ float computeBoundBoxSDF(glm::vec3& currPos) {
    glm::vec3 p = currPos;
    glm::vec3 b(1.f, 1.f, 1.f);
    float e = 0.1f;
    p = glm::abs(p) - b;
    glm::vec3 q = glm::abs(p + e) - e;
    return glm::min(glm::min(glm::length(glm::max(glm::vec3(p.x, q.y, q.z), 0.f)) + glm::min(glm::max(p.x, glm::max(q.y, q.z)), 0.f),
        glm::length(glm::max(glm::vec3(q.x, p.y, q.z), 0.f)) + glm::min(glm::max(q.x, glm::max(p.y, q.z)), 0.f)),
        glm::length(glm::max(glm::vec3(q.x, q.y, p.z), 0.f)) + glm::min(glm::max(q.x, glm::max(q.y, p.z)), 0.f));
}
__host__ __device__ void computeBoundBoxNormal(const glm::vec3& p, glm::vec3& nor)
{
    glm::vec3 pxl = p + glm::vec3(-1e-6, 0.0, 0.0);
    float distxl = computeBoundBoxSDF(pxl);

    glm::vec3 pxh = p + glm::vec3(1e-6, 0.0, 0.0);
    float distxh = computeBoundBoxSDF(pxh);

    glm::vec3 pyl = p + glm::vec3(0.0, -1e-6, 0.0);
    float distyl = computeBoundBoxSDF(pyl);

    glm::vec3 pyh = p + glm::vec3(0.0, 1e-6, 0.0);
    float distyh = computeBoundBoxSDF(pyh);

    glm::vec3 pzl = p + glm::vec3(0.0, 0.0, -1e-6);
    float distzl = computeBoundBoxSDF(pzl);

    glm::vec3 pzh = p + glm::vec3(0.0, 0.0, 1e-6);
    float distzh = computeBoundBoxSDF(pzh);

    nor = glm::normalize(glm::vec3(distxh - distxl, distyh - distyl, distzh - distzl));
}

__host__ __device__ float boundBoxIntersectionTest(Geom boundBox, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(boundBox.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(boundBox.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    glm::vec3 currPos = ro;
    bool intersected = false;
    float threshold = 0.01f;
    float t = 0;

    while (t < 20.f) {
        float d = computeBoundBoxSDF(currPos);
        if (fabs(d) < threshold) {
            intersected = true;
            // compute normal
            computeBoundBoxNormal(currPos, normal);
            intersectionPoint = multiplyMV(boundBox.transform, glm::vec4(currPos, 1.f));
            normal = glm::normalize(multiplyMV(boundBox.invTranspose, glm::vec4(normal, 0.f)));
            break;
        }
        currPos = ro + rd * t;
        t += 0.001f;
    }
    if (!intersected) t = -1.f;
    return t;
}
