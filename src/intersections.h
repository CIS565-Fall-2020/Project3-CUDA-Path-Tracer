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
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(const GeomTransform &box, Ray r, glm::vec3 &normal) {
    Ray q;
    q.origin = box.inverseTransform * glm::vec4(r.origin, 1.0f);
    q.direction = box.inverseTransform * glm::vec4(r.direction, 0.0f);

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
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
        }
        normal = glm::normalize(glm::vec3(glm::transpose(box.inverseTransform) * tmin_n));
        return tmin;
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
__host__ __device__ float sphereIntersectionTest(const GeomTransform &sphere, Ray r, glm::vec3 &normal) {
    constexpr float radius = .5;

    glm::vec3 ro = sphere.inverseTransform * glm::vec4(r.origin, 1.0f);
    glm::vec3 rd = sphere.inverseTransform * glm::vec4(r.direction, 0.0f);

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float sqrd = glm::dot(rt.direction, rt.direction);
    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - sqrd * (glm::dot(rt.origin, rt.origin) - radius * radius);
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm - squareRoot;
    float t2 = firstTerm + squareRoot;

    if (t2 < 0) {
        return -1;
    } else if (t1 < 0) {
        t1 = t2;
    }
    t1 /= sqrd;

    glm::vec3 objspaceIntersection = rt.origin + t1 * rt.direction;
    normal = glm::normalize(glm::vec3(glm::transpose(sphere.inverseTransform) * objspaceIntersection));

    return t1;
}

// glm intersectRayTriangle ignores back-facing triangles, it's copied here and slightly modified to overcome that
__host__ __device__ float triangleIntersectionTest(const GeomTriangle &tri, Ray r, glm::vec2 *bary) {
    glm::vec3 e1 = tri.vertices[1] - tri.vertices[0];
    glm::vec3 e2 = tri.vertices[2] - tri.vertices[0];

    glm::vec3 p = glm::cross(r.direction, e2);

    float f = 1.0f / glm::dot(e1, p);

    glm::vec3 s = r.origin - tri.vertices[0];
    bary->x = f * glm::dot(s, p);
    if (bary->x < 0.0f || bary->x > 1.0f) {
        return -1.0f;
    }

    glm::vec3 q = glm::cross(s, e1);
    bary->y = f * glm::dot(r.direction, q);
    if (bary->y < 0.0f || bary->y + bary->x > 1.0f) {
        return -1.0f;
    }

    return f * glm::dot(e2, q);
}