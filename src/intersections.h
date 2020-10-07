#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/transform.hpp>

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
    return -1.f;
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
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
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
    }
    else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    }
    else {
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

__host__ __device__ float meshTriangleIntersectionTest(Geom obj, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal) {
    float min_tri_t = FLT_MAX;
    glm::vec3 tmp_tri_intersect;
    glm::vec3 tmp_tri_normal;
    glm::vec3 min_tri_intersect;
    glm::vec3 min_tri_normal;

    // to object space
    glm::vec3 ro = multiplyMV(obj.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(obj.inverseTransform, glm::vec4(r.direction, 0.0f)));

    for (int j = 0; j < obj.triangles_size; j++) {
        Triangle& tri = obj.dev_triangles[j];

        // check if there is an intersection
        bool did_isect = glm::intersectRayTriangle(ro, rd, tri.v1, tri.v2, tri.v3, tmp_tri_intersect);

        if (did_isect) {
            // to world space
            tmp_tri_normal = tri.normal;
            tmp_tri_intersect = ro + rd * tmp_tri_intersect.z;

            // local to world
            tmp_tri_intersect = multiplyMV(obj.transform, glm::vec4(tmp_tri_intersect, 1.f));
            float tmp_tri_t = glm::length(r.origin - tmp_tri_intersect);
            if (tmp_tri_t < min_tri_t) {
                min_tri_intersect = tmp_tri_intersect;
                min_tri_normal = tmp_tri_normal;
                min_tri_t = tmp_tri_t;
            }
        }
    }
    if (min_tri_t > 0.0f) {
        intersectionPoint = min_tri_intersect;
        normal = min_tri_normal;
        return min_tri_t;
    }
    else {
        return -1.f;
    }
}
__host__ __device__ bool rectangleIntersectionTest(const Ray r, Geom obj) {
    glm::vec3 min = multiplyMV(obj.transform, glm::vec4(obj.min, 1.0f));
    glm::vec3 max = multiplyMV(obj.transform, glm::vec4(obj.max, 1.0f));

    float tmin = (min.x - r.origin.x) / r.direction.x;
    float tmax = (max.x - r.origin.x) / r.direction.x;

    if (tmin > tmax) {
        float temp = tmin;
        tmin = tmax;
        tmax = temp;
    }

    float tymin = (min.y - r.origin.y) / r.direction.y;
    float tymax = (max.y - r.origin.y) / r.direction.y;

    if (tymin > tymax) {
        float temp = tymin;
        tymin = tymax;
        tymax = temp;
    }

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (min.z - r.origin.z) / r.direction.z;
    float tzmax = (max.z - r.origin.z) / r.direction.z;

    if (tzmin > tzmax) {
        float temp = tzmin;
        tzmin = tzmax;
        tzmax = temp;
    }

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}

__host__ __device__ float sphereSDF(glm::vec3& p, Geom sdf, float radius) {
    float dist = glm::length(multiplyMV(sdf.inverseTransform, glm::vec4(p, 1.0f))) - radius;
    return dist * glm::min(glm::min(sdf.scale.x, sdf.scale.y), sdf.scale.z);
}

__host__ __device__ float sdfUnion(float a, float b) {
    return glm::min(a, b);
}

__host__ __device__ float sdf1(glm::vec3& p, Geom sdf) {
    return sphereSDF(p, sdf, 0.5f);
}

__host__ __device__ glm::vec3 sdfNormal(glm::vec3& p, Geom sdf) {
    glm::vec3 xOffset(0.0001f, 0.0, 0.0);
    glm::vec3 yOffset(0.0, 0.0001f, 0.0);
    glm::vec3 zOffset(0.0, 0.0, 0.0001f);
    glm::vec3 normal(1.f);
    if (sdf.type == SDF1) {
        normal = glm::vec3(sdf1(p + xOffset, sdf) - sdf1(p - xOffset, sdf),
            sdf1(p + yOffset, sdf) - sdf1(p - yOffset, sdf),
            sdf1(p + zOffset, sdf) - sdf1(p - zOffset, sdf));
    }

    return glm::normalize(normal);
}

__host__ __device__ float sdfIntersection(Geom sdf, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal) {
    float dist = 0.f;

    while (dist < 50.f) {
        float curr_dist = sdf1(r.origin + (r.direction * dist), sdf);
        if (glm::abs(curr_dist) < 0.001f) {
            break;
        }
        dist += curr_dist;
    }

    if (dist > 50.f) {
        return -1.f;
    }
    else {
        intersectionPoint = r.origin + r.direction * dist;
        normal = sdfNormal(intersectionPoint, sdf);
        return dist;
    }
}