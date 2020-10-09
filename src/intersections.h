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

__host__ __device__ float triangleIntersectionTest(Geom triangle, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
   /* Attempted initially to un-transform ray to local triangle space, but
      had some weird errors.

   glm::vec3 ro = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 p1 = triangle.tri.point1.pos;
    glm::vec3 p2 = triangle.tri.point2.pos;
    glm::vec3 p3 = triangle.tri.point3.pos; */

    glm::vec3 p1 = glm::vec3(triangle.transform * glm::vec4(triangle.tri.point1.pos, 1.0f));
    glm::vec3 p2 = glm::vec3(triangle.transform * glm::vec4(triangle.tri.point2.pos, 1.0f));
    glm::vec3 p3 = glm::vec3(triangle.transform * glm::vec4(triangle.tri.point3.pos, 1.0f));

    // Barycentric output is (barycentric coordinate 1, barycentric coordinate 2, t)
    glm::vec3 ret;
    if (!glm::intersectRayTriangle(r.origin, r.direction, p1, p2, p3, ret)) {
        return -1.0f;
    }
    
    float baryz = 1.0f - ret.x - ret.y;
    intersectionPoint = ret.x * p1 + ret.y * p2 + baryz * p3;
    normal = glm::normalize(glm::cross(p2 - p1, p3 - p1));
    //normal = ret.x * triangle.tri.point1.nor + ret.y * triangle.tri.point2.nor * baryz * triangle.tri.point3.nor;
    //normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(normal, 0.0f)));

    return ret.z;
}


////////////////////////////
//  IMPLICIT SURFACES
////////////////////////////

#define MAX_STEPS 1000
#define CUTOFF 20.0f
#define I_EPSILON 0.005f
#define STEP 0.05f
#define N_EPSILON 0.0001f

__host__ __device__ float torusFunction(Geom surface, glm::vec3 p) {
    if (surface.implicit.sdf) {
        glm::vec2 t(1.0, 0.5);
        glm::vec2 q(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
        return glm::length(q) - t.y;
    }

    return -1.0f;
}

__host__ __device__ float tanglecubeFunction(glm::vec3 p) {
    float x2 = p.x * p.x,
        y2 = p.y * p.y,
        z2 = p.z * p.z,
        x4 = x2 * x2,
        y4 = y2 * y2,
        z4 = z2 * z2;
    return x4 - 5.f * x2 + y4 - 5.f * y2 + z4 - 5.f * z2 + 11.8f;
}

__host__ __device__ glm::vec3 tanglecubeNormal(glm::vec3 p) {
    float nx = tanglecubeFunction(glm::vec3(p.x + N_EPSILON, p.y, p.z))
             - tanglecubeFunction(glm::vec3(p.x - N_EPSILON, p.y, p.z));
    float ny = tanglecubeFunction(glm::vec3(p.x, p.y + N_EPSILON, p.z))
             - tanglecubeFunction(glm::vec3(p.x, p.y - N_EPSILON, p.z));
    float nz = tanglecubeFunction(glm::vec3(p.x, p.y, p.z + N_EPSILON))
             - tanglecubeFunction(glm::vec3(p.x, p.y, p.z - N_EPSILON));
    return glm::normalize(glm::vec3(nx, ny, nz));
}

// Source of SDFs: https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
__host__ __device__ float cappedConeSDF(glm::vec3 p, float h, float r1, float r2) {
    glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y);
    glm::vec2 k1 = glm::vec2(r2, h);
    glm::vec2 k2 = glm::vec2(r2 - r1, 2.0f * h);
    glm::vec2 ca = glm::vec2(q.x - glm::min(q.x, (q.y < 0.f) ? r1 : r2), glm::abs(q.y) - h);
    glm::vec2 cb = q - k1 + k2 * glm::clamp(glm::dot(k1 - q, k2) / glm::dot(k2, k2), 0.f, 1.f);
    float s = (cb.x < 0.f && ca.y < 0.f) ? -1.f : 1.f;
    return s * glm::sqrt(glm::min(glm::dot(ca, ca), glm::dot(cb, cb)));
}

__host__ __device__ float twistFunction(glm::vec3 p) {
    const float k = 2.5f; // twist amount
    float c = glm::cos(k * p.y);
    float s = glm::sin(k * p.y);
    glm::mat2 m = glm::mat2(c, -s, s, c);
    glm::vec3 q = glm::vec3(m * glm::vec2(p.x, p.z), p.y);
    return cappedConeSDF(q, 1.0, 1., 0.5);
}

__host__ __device__ glm::vec3 twistNormal(glm::vec3 p) {
    float nx = twistFunction(glm::vec3(p.x + N_EPSILON, p.y, p.z))
             - twistFunction(glm::vec3(p.x - N_EPSILON, p.y, p.z));
    float ny = twistFunction(glm::vec3(p.x, p.y + N_EPSILON, p.z))
             - twistFunction(glm::vec3(p.x, p.y - N_EPSILON, p.z));
    float nz = twistFunction(glm::vec3(p.x, p.y, p.z + N_EPSILON))
             - twistFunction(glm::vec3(p.x, p.y, p.z - N_EPSILON));
    return glm::normalize(glm::vec3(nx, ny, nz));
}

__host__ __device__ float implicitSurfaceIntersectionTest(Geom surface, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    float t = 0.0f;
    float scale = surface.scale.x;
    glm::vec3 p;
    for (int i = 0; i < MAX_STEPS; i++) {
        // March point by t.
        p = r.origin + t * r.direction;
        // transform point
        glm::vec3 localPoint = glm::vec3(surface.inverseTransform * glm::vec4(p, 1.0f));
        float dist = 1e38f;
        switch (surface.implicit.type) {
        case TANGLECUBE:
            dist = scale * tanglecubeFunction(localPoint / scale);
        case TWIST:
            dist = scale * twistFunction(localPoint / scale);
        }

        if (dist < I_EPSILON) {
             t -= surface.implicit.shadowEpsilon;
             intersectionPoint = r.origin + t * r.direction;
             switch (surface.implicit.type) {
             case TANGLECUBE:
                 normal = tanglecubeNormal(localPoint / scale);
                 break;
             case TWIST:
                 normal = twistNormal(localPoint / scale);
                 break;
             }
             return t;
         }

         if (surface.implicit.sdf) {
             t += dist;
         } else {
             t += glm::min(STEP, dist);
         }

         if (t >= CUTOFF) {
             return -1.0f;
         }
    }
    
    return -1.0f;
}