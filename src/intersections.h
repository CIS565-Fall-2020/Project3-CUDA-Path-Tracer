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


// Function from 461
__host__ __device__
glm::vec2 getCubeUV(const glm::vec3& point) {
    glm::vec3 abs = glm::min(glm::abs(point), 0.5f);
    glm::vec2 UV; // Always offset lower-left corner
    if (abs.x > abs.y && abs.x > abs.z)
    {
        UV = glm::vec2(point.z + 0.5f, point.y + 0.5f) / 3.0f;
        //Left face
        if (point.x < 0)
        {
            UV += glm::vec2(0, 0.333f);
        }
        else
        {
            UV += glm::vec2(0, 0.667f);
        }
    }
    else if (abs.y > abs.x && abs.y > abs.z)
    {
        UV = glm::vec2(point.x + 0.5f, point.z + 0.5f) / 3.0f;
        //Left face
        if (point.y < 0)
        {
            UV += glm::vec2(0.333f, 0.333f);
        }
        else
        {
            UV += glm::vec2(0.333f, 0.667f);
        }
    }
    else
    {
        UV = glm::vec2(point.x + 0.5f, point.y + 0.5f) / 3.0f;
        //Left face
        if (point.z < 0)
        {
            UV += glm::vec2(0.667f, 0.333f);
        }
        else
        {
            UV += glm::vec2(0.667f, 0.667f);
        }
    }
    return UV;
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
__host__ __device__ float boxIntersection(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2& uv, bool &outside) {
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
        uv = getCubeUV(getPointOnRay(q, tmin));
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
__host__ __device__ float sphereIntersection(Geom sphere, Ray r,
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

// helper function that calculates the are of a triangle given its three points
__forceinline__ __host__ __device__ float calculateTriangleArea(glm::vec3& p1, glm::vec3& p2, glm::vec3& p3)
{
    return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
}

// helper function that calculate areas of the sub triangles given a point
__forceinline__ __host__ __device__ glm::vec4 calculateTriangleAreas(Geom& g, glm::vec3& P)
{
    float A_1 = calculateTriangleArea(g.t.pos[0], g.t.pos[1], g.t.pos[2]);
    float A_2 = calculateTriangleArea(g.t.pos[1], g.t.pos[2], P);
    float A_3 = calculateTriangleArea(g.t.pos[0], g.t.pos[2], P);
    float A_4 = calculateTriangleArea(g.t.pos[0], g.t.pos[1], P);
    return glm::vec4(A_1, A_2, A_3, A_4);
}

__forceinline__ __host__ __device__ glm::vec3 calculateTriangleNormal(Geom& g, glm::vec3& P)
{
    glm::vec4 areas = calculateTriangleAreas(g, P);
    return glm::normalize(g.t.nor[0] * areas[1] / areas[0] 
                        + g.t.nor[1] * areas[2] / areas[0] 
                        + g.t.nor[2] * areas[3] / areas[0]);
}

__forceinline__ __host__ __device__ glm::vec2 calculateTriangleUVs(Geom& g, glm::vec3& P)
{
    glm::vec4 areas = calculateTriangleAreas(g, P);
    return glm::clamp(g.t.uv[0] * areas[1] / areas[0] 
                    + g.t.uv[1] * areas[2] / areas[0] 
                    + g.t.uv[2] * areas[3] / areas[0], 0.f, 1.f);
}

__host__ __device__ float triangleIntersection(Geom triangle, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside) {
    // TO DO: Check for bounding box
    if (glm::intersectRayTriangle(r.origin, r.direction, triangle.t.pos[0], triangle.t.pos[1], triangle.t.pos[2], intersectionPoint)) { // was hit
        // solve for normal
        normal = calculateTriangleNormal(triangle, intersectionPoint);
        
        // solve for t
        float t = (glm::dot(normal, triangle.t.pos[0] - r.origin)) / glm::dot(normal, r.direction);

        // solve for uvs
        glm::vec3 point = r.origin + t * r.direction;
        uv = calculateTriangleUVs(triangle, point);

        return t;
    }
    return -1;
}

__host__ __device__ bool boundingBoxIntersection(Ray r, BoundingBox bb)
{
    glm::vec3 invD = 1.0f / (r.direction);
    glm::vec3 t0s = (bb.min - r.origin) * invD;
    glm::vec3 t1s = (bb.max - r.origin) * invD;
    
    glm::vec3 tsmaller = glm::min(t0s, t1s);
    glm::vec3 tbigger = glm::max(t0s, t1s);
    
    float tmin = max(tmin, max(tsmaller[0], max(tsmaller[1], tsmaller[2])));
    float tmax = min(tmax, min(tbigger[0], min(tbigger[1], tbigger[2])));
    
    return (tmin < tmax);
}

// Returns +/- [0, 2]
__host__ __device__
int GetFaceIndex(const glm::vec3& P) {
    int idx = 0;
    float val = -1;
    for (int i = 0; i < 3; i++) {
        if (glm::abs(P[i]) > val) {
            idx = i * glm::sign(P[i]);
            val = glm::abs(P[i]);
        }
    }
    return idx;
}

__host__ __device__
void CoordinateSystem(const glm::vec3& v1, glm::vec3* v2, glm::vec3* v3) {
    if (glm::abs(v1.x) > glm::abs(v1.y))
        *v2 = glm::vec3(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        *v2 = glm::vec3(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
    *v3 = glm::cross(v1, *v2);
}

__host__ __device__
void triangleComputeTBN(Geom& geom,
    glm::vec3 P, glm::vec3* nor, glm::vec3* tan, glm::vec3* bit) {
    *nor = calculateTriangleNormal(geom, P);
    CoordinateSystem(*nor, tan, bit);
}

__host__ __device__
glm::vec4 GetCubeNormal(const glm::vec3& P)
{
    int idx = glm::abs(GetFaceIndex(glm::vec3(P)));
    glm::vec3 N(0, 0, 0);
    N[idx] = glm::sign(P[idx]);
    return glm::vec4(N, 0.f);
}

__host__ __device__
void cubeComputeTBN(glm::mat4 transform, glm::mat4 invT, const glm::vec3 P, glm::vec3* nor, glm::vec3* tan, glm::vec3* bit) {
    *nor = glm::vec3(glm::normalize(invT * GetCubeNormal(P)));
    CoordinateSystem(*nor, tan, bit);
}

__host__ __device__
void computeTBN(Geom& geom, const glm::vec3 P, glm::vec3* nor, glm::vec3* tan, glm::vec3* bit) {
    if (geom.type == CUBE) {
        cubeComputeTBN(geom.transform, geom.invTranspose, P, nor, tan, bit);
    }
    else if (geom.type == TRIANGLE) {
        triangleComputeTBN(geom, P, nor, tan, bit);
    }
}
