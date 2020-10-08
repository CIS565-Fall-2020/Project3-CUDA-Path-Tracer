#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "cfg.h"

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

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Geom& supp_geom,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal
) {
    Ray q;
    q.origin    =                multiplyMV(supp_geom.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(supp_geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = -1;
    if (glm::intersectRayTriangle(
        q.origin,
        q.direction,
        triangle.v0,
        triangle.v1,
        triangle.v2,
        intersectionPoint)) {
        t = intersectionPoint.z;
        // transform to world space
        intersectionPoint = multiplyMV(supp_geom.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
        // intepolate normal
        normal =
            intersectionPoint.x * triangle.v1 +
            intersectionPoint.y * triangle.v2 +
            (1.0f - intersectionPoint.x - intersectionPoint.y) * triangle.v0;
        normal = glm::normalize(multiplyMV(supp_geom.invTranspose, glm::vec4(normal, 0.0f)));
    }
    return t;
}

__host__ __device__ float meshIntersectionTest(
    const Geom &bbox,
    GLTF_Model* models,
    Triangle* triangles,
    const Ray &r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside) {
    ///
    /// check bbox then triangles
    /// 
    float t = -1.0f;
#if usebbox
    // got no idea what this would do, but anyway...
    //bool bbox_outside = true;
    t = boxIntersectionTest(bbox, r, intersectionPoint, normal, outside);
    if (t < 0) {
        return -1;
    }
#else
#endif
    // intersect with triangle
    GLTF_Model cur_model = models[bbox.mesh_idx];
    int start_idx = cur_model.triangle_idx;
    int end_idx = cur_model.triangle_count + start_idx;

    glm::vec3 tmp_intersection;
    glm::vec3 tmp_normal;

    float t_min = INFINITY;
    for (int idx = start_idx; idx < end_idx; idx++) {
        const Triangle& cur_triangle = triangles[idx];
        float t_tmp = -1.0f;
        t_tmp = triangleIntersectionTest(
            cur_triangle, 
            cur_model.self_geom,
            r,
            tmp_intersection,
            tmp_normal);
        if (t_tmp > 0.0f && t_tmp < t_min) {
            t_min = t_tmp;
            intersectionPoint = tmp_intersection;
            normal = tmp_normal;
        }
    }
    if (t_min > 0.0 && t_min < INFINITY) {
        t = t_min;
    }
    else {
        t = -1.0f;
    }
    return t;
}

// Jack12 add intersections helper function here
__global__ void construct_materialIDs(int num_paths, ShadeableIntersection* intersections, int * materialID) {

    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        materialID[path_index] = intersections[path_index].materialId;
    }
}