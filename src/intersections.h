#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

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

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMVHomo(glm::mat4 m, glm::vec4 v) {
    glm::vec4 tmp = m * v;

    return glm::vec3(m * v) / tmp.w;
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
__host__ __device__ 
Float boxIntersectionTest(
    const GeomTransform& box, 
    const Ray& r,
    Vertex& itsct, 
    bool &outside) {
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
        itsct.pos = multiplyMVHomo(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        itsct.normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - itsct.pos);
    }
    return -1;
}

__host__ __device__
float errGamma(const int& n) {
    float MachineEpsilon = 5.97e-7; // 2^(-24) bound value for float
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

__host__ __device__ float aabbRayIntersectionTest(const aabbBounds& p, const Ray& r, const vc3& invDir, const vc3& dirIsNeg) {
    // TODO use only aabb to solve(can make it faster maybe?)
    /*Geom box;
    box.geomT.translation = p.bmax / (Float)2.0 + p.bmin / (Float)2.0;
    box.geomT.scale = p.bmax - p.bmin;

    glm::mat4 translationMat = glm::translate(glm::mat4(), box.geomT.translation);
    glm::mat4 rotationMat = glm::mat4(1);
    glm::mat4 scaleMat = glm::scale(glm::mat4(), box.geomT.scale);
    box.geomT.transform = translationMat * rotationMat * scaleMat;
    box.geomT.inverseTransform = glm::inverse(box.geomT.transform);
    box.geomT.invTranspose = glm::inverseTranspose(box.geomT.transform);

    Vertex itsct; bool outside = false;
    return boxIntersectionTest(box.geomT, r, itsct, outside);*/
    constexpr int dim = 3;
    Float tMin[dim];
    Float tMax[dim];

    for (int i = 0; i < dim; i++) {
        if (dirIsNeg[i] == 0) {
            tMin[i] = (p.bmin[i] - r.origin[i]) * invDir[i];
            tMax[i] = (p.bmax[i] - r.origin[i]) * invDir[i] * errGamma(3);
        }
        else {
            tMin[i] = (p.bmax[i] - r.origin[i]) * invDir[i];
            tMax[i] = (p.bmin[i] - r.origin[i]) * invDir[i] * errGamma(3);
        }
    }

    if (tMin[0] > tMax[1] || tMin[1] > tMax[0]) {
        return false;
    }
    tMin[0] = glm::max(tMin[1], tMin[0]);
    /*if (tMin[1] > tMin[0]) {
        tMin[0] = tMin[1];
    }*/
    tMax[0] = glm::min(tMax[1], tMax[0]);
    /*if (tMax[1] < tMax[0]) {
        tMax[0] = tMax[1];
    }*/
    if (tMin[0] > tMax[2] || tMin[2] > tMax[0]) {
        return false;
    }
    tMin[0] = glm::max(tMin[2], tMin[0]);
    tMax[0] = glm::min(tMax[2], tMax[0]);

    return (tMin[0] < r.time) && (r.time > 0);
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
__host__ __device__ float sphereIntersectionTest(
    const GeomTransform& sphere, 
    const Ray& r,
    Vertex& itsct, 
    bool &outside) {
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
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    itsct.pos = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    itsct.normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        itsct.normal *= -1;
    }

    return glm::length(r.origin - itsct.pos);
}

__host__ __device__ Float planeIntersectionTest(
    const GeomTransform& plane,
    const Ray& r,
    Vertex& itsct,
    bool& outside) {
    Float t = -1;

    Ray q;
    q.origin = multiplyMVHomo(plane.inverseTransform, vc4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(plane.inverseTransform, vc4(r.direction, 0.0f)));

    Float local_t = -q.origin.z / q.direction.z;
    if (q.direction.z != 0 && local_t > (Float)0) {
        vc3 p = getPointOnRay(q, local_t);
        bool isInSquare = glm::all(glm::lessThan(glm::abs(vc2(p)), vc2(0.5)));
        if (isInSquare) {
            itsct.pos = multiplyMVHomo(plane.transform, vc4(p, 1.f));
            itsct.normal = glm::normalize(multiplyMV(plane.invTranspose, glm::vec4(vc3(0, 0, 1.), 0.f)));
            itsct.normal *= (-2 * (glm::dot(itsct.normal, r.direction) > 0) + 1); // reverse the normal
            itsct.uv = vc2(0.5) + vc2(p);
            outside = true;
            t = glm::length(r.origin - itsct.pos);
        }
    }
    return t;
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Geom& supp_geom,
    const Ray& r,
    Vertex& itsct
) {
    Ray q;
    q.origin    = multiplyMVHomo(supp_geom.geomT.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(supp_geom.geomT.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t = -1;
    glm::vec3 baryCoor;
    if (glm::intersectRayTriangle(
        q.origin,
        q.direction,
        triangle.v0,
        triangle.v1,
        triangle.v2,
        baryCoor)) {
        t = baryCoor.z;
        // transform to world space
        /*glm::vec3 local_itsct_pos = (1.f - baryCoor.x - baryCoor.y) * triangle.v0 + baryCoor.x * triangle.v1 + baryCoor.y * triangle.v2;
        intersectionPoint = multiplyMVHomo(supp_geom.transform, glm::vec4(local_itsct_pos, 1.0f));*/
        itsct.pos = multiplyMVHomo(supp_geom.geomT.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
        // intepolate normal
        itsct.normal =
            baryCoor.x * triangle.n0 +
            baryCoor.y * triangle.n1 +
            (1.0f - baryCoor.x - baryCoor.y) * triangle.n2;
        itsct.normal = glm::normalize(multiplyMV(supp_geom.geomT.invTranspose, glm::vec4(itsct.normal, 0.0f)));
        // inverse the normal if ray shot in the front of the triangle
        itsct.normal *= 2 * (glm::dot(itsct.normal, r.direction) < 0) - 1;
        itsct.uv = 
            baryCoor.x * triangle.uv0 +
            baryCoor.y * triangle.uv1 +
            (1.0f - baryCoor.x - baryCoor.y) * triangle.uv2;
        return glm::length(r.origin - itsct.pos);
    }
    return -1.0;
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& r,
    Vertex& itsct) {
    // here triangle property is in world space
    // TOCHECK
    float t = -1;

    vc3 baryCoor;
    if (glm::intersectRayTriangle(
        r.origin,
        r.direction,
        triangle.v0,
        triangle.v1,
        triangle.v2,
        baryCoor)) {
        t = baryCoor.z;
        itsct.pos = getPointOnRay(r, t);
        // intepolate normal
        itsct.normal =
            baryCoor.x * triangle.n0 +
            baryCoor.y * triangle.n1 +
            (1.0f - baryCoor.x - baryCoor.y) * triangle.n2;
        // inverse the normal if ray shot in the front of the triangle
        itsct.normal *= 2 * (glm::dot(itsct.normal, r.direction) < 0) - 1;
        itsct.uv =
            baryCoor.x * triangle.uv0 +
            baryCoor.y * triangle.uv1 +
            (1.0f - baryCoor.x - baryCoor.y) * triangle.uv2;
        t = glm::length(r.origin - itsct.pos);
    }

    return t;
}

__host__ __device__ float meshIntersectionTest(
    const Geom &bbox,
    GLTF_Model* models,
    Triangle* triangles,
    const Ray &r,
    Vertex& itsct,
    bool& outside) {
    ///
    /// check bbox then triangles
    /// 
    float t = -1.0f;
#if usebbox
    //bool bbox_outside = true;
    t = boxIntersectionTest(bbox.geomT, r, itsct, outside);
    if (t < 0) {
        return -1;
    }
#else
#endif
    // intersect with triangle
    GLTF_Model cur_model = models[bbox.mesh_idx];
    int start_idx = cur_model.triangle_idx;
    int end_idx = cur_model.triangle_count + start_idx;

    Vertex tmp_itsct;

    float t_min = INFINITY;
    for (int idx = start_idx; idx < end_idx; idx++) {
        const Triangle& cur_triangle = triangles[idx];
        float t_tmp = -1.0f;
        t_tmp = triangleIntersectionTest(
            cur_triangle, 
            cur_model.self_geom,
            r,
            tmp_itsct);
        if (t_tmp > 0.0f && t_tmp < t_min) {
            t_min = t_tmp;
            itsct = tmp_itsct;
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

__host__ __device__ Float primitiveRayIntersectionTest(
    const Primitive& p, 
    const Ray& r, 
    ShadeableIntersection& itsct,
    bool& outside) {
    /// <summary>
    /// 
    /// </summary>
    /// <param name="p"></param>
    /// <param name="r"></param>
    /// <param name="itsct"></param> contains the intersect from previous test
    /// <param name="outside"></param>
    /// <returns></returns>
    Float t = -1.0;
    Vertex v;
    if (p.type == TRIANGLE) {
        t = triangleIntersectionTest(p.triangle, r, v);
    }
    else {
        // primitive like box, sphere
        
        if (p.type == CUBE) {
            t = boxIntersectionTest(p.trans, r, v, outside);
            //printf("hit box translation: %d, %d, th t%d\n", p.trans.translation.x, p.trans.translation.y, t);
        }else if (p.type == SPHERE) {
            t = sphereIntersectionTest(p.trans, r, v, outside);
            //printf("hit sphere translation: %d, %d, with t%d\n", p.trans.translation.x, p.trans.translation.y, t);
        }
        else if (p.type == PLANE) {
            t = planeIntersectionTest(p.trans, r, v, outside);
        }
    }
    if (t > 0) {
        itsct.t = t;
        itsct.materialId = p.materialid;
        itsct.geom_idx = p.geom_idx;
        itsct.vtx = v;
    }
    //printf("t: %f while Intersect t: %f\n", t, itsct.t);
    return t;
}

// Jack12 add intersections helper function here
__global__ void construct_materialIDs(int num_paths, ShadeableIntersection* intersections, int * materialID) {

    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        materialID[path_index] = intersections[path_index].materialId;
    }
}