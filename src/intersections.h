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

// Find intersection of mesh bounding box and ray
__host__ __device__ float meshBboxIntersectionTest(Geom MeshBbox, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    Ray q;
    q.origin = multiplyMV(MeshBbox.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(MeshBbox.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (MeshBbox.bboxMin[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (MeshBbox.bboxMax[xyz] - q.origin[xyz]) / qdxyz;
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
        intersectionPoint = multiplyMV(MeshBbox.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(MeshBbox.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// Find intersection of each triangle and the ray
__host__ __device__ float triangleIntersectionTest(Geom geom, Triangle *triangles, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, glm::vec2 &uv) {
    
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tMin = FLT_MAX;
    int iMin = -1;
    glm::vec3 bary;

    for (int i = geom.startTriangleIndex; i <= geom.endTriangleIndex; i++) {
        Triangle& tri = triangles[i];
        glm::vec3 baryPosition;
        bool isIntersect = glm::intersectRayTriangle(q.origin, q.direction, tri.vert[0], tri.vert[1], tri.vert[2], baryPosition);
        if (!isIntersect) {
            continue;
        }
        float t = baryPosition.z;
        if (t > 0 && t < tMin) {
            iMin = i;
            tMin = t;
            bary = baryPosition;
        }
    }

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(getPointOnRay(q, tMin), 1.0f));
    normal = glm::normalize(multiplyMV(geom.transform, glm::vec4(triangles[iMin].nor, 0.0f)));
    uv = triangles[iMin].uv[0] * (1 - bary.x - bary.y) + triangles[iMin].uv[1] * bary.x + triangles[iMin].uv[2] * bary.y;
    if (glm::dot(normal, r.direction) < 0) {
        outside = true;
    }
    else {
        outside = false;
    }

    return tMin;
}



// Find intersection of mesh bounding box and ray
__host__ __device__ float AABBIntersectionTest(Geom geom, Ray r, glm::vec3 &v1, glm::vec3 &v2) {
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (v1[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (v2[xyz] - q.origin[xyz]) / qdxyz;
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

        return tmin;
    }
    return -1;
}

__host__ __device__
float find(Geom geom, Triangle* triangles, OctreeNode * octreeNodes, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, int n) {
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_min = -1;
    int i_min = -1;
    
    for (int i = 0; i < n; i++) {
        OctreeNode &oc = octreeNodes[i];
        glm::vec3 v1 = oc.bboxMin;
        glm::vec3 v2 = oc.bboxMax;
        
        if (oc.triangleIdx > 0) {
            float ss = 01;
        }
        // If there is no intersection of the ray and AABB
        if (oc.triangleIdx == -2 &&  AABBIntersectionTest(geom, r, oc.bboxMin, oc.bboxMax) == -1) {
            continue;
        }
        // This is an internal node
        // Skip to find its children
        if (oc.triangleIdx == -2) {
            i = oc.childStartIndex - 1;
            continue;

        }
        // This is a leaf empty node
        if (oc.triangleIdx == -1) {
            continue;
        }

        //if (oc.triangleIdx < 0) continue;
        // This is a leaf node, check intersection
        glm::vec3 baryPosition;
        Triangle& tri = triangles[oc.triangleIdx];
        bool isIntersect = glm::intersectRayTriangle(q.origin, q.direction, tri.vert[0], tri.vert[1], tri.vert[2], baryPosition);
        if (!isIntersect) {
            continue;
        }
        float t = baryPosition.z;
        if (t_min == -1 || (t > 0 && t < t_min)) {
            i_min = oc.triangleIdx;
            t_min = t;
        }
        break;
    }

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(getPointOnRay(q, t_min), 1.0f));
    normal = glm::normalize(multiplyMV(geom.transform, glm::vec4(triangles[i_min].nor, 0.0f)));
    if (glm::dot(normal, r.direction) < 0) {
        outside = true;
    }
    else {
        outside = false;
    }

    return t_min;

}

__host__ __device__
float octreeIntersectionTest(Geom geom, Triangle* triangles, OctreeNode* octrees, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, int numObjects) {
    float t = -1;
    //glm::vec3 ss(mesh.children[0]->bottomRightBack);
    //return (AABBIntersectionTest(geom, r, mesh->children[1]->topLeftFront, mesh->children[1]->bottomRightBack));

    return find(geom, triangles,octrees, r, intersectionPoint, normal, outside, numObjects);
}