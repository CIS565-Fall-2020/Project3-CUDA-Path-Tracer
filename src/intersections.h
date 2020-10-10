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

__host__ __device__ float meshIntersectionTest(Mesh* mesh, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    float dist = -1.f; 
    glm::vec3 	baryPosition(0.f);
    float min_dist = FLT_MAX; 
    bool success = false; 
    for (Triangle t : mesh->triangles)
    {
        //Ray Triangle Intersection 
        //https://glm.g-truc.net/0.9.0/api/a00162.html#a6ce58ac1371605381abb3e00cfe36d78
        success = glm::intersectRayTriangle(r.origin, r.direction, t.vert1, t.vert2, t.vert3, baryPosition); 
        if (success)
        {
            dist = glm::length(baryPosition - r.origin); 
            intersectionPoint = getPointOnRay(r, dist);
            normal = glm::normalize(glm::cross(t.vert1 - t.vert3, t.vert1 - t.vert2));
            if (glm::dot(r.origin, normal) < 0)
            {
                outside = true; 
            }
        }
    }
    return dist; 
}

__host__ __device__ float trianglesIntersectionTest(Geom mesh, Triangle* triangles, int num_triangles, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    float t = -1.f, tmin = FLT_MAX;
    glm::vec3 baryPosition(0.f);
    //bool success = false;
    Ray rt; 
    rt.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.f));
    rt.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.f)));

    //Loop over all triangles in the mesh 
    for (int i = 0; i < num_triangles; ++i)
    {
        Triangle& triangle = triangles[i];
        //Ray Triangle Intersection 
        //glm::vec3 currvert1 = triangle.vert1;
        //glm::vec3 currvert2 = triangle.vert2;
        //glm::vec3 currvert3 = triangle.vert3;
        //glm::vec3 rayoriginOrig = r.origin;
        //glm::vec3 raydirOrig = r.direction;
        bool success = glm::intersectRayTriangle(rt.origin, rt.direction, triangle.vert1, triangle.vert2, triangle.vert3, baryPosition);

        //Barycentric interpolation 
        //t = baryPosition.z;
        //if (t < 0)
        //{
        //    return tmin; 
        //}
        //glm::vec3 p = rt.origin + t * rt.direction; 

        //float s = 0.5 * glm::length(glm::cross(triangle.vert1 - triangle.vert2, triangle.vert1 - triangle.vert3)); 
        //float s1 = 0.5 * glm::length(glm::cross(p - triangle.vert2, p - triangle.vert3)) / s; 
        //float s2 = 0.5 * glm::length(glm::cross(p - triangle.vert3, p - triangle.vert1)) / s;
        //float s3 = 0.5 * glm::length(glm::cross(p - triangle.vert1, p - triangle.vert2)) / s;
        //float sum = s1 + s2 + s3; 

        //if (s1 >= 0 && s1 <= 1 
            //&& s2 >= 0 && s2 <= 1 
            //&& s3 >= 0 && s3 <= 1 
            //&& (sum - 1.f) < EPSILON)
        if(success)
        {
            //Point lies inside triangle 
            glm::vec3 isect = getPointOnRay(rt, t); 
            glm::vec3 isect_local = multiplyMV(mesh.transform, glm::vec4(isect, 1.f)); 
            
            //Find t as distance from ray to the intersection on triangle 
            t = glm::length(r.origin - isect_local); 
            if (t > 0.f && tmin > t)
            {
                //Set min t to be the current t 
                tmin = t;
                
                //Get intersection and normal on mesh 
                intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(rt, tmin), 1.0f));

                //Find the surface normal of the triangle 
                glm::vec3 tri_normal = glm::normalize(glm::cross(triangle.vert1 - triangle.vert3, triangle.vert1 - triangle.vert2));
                normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tri_normal, 0.f))); 
                
                if (glm::dot(rt.origin, normal) < 0)
                {
                    outside = true;
                }
                else
                {
                    outside = false; 
                    normal = -normal; //invert the normal 
                }
            }
        }
    }
    return glm::length(r.origin - intersectionPoint);
}

__device__ void barycentric_interpolation(glm::vec3 ray_origin , glm::vec3 ray_dir, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3* baryPosition)
{

}