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

__host__ __device__ float meshIntersectionTest(Geom mesh, Triangle *triangles, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
	
	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	for (int i = mesh.tIndexStart; i <= mesh.tIndexEnd; i++)
	{
		Triangle triangle = triangles[i];
		glm::vec3 p0 = triangle.vertices[0];
		glm::vec3 p1 = triangle.vertices[1];
		glm::vec3 p2 = triangle.vertices[2];
		glm::vec3 n = glm::cross(p1 - p0, p2 - p0); // normal to check intersection inside a triangle, could opposite to triangle.normal 
		float d = -glm::dot(triangle.normal, p1);

		if (glm::abs(glm::dot(triangle.normal, rd)) < EPSILON) {
			continue;
		}
		float t = (-d - glm::dot(triangle.normal, ro)) / (glm::dot(triangle.normal, rd));
		if (t < 0) {
			continue;
		}
		glm::vec3 p = ro + t * rd;
	
		if (glm::dot(glm::cross(p1 - p0, p - p0), n) < 0) {
			continue;
		}
		if (glm::dot(glm::cross(p2 - p1, p - p1), n) < 0) {
			continue;
		}
		if (glm::dot(glm::cross(p0 - p2, p - p2), n) < 0) {
			continue;
		}
		if (glm::dot(rd, triangle.normal) <= 0) {
			outside = true;
		}
		else {
			outside = false;
		}

		glm::vec3 objspaceIntersection = p;
		intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
		normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(triangle.normal, 0.f)));
		return glm::length(r.origin - intersectionPoint);
	}
	return -1;

}

__host__ __device__ void octreeIntersection(Geom mesh, Triangle *triangles,
	int *sortTringles, OctreeNode_cuda *octreeVector, int nodeIdx,
	Ray &r, glm::vec3 &intersectionPoint, glm::vec3 &normal,
	bool &outside, glm::vec3 &ro, glm::vec3 &rd, float &t) {

	if (nodeIdx == -1) {
		return;
	}
	OctreeNode_cuda root = octreeVector[nodeIdx];
	float tmin = -1e38f;
	float tmax = 1e38f;
	float amin, amax;
	for (int xyz = 0; xyz < 3; ++xyz) {
		if (xyz == 0) {
			amin = root.xmin;
			amax = root.xmax;
		}
		else if (xyz == 1) {
			amin = root.ymin;
			amax = root.ymax;
		}
		else {
			amin = root.zmin;
			amax = root.zmax;
		}
		float rdxyz = r.direction[xyz];
		{
			float t1 = (amin - r.origin[xyz]) / rdxyz;
			float t2 = (amax - r.origin[xyz]) / rdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) {
				tmin = ta;
			}
			if (tb < tmax) {
				tmax = tb;
			}
		}
	}

	if (tmax >= tmin && tmax > 0) {
		if (root.triangleStart != -1 && root.triangleEnd != -1) {
			for (int i = root.triangleStart; i <= root.triangleEnd; i++) {

				int triangleIdx = sortTringles[i];
				Triangle triangle = triangles[triangleIdx];
				glm::vec3 p0 = triangle.vertices[0];
				glm::vec3 p1 = triangle.vertices[1];
				glm::vec3 p2 = triangle.vertices[2];
				glm::vec3 n = glm::cross(p1 - p0, p2 - p0); // normal to check intersection inside a triangle, could opposite to triangle.normal 
				float d = -glm::dot(triangle.normal, p1);

				if (glm::abs(glm::dot(triangle.normal, rd)) < EPSILON) {
					continue;
				}
				float t = (-d - glm::dot(triangle.normal, ro)) / (glm::dot(triangle.normal, rd));
				if (t < 0) {
					continue;
				}
				glm::vec3 p = ro + t * rd;

				if (glm::dot(glm::cross(p1 - p0, p - p0), n) < 0) {
					continue;
				}
				if (glm::dot(glm::cross(p2 - p1, p - p1), n) < 0) {
					continue;
				}
				if (glm::dot(glm::cross(p0 - p2, p - p2), n) < 0) {
					continue;
				}
				if (glm::dot(rd, triangle.normal) <= 0) {
					outside = true;
				}
				else {
					outside = false;
				}

				glm::vec3 objspaceIntersection = p;
				intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
				normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(triangle.normal, 0.f)));
				t = glm::length(r.origin - intersectionPoint);
				return;
			}
		}
		else {
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.tlf, r, intersectionPoint, normal, outside, ro, rd, t);
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.tlb, r, intersectionPoint, normal, outside, ro, rd, t);
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.trf, r, intersectionPoint, normal, outside, ro, rd, t);
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.trb, r, intersectionPoint, normal, outside, ro, rd, t);
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.blf, r, intersectionPoint, normal, outside, ro, rd, t);
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.blb, r, intersectionPoint, normal, outside, ro, rd, t);
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.brf, r, intersectionPoint, normal, outside, ro, rd, t);
			octreeIntersection(mesh, triangles, sortTringles, octreeVector, root.brb, r, intersectionPoint, normal, outside, ro, rd, t);
		}
	}
	return;
}
