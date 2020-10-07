#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utilities.h"
#include "gltf-loader.h"
#include "glm/gtc/matrix_inverse.hpp"
#include <glm/gtc/matrix_transform.hpp>

#define BOUNDINGBOXINTERSECTIONTEST true

__host__ __device__
glm::mat4 getTansformation(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) 
{
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
}

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) 
{
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
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t)
{
	return r.origin + (t - 0.0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
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
__host__ __device__ float boxIntersectionTest(Geom box,
											  Ray r,
											  glm::vec3& intersectionPoint, 
											  glm::vec3& normal, 
											  bool& outside)
{
	Ray q;
	q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; ++xyz)
	{
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
			float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) 
			{
				tmin = ta;
				tmin_n = n;
			}
			if (tb < tmax) 
			{
				tmax = tb;
				tmax_n = n;
			}
		}
	}

	if (tmax >= tmin && tmax > 0) 
	{
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
__host__ __device__ float sphereIntersectionTest(Geom sphere, 
												 Ray r,
												 glm::vec3& intersectionPoint, 
												 glm::vec3& normal, 
												 bool& outside)
{
	float radius = 0.5f;

	glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	float vDotDirection = glm::dot(rt.origin, rt.direction);
	float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
	if (radicand < 0)
	{
		return -1;
	}

	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;

	float t = 0;
	if (t1 < 0 && t2 < 0)
	{
		return -1;
	}
	else if (t1 > 0 && t2 > 0)
	{
		t = min(t1, t2);
		outside = true;
	}
	else 
	{
		t = max(t1, t2);
		outside = false;
	}

	glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

	intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
	normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
	if (!outside) 
	{
		normal = -normal;
	}

	return glm::length(r.origin - intersectionPoint);
}

/**
 * Test intersection between a ray and a mesh loaded from glTF file. 
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom mesh,
											   Ray r,
											   glm::vec3& intersectionPoint,
											   glm::vec3& normal,
											   bool& outside,
											   int total_meshes,
											   unsigned int* faces,
											   float* vertices,
									           unsigned int* num_faces,
											   unsigned int* num_vertices,
											   float* bbox_verts)
{
	float t = 0;

	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	for (int i = 0, faces_offset = 0, vertices_offset = 0; i < total_meshes; i++)
	{
#if BOUNDINGBOXINTERSECTIONTEST
		Geom bbox_geom;
		bbox_geom.type = GeomType::CUBE;
		glm::vec3 bbox_scale(bbox_verts[i / 6 + 3] - bbox_verts[i / 6 + 0],
							 bbox_verts[i / 6 + 4] - bbox_verts[i / 6 + 1],
						     bbox_verts[i / 6 + 5] - bbox_verts[i / 6 + 2]);

		setGeomTransform(&bbox_geom, mesh.transform * getTansformation(glm::vec3(0), glm::vec3(0), bbox_scale));
		t = boxIntersectionTest(bbox_geom, r, intersectionPoint, normal, outside);
		if (t < 0)
		{
			continue;
		}
#endif // BOUNDINGBOXINTERSECTIONTEST

		int cur_num_faces = num_faces[i];
		int cur_num_vertices = num_vertices[i];
	
		for (int face_idx = 0; face_idx < cur_num_faces / 3; face_idx++)
		{
			unsigned int f0, f1, f2;
			float v0[3], v1[3], v2[3];

			f0 = faces[3 * face_idx + 0 + faces_offset];
			f1 = faces[3 * face_idx + 1 + faces_offset];
			f2 = faces[3 * face_idx + 2 + faces_offset];

			v0[0] = vertices[3 * f0 + 0 + vertices_offset];
			v0[1] = vertices[3 * f0 + 1 + vertices_offset];
			v0[2] = vertices[3 * f0 + 2 + vertices_offset];

			v1[0] = vertices[3 * f1 + 0 + vertices_offset];
			v1[1] = vertices[3 * f1 + 1 + vertices_offset];
			v1[2] = vertices[3 * f1 + 2 + vertices_offset];

			v2[0] = vertices[3 * f2 + 0 + vertices_offset];
			v2[1] = vertices[3 * f2 + 1 + vertices_offset];
			v2[2] = vertices[3 * f2 + 2 + vertices_offset];

			glm::vec3 p0(v0[0], v0[1], v0[2]);
			glm::vec3 p1(v1[0], v1[1], v1[2]);
			glm::vec3 p2(v2[0], v2[1], v2[2]);

			glm::vec3 res;
			bool intersected = glm::intersectRayTriangle(ro, rd, p0, p1, p2, res);
			if (intersected)
			{
				t = res.z;
				outside = false;

				glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
				intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
				normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
		
				return glm::length(r.origin - intersectionPoint);
			}
		}

		faces_offset += cur_num_faces;
		vertices_offset += cur_num_vertices;
	}

	return -1;
}
