#include "octree.h"
#include <algorithm>

OctreeNode::OctreeNode()
{}

OctreeNode::OctreeNode(glm::vec3 &v0, glm::vec3 &v1)
	: bp0(v0), bp1(v1)
{
	center = (v0 + v1) / 2.f;
}

bool OctreeNode::intersectTriangle(const Geom &geom) const {
	glm::vec3 vt0 = geom.v0 - center;
	glm::vec3 vt1 = geom.v1 - center;
	glm::vec3 vt2 = geom.v2 - center;

	glm::vec3 e0 = vt1 - vt0;
	glm::vec3 e1 = vt2 - vt1;
	glm::vec3 e2 = vt0 - vt2;

	glm::vec3 n0 = glm::vec3(1.f, 0.f, 0.f);
	glm::vec3 n1 = glm::vec3(0.f, 1.f, 0.f);
	glm::vec3 n2 = glm::vec3(0.f, 0.f, 1.f);

	float dx = center[0] - bp0[0];
	float dy = center[1] - bp0[1];
	float dz = center[2] - bp0[2];

	vector<glm::vec3> test_axis;
	// Box normals and triangle edges (9 cases)
	for (const auto &v : { e0, e1, e2 }) {
		for (const auto &u : { n0, n1, n2 }) {
			glm::vec3 axis = glm::cross(u, v);
			test_axis.push_back(axis);
		}
	}

	// Box normals as axis (3 cases)
	test_axis.push_back(n0);
	test_axis.push_back(n1);
	test_axis.push_back(n2);

	// triangle normal (1 case)
	test_axis.push_back(geom.normal);

	// iterate through all axis
	for (const auto &axis : test_axis) {
		float p0 = glm::dot(vt0, axis);
		float p1 = glm::dot(vt1, axis);
		float p2 = glm::dot(vt2, axis);
		float r = dx * abs(glm::dot(n0, axis)) + dy * abs(glm::dot(n1, axis)) + dz * abs(glm::dot(n2, axis));
		if (max(max(p0, p1), p2) < -r || min(min(p0, p1), p2) > r) {
			// can be separated
			return false;
		}
	}

	return true;
}

