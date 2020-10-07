#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "sceneStructs.h"

#define MAX_DEPTH 4

using namespace std;

class OctreeNode {
public:
	glm::vec3 center;
	glm::vec3 bp0; // top left bounding point
	glm::vec3 bp1;
	vector<int> childrenIndices;
	vector<int> geomIndices;

	OctreeNode();
	OctreeNode(glm::vec3 &v0, glm::vec3 &v1);

	bool intersectTriangle (const Geom &geom) const;
	float intersectRay(const Ray &ray, 
		glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) const;
};
