#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "sceneStructs.h"

#define MAX_DEPTH 5

using namespace std;

class OctreeNode {
public:
	glm::vec3 center;
	glm::vec3 bp0; // top left bounding point
	glm::vec3 bp1;
	int childrenIndices[8];
	int geom_idx_start;
	int geom_idx_end; // [start, end)
	glm::mat4 transform;
	glm::mat4 invTransform;
	glm::mat4 invTranspose;

	OctreeNode();
	OctreeNode(glm::vec3 &v0, glm::vec3 &v1);

	bool intersectTriangle (const Geom &geom) const;
};
