#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "sceneStructs.h"

using namespace std;

class OctreeNode {
public:
	glm::vec3 center;
	glm::vec3 bp0; // top left bounding point
	glm::vec3 bp1;
	vector<OctreeNode*> children;
	int geomIdx;

	OctreeNode();
	OctreeNode(glm::vec3 &c, glm::vec3 &v0, glm::vec3 &v1);

};
