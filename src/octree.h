#pragma once

#include <glm/glm.hpp>
#include <vector>

#define MAX_DEPTH 4

using namespace std;

class OctreeNode {
public:
	glm::vec3 center;
	glm::vec3 bp0; // top left bounding point
	glm::vec3 bp1;
	int index;
	vector<int> childrenIndices;
	vector<int> geomIndices;

	OctreeNode();
	OctreeNode(glm::vec3 &c, glm::vec3 &v0, glm::vec3 &v1);

	bool intersectTriangle(glm::vec3 &v0, glm::vec3 &v1, glm::vec3 &v2);
	void subdivide();
	void fillChildrenIndices();
};
