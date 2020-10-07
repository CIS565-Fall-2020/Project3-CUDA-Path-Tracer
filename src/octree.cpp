#include "octree.h"

OctreeNode::OctreeNode()
{}

OctreeNode::OctreeNode(glm::vec3 &c, glm::vec3 &v0, glm::vec3 &v1)
	: center(c), bp0(v0), bp1(v1)
{}

