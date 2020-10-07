#include "octree.h"

OctreeNode::OctreeNode()
{}

OctreeNode::OctreeNode(glm::vec3 &v0, glm::vec3 &v1)
	: bp0(v0), bp1(v1)
{
	center = (v0 + v1) / 2.f;
}

bool OctreeNode::intersectTriangle(const Geom &geom) const {

}

