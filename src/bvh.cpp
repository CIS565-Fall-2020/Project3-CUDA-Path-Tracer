#include "bvh.h"

// << BVHBuildNode Public Methods >>

void BVHBuildNode::InitLeaf(int first, int n, const aabbBounds& b) {
    firstPrimOffset = first;
    nPrimitives = n;
    bounds = b;
    children[0] = children[1] = nullptr;
}

void BVHBuildNode::InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1) {
    children[0] = c0;
    children[1] = c1;
    bounds = geometry::bbUnion(c0->bounds, c1->bounds);
    splitAxis = axis;
    nPrimitives = 0;
}
