#pragma once

#include "sceneStructs.h"

struct BVHBuildNode {
    // << BVHBuildNode Public Methods >>
    void InitLeaf(int first, int n, const aabbBounds& b);

    void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1);

    aabbBounds bounds;
    BVHBuildNode* children[2];
    // store primitives from [firstPrimOffset, firstPrimOffset + nPrimitives)
    // nPrimitives == 0: internal flattened node
    int splitAxis, firstPrimOffset, nPrimitives; 
    
};


struct LinearBVHNode {
    aabbBounds bounds;
    union {
        int primitivesOffset;    // leaf
        int secondChildOffset;   // interior
    };
    uint16_t nPrimitives;  // 0 -> interior node
    uint8_t axis;          // interior node: xyz
    uint8_t pad[1];        // ensure 32 byte total size
};