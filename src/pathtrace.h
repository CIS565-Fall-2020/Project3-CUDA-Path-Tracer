#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, bool octree, int treeDepth, int numGeoms, int totalIter);
void pathtraceFree(bool octree);
void pathtrace(uchar4 *pbo, int frame, int iteration, bool cacheFirstBounce, bool sortByMaterial, bool useMeshBounds);
