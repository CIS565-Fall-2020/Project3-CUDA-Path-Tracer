#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, bool octree);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, bool cacheFirstBounce, bool sortByMaterial);
