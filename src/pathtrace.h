#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4* pbo, int frame, int iteration, bool sort_by_material, bool cache_first_iteration, bool DOF_on, bool AA_on, bool direct_light);
