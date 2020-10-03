#pragma once

#include <vector>
#include "scene.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
