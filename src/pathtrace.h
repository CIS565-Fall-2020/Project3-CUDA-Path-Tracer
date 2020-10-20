#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void meshInit(Scene* scene);
void meshFree(Scene* scene);
void pathtraceFree(Scene* scene);
// void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void directlight_pathtrace(uchar4* pbo, int frame, int iter);
void denoise(uchar4* pbo, int iter, int filter_size);
void gauss_denoise(uchar4* pbo, int iter, int filter_size);