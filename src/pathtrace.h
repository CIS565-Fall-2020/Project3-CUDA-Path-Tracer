#pragma once

#include <vector>
#include "scene.h"
#include "utilities.h"

void pathtraceInit(Scene *scene, int sqrtStratifiedSamples);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, int directLight, int numLights);
void updateStratifiedSamples(
	const std::vector<std::vector<IntersectionSample>> &samplers, const std::vector<CameraSample> &camSamples
);
