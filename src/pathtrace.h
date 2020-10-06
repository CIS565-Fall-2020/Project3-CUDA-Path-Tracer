#pragma once

#include <vector>
#include "scene.h"
#include "utilities.h"

constexpr std::size_t log2SqrtNumStratifiedSamples = 5;
constexpr std::size_t sqrtNumStratifiedSamples = 1 << log2SqrtNumStratifiedSamples;
constexpr std::size_t numStratifiedSamples = sqrtNumStratifiedSamples * sqrtNumStratifiedSamples;

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, float stratifiedRange);
void updateStratifiedSamples(const std::vector<StratifiedSampler> &samplers);
