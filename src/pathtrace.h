#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, const std::vector<glm::vec3>& vertices, const std::vector<glm::vec3>& normals, int numVertices, const Geom& meshBB);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, int samplesPerPixel);
