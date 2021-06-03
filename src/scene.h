#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    Geom Scene::createTriangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3, glm::mat4& transform, int materialId);
    int loadMesh(string objectId);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<glm::vec4> bounding_boxes;
    std::vector<int> obj_starts;
    glm::vec3 globalLight;
    std::vector<Material> materials;
    RenderState state;
};
