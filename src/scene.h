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
    int loadObj(string filename, int materialid, glm::vec3 translation, glm::vec3 rotation,
        glm::vec3 scale, glm::mat4 transform, glm::mat4 inverseTransform, glm::mat4 invTranspose);

public:
    Scene(string filename);
    ~Scene();

    std::vector<std::vector<Triangle>> triangles;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
