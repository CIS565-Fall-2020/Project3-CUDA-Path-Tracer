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
    BoundingBox loadOBJ(string filename, int material_id);
    bool loadTexture(Material& newMaterial, string path, bool bump);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<glm::vec3> texture; // a collection of all the colors of a texture
    RenderState state;
    BoundingBox bounding_box;
};
