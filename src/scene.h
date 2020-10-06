#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

// Jack12 add gltf support 


using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    // jack12
    int loadGLTFMesh(
        const std::string& file_path,
        const Geom& parent);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<GLTF_Model> gltf_models;

    std::vector<Material> materials;
    RenderState state;
};
