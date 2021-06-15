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
    bool loadMesh(); 
public:
    Scene(string filename);
    Scene(string filename, string obj_filename); 
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Geom> lights; 
    RenderState state;
    Mesh mesh; 
};
