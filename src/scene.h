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
    int loadMesh(string filename, glm::vec3& min_bound, glm::vec3& max_bound);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<std::vector<Triangle>> meshes;
    int totalTriangles;  // total triangle numbers
    std::vector<int> idxOfEachMesh;   // start index of the triangles of each mesh 
    std::vector<int> endIdxOfEachMesh;
    RenderState state;
};
