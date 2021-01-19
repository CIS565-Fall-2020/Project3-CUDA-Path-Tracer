#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "mesh.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(
        string filename,
        bool usingCulling = false,
        float sceneSize = 40.f);
    ~Scene() {}

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<glm::vec3> triangles; // Triangles of meshes
    bool isWorldSpace; // In which frame of reference are triangles defined
    bool usingCulling; // Whether we are using culling or not

    Octree octree;
    int meshMaterial;
    std::vector<OctreeNodeDevice> result;
    bool prepared;
    std::vector<OctreeNodeDevice> prepareOctree();
    
    RenderState state;
};
