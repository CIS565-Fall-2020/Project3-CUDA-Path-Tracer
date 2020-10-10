#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"


#include "mesh.h"
#include "material.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    
    //int loadMesh(string location, Geom& newGeom);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<Triangle> triangles;
    std::vector<Geom> bboxs;
    int currTrigIdx = 0;
    int currBoxIdx = 0;
    std::vector<example::gltfMesh<float> > gltfMeshes;
    std::vector<example::gltfMaterial> gltfMaterials;
    std::vector<example::gltfTexture> gltfTextures;
};
