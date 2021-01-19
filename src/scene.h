#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "mesh.h"
#include "material.h"
#include "octree.h"



using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> lights;
    std::vector<OctBox> OctreeBox;
    std::vector<Material> materials;
    std::vector<std::vector<example::Mesh<float>>> meshes;
    std::vector<example::Material> gltfMaterials;
    std::vector<example::Texture> gltfTextures;
    std::vector<BoundingBox> boundingBoxes;
    Octree octree;
    RenderState state;
    int faceCount;
    int posCount;
    int meshCount;
};
