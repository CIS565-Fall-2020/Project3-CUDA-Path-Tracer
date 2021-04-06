#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "octree.h"


using namespace std;


class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadGeomFromGLTF(string filename);

    bool loadImage(string filename);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles; // store all the triangles in ths scene
    std::vector<OctreeNode> octrees; // Use octree to store mesh
    std::vector<unsigned char> image;
    RenderState state;
};
