#pragma once
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "gltf-loader.h"

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
    std::vector<Material> materials;
    RenderState state;

    int getMeshesSize() const;

    // class members for gltf meshes
    std::vector<gltf::Mesh<float>> meshes;
    std::vector<unsigned int> faces_per_mesh;
    std::vector<unsigned int> vertices_per_mesh;

    int total_faces;
    int total_vertices;
};
