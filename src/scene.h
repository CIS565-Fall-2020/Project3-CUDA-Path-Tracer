#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

//#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	
	void traverseNode(const tinygltf::Model &model, const tinygltf::Node &node, glm::mat4 pTran);

public:
    Scene(string filename);
    ~Scene();

	int loadGltf(string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
