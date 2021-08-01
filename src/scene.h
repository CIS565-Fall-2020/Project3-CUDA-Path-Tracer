#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "octree.h"

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
	void updateBoundingBox(const glm::vec3 &v0, const glm::vec3 &v1, const glm::mat4 &tMat);
	void buildOctreeNode(OctreeNode &node, int depth);

public:
    Scene(string filename);
    ~Scene();

	int loadGltf(string filename);
	void buildOctree();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

	std::vector<OctreeNode> octree;
	std::vector<int> geom_indices;
	glm::vec3 pMin;
	glm::vec3 pMax;
};
