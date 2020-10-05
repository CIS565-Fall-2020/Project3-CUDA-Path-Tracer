#include "tiny_gltf.h"

#include <string>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <vector>
#include "sceneStructs.h"

class MeshLoader {
public:
	bool load(std::string filename); // Load the file to model
	void pushTriangles(std::vector<glm::vec3> &triangles);
	
private:
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err, warn;
};

struct OctreeNodeDevice {
	glm::vec3 minCorner;
	glm::vec3 maxCorner;
	int children[8];
	int triangleStart;
	int triangleCount;
	int geomStart;
	int geomCount;
};

class OctreeNode {
public:
	glm::vec3 minCorner;
	glm::vec3 maxCorner;
	OctreeNode* children[8];
	int triangleStart;
	int triangleCount;
	int geomStart;
	int geomCount;


	~OctreeNode() {
		for (int i = 0; i < 8; i++) {
			delete children[i];
		}
	}
};

class Octree {
public:
	OctreeNode* root;

	void addTriangles(const std::vector<glm::vec3>& triangles);
	void addGeoms(const std::vector<Geom>& geoms);

private:

	void addHelper(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2);
	
	int maxLevel;
};