#ifndef MESH_H
#define MESH_H
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
	std::vector<int> triangleIndices;
	std::vector<int> geomIndices;

	OctreeNode() {
		for (int i = 0; i < 8; i++) {
			children[i] = nullptr;
		}
	}
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

	Octree(int maxLevel, float sceneSize) :
		root(new OctreeNode()), maxLevel(maxLevel), sceneSize(sceneSize) {
		root->maxCorner = glm::vec3(sceneSize * 0.5f);
		root->minCorner = -root->maxCorner;
	}
	~Octree() { delete root; }

private:
	void addPrimitive(glm::vec3 minCorner, glm::vec3 maxCorner,
		bool isTriangle, int primitiveIndex);
	int childIndex(const glm::vec3& point, const glm::vec3& split);
	void childBoundingBox(const glm::vec3& parentMin, const glm::vec3& parentMax,
		int index, glm::vec3& childMin, glm::vec3& childMax);

	int maxLevel; // Max level of octree
	float sceneSize; // Determine the bounding box for the entire scene
};


#endif // !MESH_H