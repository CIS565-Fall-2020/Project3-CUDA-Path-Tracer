#pragma once
#include "sceneStructs.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilities.h"

class OctreeNode 
{
public:
	glm::vec3 boxCenter;
	float scale;
	int depth;
	int objNum;
	int childCount;
	Geom octBlock;

	bool isDivided;
	bool hasChild[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	int nodeIndices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
	int* primitiveArray;
	int* meshTriangleArray;
	std::vector<Geom> storeCache;
	std::vector<glm::vec3> storeTriCache;

	std::vector<int> primitiveIndices;
	std::vector<int> meshTriangleIndices;

	int primitiveCount;
	int meshTriCount;


public:
	OctreeNode();
	OctreeNode(glm::vec3 boxCenter, float scale, int depth);

	~OctreeNode();
};

class Octree 
{
public:
	int maxDepth;
	int minObj;
	std::vector<OctreeNode> nodeData;

public:
	Octree();

	void insertPrim(int geoIndex, Geom geometry);
	void insertMeshTri(int faceIndex, glm::vec3 meshPosX, glm::vec3 meshPosY, glm::vec3 meshPosZ);
	void pointerize();

	~Octree();

private:
	int computeMinObj;
	bool boxBoxAABBInter(glm::vec3 pMin, glm::vec3 pMax, glm::vec3 bMin, glm::vec3 bMax);
	bool newNodeTest(OctreeNode& octNode, int geoIndex, Geom geometry);
	bool insertPrimToNode(int nodeIndex, int geoIndex, Geom geometry);
	bool insertMeshTriToNode(OctreeNode& octNode, int faceIndex, glm::vec3 meshPosX, glm::vec3 meshPosY, glm::vec3 meshPosZ);

};