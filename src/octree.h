#pragma once
#include "sceneStructs.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilities.h"
#include "macros.h"

class OctreeNode 
{
public:
	glm::vec3 boxCenter;
	float scale;
	int depth;
	int objNum;
	int primNum;
	int childCount;
	Geom octBlock;

	bool isDivided;
	bool hasChild[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	int nodeIndices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
	int* primitiveArray;
	int* meshTriangleArray;
	std::vector<Geom> storeCache;
	std::vector<MeshTri> storeTriCache;

	std::vector<int> primitiveIndices;
	std::vector<int> meshTriangleIndices;

	int primitiveCount;
	int meshTriCount;
	int primOffset;
	int meshTriOffset;


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
	int depth;
	int primitiveCount;
	int meshTriCount;
	std::vector<OctreeNode> nodeData;

public:
	Octree();

	void insertPrim(int geoIndex, Geom geometry);
	void insertMeshTri(MeshTri tri);
	void pointerize();

	~Octree();

private:
	int computeMinObj;
	bool boxBoxAABBInter(glm::vec3 pMin, glm::vec3 pMax, glm::vec3 bMin, glm::vec3 bMax);
	bool boxBoxAABBContain(glm::vec3 pMin, glm::vec3 pMax, glm::vec3 bMin, glm::vec3 bMax);
	bool newNodeTest(OctreeNode& octNode, int geoIndex, Geom geometry);
	bool newNodeTriTest(OctreeNode& octNode, MeshTri tri);
	bool insertPrimToNode(int nodeIndex, int geoIndex, Geom geometry);
	bool insertMeshTriToNode(int nodeIndex, MeshTri tri);
	bool triBoxContain(MeshTri tri, glm::vec3 bMin, glm::vec3 bMax);
};