#include "octree.h"

OctreeNode::OctreeNode() :boxCenter(glm::vec3(0.0f, 0.0f, 0.0f)), scale(50.0f), depth(0), objNum(0), childCount(0), 
                          isDivided(false), primOffset(0), meshTriOffset(0), primNum(0)
{
	glm::vec3 center = boxCenter;

	octBlock.type = CUBE;
	octBlock.translation = center;
	octBlock.scale = glm::vec3(scale);
	octBlock.rotation = glm::vec3(0.0f);

	glm::mat4 translationMat = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 0.0f));
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), octBlock.rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), octBlock.rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), octBlock.rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), octBlock.scale);
	octBlock.transform = translationMat * rotationMat * scaleMat;
	octBlock.inverseTransform = glm::inverse(octBlock.transform);
	octBlock.invTranspose = glm::inverseTranspose(octBlock.transform);
}

OctreeNode::OctreeNode(glm::vec3 boxCenter, float scale, int depth):
	boxCenter(boxCenter), scale(scale), depth(depth), objNum(0), childCount(0), 
	isDivided(false), primitiveCount(0), meshTriCount(0), primOffset(0), meshTriOffset(0), primNum(0)
{
	glm::vec3 center = boxCenter;

	octBlock.type = CUBE;
	octBlock.translation = center;
	octBlock.scale = glm::vec3(scale);
	octBlock.rotation = glm::vec3(0.0f);

	glm::mat4 translationMat = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 0.0f));
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), octBlock.rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), octBlock.rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), octBlock.rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), octBlock.scale);
	octBlock.transform = translationMat * rotationMat * scaleMat;
	octBlock.inverseTransform = glm::inverse(octBlock.transform);
	octBlock.invTranspose = glm::inverseTranspose(octBlock.transform);
}

OctreeNode::~OctreeNode() {}

Octree::Octree() :maxDepth(OCT_MAX_DEPTH), minObj(2), computeMinObj(minObj - 1), primitiveCount(0), meshTriCount(0)
{
	OctreeNode rootNode = OctreeNode();
	nodeData.push_back(rootNode);
}

Octree::~Octree() {}

void Octree::insertPrim(int geoIndex, Geom geometry) 
{
	insertPrimToNode(0, geoIndex, geometry);
}

bool Octree::newNodeTest(OctreeNode& octNode, int geoIndex, Geom geometry) 
{
	if (geometry.type == CUBE || geometry.type == SPHERE)
	{
		glm::vec3 pos0 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, -0.5f, -0.5f), 1.0f));
		glm::vec3 pos1 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, -0.5f, -0.5f), 1.0f));
		glm::vec3 pos2 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, -0.5f, 0.5f), 1.0f));
		glm::vec3 pos3 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, -0.5f, 0.5f), 1.0f));
		glm::vec3 pos4 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, 0.5f, 0.5f), 1.0f));
		glm::vec3 pos5 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, 0.5f, 0.5f), 1.0f));
		glm::vec3 pos6 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, 0.5f, -0.5f), 1.0f));
		glm::vec3 pos7 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, 0.5f, -0.5f), 1.0f));

		std::vector<glm::vec3> cubePos = { pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7 };

		glm::vec3 pMin = pos0;
		glm::vec3 pMax = pos0;
		for (glm::vec3 pos : cubePos)
		{
			if (pMin.x > pos.x)
				pMin.x = pos.x;

			if (pMax.x < pos.x)
				pMax.x = pos.x;

			if (pMin.y > pos.y)
				pMin.y = pos.y;

			if (pMax.y < pos.y)
				pMax.y = pos.y;

			if (pMin.z > pos.z)
				pMin.z = pos.z;

			if (pMax.z < pos.z)
				pMax.z = pos.z;
		}

		glm::vec3 bMin = glm::vec3(octNode.boxCenter - octNode.scale / 2.0f);
		glm::vec3 bMax = glm::vec3(octNode.boxCenter + octNode.scale / 2.0f);

		if (boxBoxAABBContain(pMin, pMax, bMin, bMax))
		{
			octNode.primitiveIndices.push_back(geoIndex);
			octNode.objNum++;

			if (octNode.objNum <= computeMinObj)
			{
				octNode.storeCache.push_back(geometry);
			}

			return true;
		}
		else 
		{
			return false;
		}
	}
}

bool Octree::insertPrimToNode(int nodeIndex, int geoIndex, Geom geometry) 
{
	if (geometry.type == CUBE || geometry.type == SPHERE) 
	{
		glm::vec3 pos0 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, -0.5f, -0.5f), 1.0f));
		glm::vec3 pos1 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, -0.5f, -0.5f), 1.0f));
		glm::vec3 pos2 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, -0.5f, 0.5f), 1.0f));
		glm::vec3 pos3 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, -0.5f, 0.5f), 1.0f));
		glm::vec3 pos4 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, 0.5f, 0.5f), 1.0f));
		glm::vec3 pos5 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, 0.5f, 0.5f), 1.0f));
		glm::vec3 pos6 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(0.5f, 0.5f, -0.5f), 1.0f));
		glm::vec3 pos7 = glm::vec3(geometry.transform * glm::vec4(glm::vec3(-0.5f, 0.5f, -0.5f), 1.0f));

		std::vector<glm::vec3> cubePos = { pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7 };

		glm::vec3 pMin = pos0;
		glm::vec3 pMax = pos0;
		for (glm::vec3 pos : cubePos) 
		{
			if (pMin.x > pos.x)
				pMin.x = pos.x;

			if (pMax.x < pos.x)
				pMax.x = pos.x;

			if (pMin.y > pos.y)
				pMin.y = pos.y;

			if (pMax.y < pos.y)
				pMax.y = pos.y;

			if (pMin.z > pos.z)
				pMin.z = pos.z;

			if (pMax.z < pos.z)
				pMax.z = pos.z;
		}

		glm::vec3 bMin = glm::vec3(nodeData.at(nodeIndex).boxCenter - nodeData.at(nodeIndex).scale / 2.0f);
		glm::vec3 bMax = glm::vec3(nodeData.at(nodeIndex).boxCenter + nodeData.at(nodeIndex).scale / 2.0f);

		if (boxBoxAABBContain(pMin, pMax, bMin, bMax)) 
		{
			nodeData.at(nodeIndex).primitiveIndices.push_back(geoIndex);
			nodeData.at(nodeIndex).objNum++;
			
			if (nodeData.at(nodeIndex).objNum <= computeMinObj)
			{
				nodeData.at(nodeIndex).storeCache.push_back(geometry);
			}
			bool curDiv = false;
			// Divide the cube
			if (nodeData.at(nodeIndex).objNum > computeMinObj && nodeData.at(nodeIndex).depth != this->maxDepth) 
			{
				// Split the blocks
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 2; k++)
						{
							glm::vec3 curNode = nodeData.at(nodeIndex).boxCenter - nodeData.at(nodeIndex).scale / 4.0f +
								(float)i * glm::vec3(nodeData.at(nodeIndex).scale / 2.0f, 0.0f, 0.0f) +
								(float)j * glm::vec3(0.0f, nodeData.at(nodeIndex).scale / 2.0f, 0.0f) +
								(float)k * glm::vec3(0.0f, 0.0f, nodeData.at(nodeIndex).scale / 2.0f);

							float curScale = nodeData.at(nodeIndex).scale / 2.0f;

							if (!nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) 
							{
								OctreeNode newNode = OctreeNode(curNode, curScale, nodeData.at(nodeIndex).depth + 1);

								// Send detected geos
								if (nodeData.at(nodeIndex).objNum > computeMinObj && !nodeData.at(nodeIndex).isDivided)
								{
									
									for (int w = 0; w < nodeData.at(nodeIndex).storeCache.size(); w++)
									{
										bool isInter = newNodeTest(newNode, nodeData.at(nodeIndex).primitiveIndices[w], nodeData.at(nodeIndex).storeCache[w]);
										if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter == true)
										{
											curDiv = true;
											nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
											nodeData.at(nodeIndex).primitiveIndices.pop_back();
										}
										
									}

									for (int w = 0; w < nodeData.at(nodeIndex).storeTriCache.size(); w++)
									{
										bool isInter = newNodeTriTest(newNode, nodeData.at(nodeIndex).storeTriCache[w]);
										if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter == true)
										{
											curDiv = true;
											nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
											nodeData.at(nodeIndex).meshTriangleIndices.pop_back();
										}
										
									}
								}

								bool isInter = newNodeTest(newNode, geoIndex, geometry);

								if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter)
								{
									nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
									nodeData.at(nodeIndex).primitiveIndices.pop_back();
								}

								if (nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) 
								{
									nodeData.at(nodeIndex).nodeIndices[i * 2 * 2 + j * 2 + k] = nodeData.size();
									nodeData.at(nodeIndex).childCount++;
									this->nodeData.push_back(newNode);
								}
							}
							else 
							{
								int childIndex = nodeData.at(nodeIndex).nodeIndices[i * 2 * 2 + j * 2 + k];
								bool isInter = insertPrimToNode(childIndex, geoIndex, geometry);
							}
						}
					}
				}

				if (curDiv) 
				{
					nodeData.at(nodeIndex).isDivided = true;
				}
			}
			return true;
		}
	}
	return false;	
}

bool Octree::boxBoxAABBInter(glm::vec3 pMin, glm::vec3 pMax, glm::vec3 bMin, glm::vec3 bMax) 
{
	return (pMin.x <= bMax.x && pMax.x >= bMin.x) &&
		   (pMin.y <= bMax.y && pMax.y >= bMin.y) &&
		   (pMin.z <= bMax.z && pMax.z >= bMin.z);
}

bool Octree::boxBoxAABBContain(glm::vec3 pMin, glm::vec3 pMax, glm::vec3 bMin, glm::vec3 bMax)
{
	return (pMax.x <= bMax.x && pMin.x >= bMin.x) &&
		(pMax.y <= bMax.y && pMin.y >= bMin.y) &&
		(pMax.z <= bMax.z && pMin.z >= bMin.z);
}

bool Octree::insertMeshTriToNode(int nodeIndex, MeshTri tri) 
{
	glm::vec3 bMin = glm::vec3(nodeData.at(nodeIndex).boxCenter - nodeData.at(nodeIndex).scale / 2.0f);
	glm::vec3 bMax = glm::vec3(nodeData.at(nodeIndex).boxCenter + nodeData.at(nodeIndex).scale / 2.0f);

	if (triBoxContain(tri, bMin, bMax))
	{
		nodeData.at(nodeIndex).meshTriangleIndices.push_back(tri.faceIndex);
		nodeData.at(nodeIndex).objNum++;

		if (nodeData.at(nodeIndex).objNum <= computeMinObj)
		{
			nodeData.at(nodeIndex).storeTriCache.push_back(tri);
		}

		// Divide the cube
		if (nodeData.at(nodeIndex).objNum > computeMinObj&& nodeData.at(nodeIndex).depth != this->maxDepth)
		{
			bool curDiv = false;
			// Split the blocks
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					for (int k = 0; k < 2; k++)
					{
						glm::vec3 curNode = nodeData.at(nodeIndex).boxCenter - nodeData.at(nodeIndex).scale / 4.0f +
							(float)i * glm::vec3(nodeData.at(nodeIndex).scale / 2.0f, 0.0f, 0.0f) +
							(float)j * glm::vec3(0.0f, nodeData.at(nodeIndex).scale / 2.0f, 0.0f) +
							(float)k * glm::vec3(0.0f, 0.0f, nodeData.at(nodeIndex).scale / 2.0f);

						float curScale = nodeData.at(nodeIndex).scale / 2.0f;

						if (!nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k])
						{
							OctreeNode newNode = OctreeNode(curNode, curScale, nodeData.at(nodeIndex).depth + 1);

							// Send detected geos
							if (nodeData.at(nodeIndex).objNum > computeMinObj && !nodeData.at(nodeIndex).isDivided)
							{
								for (int w = 0; w < nodeData.at(nodeIndex).storeCache.size(); w++)
								{
									bool isInter = newNodeTest(newNode, nodeData.at(nodeIndex).primitiveIndices[w], nodeData.at(nodeIndex).storeCache[w]);
									if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter == true)
									{
										curDiv = true;
										nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
										nodeData.at(nodeIndex).primitiveIndices.pop_back();
									}

								}

								for (int w = 0; w < nodeData.at(nodeIndex).storeTriCache.size(); w++)
								{
									bool isInter = newNodeTriTest(newNode, nodeData.at(nodeIndex).storeTriCache[w]);
									if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter == true)
									{
										curDiv = true;
										nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
										nodeData.at(nodeIndex).meshTriangleIndices.pop_back();
									}
								}
							}

							bool isInter = newNodeTriTest(newNode, tri);

							if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter)
							{
								nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
								nodeData.at(nodeIndex).meshTriangleIndices.pop_back();
							}

							if (nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k])
							{
								nodeData.at(nodeIndex).nodeIndices[i * 2 * 2 + j * 2 + k] = nodeData.size();
								nodeData.at(nodeIndex).childCount++;
								this->nodeData.push_back(newNode);
							}
						}
						else
						{
							int childIndex = nodeData.at(nodeIndex).nodeIndices[i * 2 * 2 + j * 2 + k];
							bool isInter = insertMeshTriToNode(childIndex, tri);

							if (isInter) 
							{
								nodeData.at(nodeIndex).meshTriangleIndices.pop_back();
							}
						}
					}
				}
			}
			if (curDiv) 
			{
				nodeData.at(nodeIndex).isDivided = true;
			}
		}
		return true;
	}
	
	return false;
}

bool Octree::newNodeTriTest(OctreeNode& octNode, MeshTri tri) 
{
		glm::vec3 bMin = glm::vec3(octNode.boxCenter - octNode.scale / 2.0f);
		glm::vec3 bMax = glm::vec3(octNode.boxCenter + octNode.scale / 2.0f);


		if (triBoxContain(tri, bMin, bMax))
		{
			octNode.meshTriangleIndices.push_back(tri.faceIndex);
			octNode.objNum++;

			if (octNode.objNum <= computeMinObj)
			{
				octNode.storeTriCache.push_back(tri);
			}

			return true;
		}
		else
		{
			return false;
		}
}

void Octree::pointerize() 
{
	int maxDepth = INT_MIN;

	for (int i = 0; i < nodeData.size(); i++) 
	{
		nodeData.at(i).meshTriangleArray = nodeData.at(i).meshTriangleIndices.data();
		nodeData.at(i).primitiveArray = nodeData.at(i).primitiveIndices.data();
		nodeData.at(i).primitiveCount = nodeData.at(i).primitiveIndices.size();
		nodeData.at(i).meshTriCount = nodeData.at(i).meshTriangleIndices.size();

		nodeData.at(i).primOffset = primitiveCount;
		nodeData.at(i).meshTriOffset = meshTriCount;

		primitiveCount += nodeData.at(i).primitiveCount;
		meshTriCount += nodeData.at(i).meshTriCount;



		if (nodeData.at(i).depth > maxDepth) 
		{
			maxDepth = nodeData.at(i).depth;
		}
	}

	depth = maxDepth;
}

void Octree::insertMeshTri(MeshTri tri) 
{
	insertMeshTriToNode(0, tri);
}

bool Octree::triBoxContain(MeshTri tri, glm::vec3 bMin, glm::vec3 bMax)
{
	if (bMin.x <= tri.x.x && bMax.x >= tri.x.x
		&& bMin.y <= tri.x.y && bMax.y >= tri.x.y
		&& bMin.z <= tri.x.z && bMax.z >= tri.x.z
		&& bMin.x <= tri.y.x && bMax.x >= tri.y.x
		&& bMin.y <= tri.y.y && bMax.y >= tri.y.y
		&& bMin.z <= tri.y.z && bMax.z >= tri.y.z
		&& bMin.x <= tri.z.x && bMax.x >= tri.z.x
		&& bMin.y <= tri.z.y && bMax.y >= tri.z.y
		&& bMin.z <= tri.z.z && bMax.z >= tri.z.z) 
	{
		return true;
	}
	else 
	{
		return false;
	}
}