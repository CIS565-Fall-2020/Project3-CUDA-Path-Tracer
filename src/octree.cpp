#include "octree.h"

OctreeNode::OctreeNode() :boxCenter(glm::vec3(0.0f, 0.0f, 0.0f)), scale(50.0f), depth(0), objNum(0), childCount(0), isDivided(false) 
{
	glm::vec3 center = boxCenter;

	octBlock.type = CUBE;
	octBlock.translation = center;
	octBlock.scale = glm::vec3(scale, scale, scale);
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
	boxCenter(boxCenter), scale(scale), depth(depth), objNum(0), childCount(0), isDivided(false) 
{
	glm::vec3 center = boxCenter;

	octBlock.type = CUBE;
	octBlock.translation = center;
	octBlock.scale = glm::vec3(scale, scale, scale);
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

Octree::Octree() :maxDepth(8), minObj(2), computeMinObj(minObj - 1)
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
	if (geometry.type == CUBE)
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

		if (boxBoxAABBInter(pMin, pMax, bMin, bMax))
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
	if (geometry.type == CUBE) 
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

		if (boxBoxAABBInter(pMin, pMax, bMin, bMax)) 
		{
			nodeData.at(nodeIndex).primitiveIndices.push_back(geoIndex);
			nodeData.at(nodeIndex).objNum++;
			
			if (nodeData.at(nodeIndex).objNum <= computeMinObj)
			{
				nodeData.at(nodeIndex).storeCache.push_back(geometry);
			}

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
									nodeData.at(nodeIndex).isDivided = false;
									for (int w = 0; w < nodeData.at(nodeIndex).storeCache.size(); w++)
									{
										bool isInter = newNodeTest(newNode, nodeData.at(nodeIndex).primitiveIndices[w], nodeData.at(nodeIndex).storeCache[w]);
										if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter == true)
										{
											nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
										}
									}

									for (int w = 0; w < nodeData.at(nodeIndex).storeTriCache.size(); w++)
									{
										bool isInter = insertMeshTriToNode(newNode,
											nodeData.at(nodeIndex).meshTriangleIndices[w],
											nodeData.at(nodeIndex).storeTriCache[3 * w],
											nodeData.at(nodeIndex).storeTriCache[3 * w + 1],
											nodeData.at(nodeIndex).storeTriCache[3 * w + 2]);
										if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter == true)
										{
											nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
										}
									}
								}

								bool isInter = newNodeTest(newNode, geoIndex, geometry);

								if (!(nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k]) && isInter)
								{
									nodeData.at(nodeIndex).hasChild[i * 2 * 2 + j * 2 + k] = isInter;
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

bool Octree::insertMeshTriToNode(OctreeNode& octNode, int faceIndex, glm::vec3 meshPosX, glm::vec3 meshPosY, glm::vec3 meshPosZ) 
{
	return false;
}

void Octree::pointerize() 
{
	for (int i = 0; i < nodeData.size(); i++) 
	{
		nodeData.at(i).meshTriangleArray = nodeData.at(i).meshTriangleIndices.data();
		nodeData.at(i).primitiveArray = nodeData.at(i).primitiveIndices.data();
	}
}
