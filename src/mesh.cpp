// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

#include "mesh.h"

bool MeshLoader::load(std::string filename) {
	bool status = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
	if (!warn.empty()) {
		std::cout << "Warn: " << warn << std::endl;
	}
	if (!err.empty()) {
		std::cerr << "Error: " << err << std::endl;
	}
	if (!status) {
		std::cout << "Failed to parse glTF: " << filename << std::endl;
	}
	return status;
}

void MeshLoader::pushTriangles(std::vector<glm::vec3>& triangles) {
	for (auto &primitive : model.meshes[0].primitives) {
		// Get vertex positions
		std::vector<glm::vec3> positions;
		const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes["POSITION"]];
		const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
		const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

		// bufferView.byteOffset + accessor.byteOffset tells you where the
		// actual position/normal/uv data is within the buffer.
		// when multiple accessors refer to the same bufferView, then the
		// byteOffset describes where the data of the accessor starts,
		// relative to the bufferView that it refers to
		const float* posData = reinterpret_cast<const float*>(
			&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);
		int posStride = posBufferView.byteStride == 0 ? 3 : (posBufferView.byteStride / 4);
		for (int i = 0, ct = 0; ct < posAccessor.count; i += posStride, ct++) {
			positions.push_back(glm::vec3(posData[i], posData[i + 1], posData[i + 2]));
		}

		// Get vertex indices
		const tinygltf::Accessor& idxAccessor = model.accessors[primitive.indices];
		const tinygltf::BufferView& idxBufferView = model.bufferViews[idxAccessor.bufferView];
		const tinygltf::Buffer& idxBuffer = model.buffers[idxBufferView.buffer];
		assert(idxBufferView.byteStride == 0);
		
		const unsigned short* idxData = reinterpret_cast<const unsigned short*>(
			&idxBuffer.data[idxBufferView.byteOffset + idxAccessor.byteOffset]);
		for (int i = 0; i < idxAccessor.count; i++) {
			triangles.push_back(positions[idxData[i]]);
		}
	}
}

void Octree::addTriangles(const std::vector<glm::vec3>& triangles) {
	for (int i = 0; i < triangles.size(); i += 3) {
		glm::vec3 minCorner(FLT_MAX);
		glm::vec3 maxCorner(-FLT_MAX);
		for (int j = i; j < i + 3; j++) {
			const glm::vec3& t = triangles[j];
			minCorner.x = std::min(minCorner.x, t.x);
			minCorner.y = std::min(minCorner.y, t.y);
			minCorner.z = std::min(minCorner.z, t.z);
			maxCorner.x = std::max(maxCorner.x, t.x);
			maxCorner.y = std::max(maxCorner.y, t.y);
			maxCorner.z = std::max(maxCorner.z, t.z);
		}
		// Offset the corner a little bit to avoid error from axis-aligned triangles
		maxCorner += glm::vec3(0.01f);
		minCorner -= glm::vec3(0.01f);
		addPrimitive(minCorner, maxCorner, true, i);
	}
}

void Octree::addGeoms(const std::vector<Geom>& geoms) {
	for (int i = 0; i < geoms.size(); i++) {
		auto& geom = geoms[i];
		if (geom.type != MESH) {
			glm::vec3 minCorner, maxCorner;
			geom.getBoundingBox(minCorner, maxCorner);
			addPrimitive(minCorner, maxCorner, false, i);
		}
	}
}

/// <summary>
/// A helper function for adding a cube/sphere/triangle to the octree.
/// </summary>
/// <param name="minCorner"> Bounding box lower corner. </param>
/// <param name="maxCorner"> Bounding box upper corner. </param>
void Octree::addPrimitive(glm::vec3 minCorner, glm::vec3 maxCorner,
	bool isTriangle, int primitiveIndex) {
	int level = 1;
	for (OctreeNode* node = root; ; level++) {
		glm::vec3 split = (node->minCorner + node->maxCorner) * 0.5f;
		int minChild = childIndex(minCorner, split);
		int maxChild = childIndex(maxCorner, split);
		if (level >= maxLevel || minChild != maxChild) {
			if (isTriangle) {
				node->triangleIndices.push_back(primitiveIndex);
			}
			else {
				node->geomIndices.push_back(primitiveIndex);
			}
			return;
		}
		else { // Fit in a lower level, i.e., a child bounding box
			if (node->children[minChild] == nullptr) {
				node->children[minChild] = new OctreeNode();
				childBoundingBox(node->minCorner, node->maxCorner, minChild,
					node->children[minChild]->minCorner, node->children[minChild]->maxCorner);
			}
			node = node->children[minChild];
		}
	}
}

/// <summary>
/// Determine the index of child this point is located in.
/// </summary>
/// <returns> An index from 0 to 7. </returns>
int Octree::childIndex(const glm::vec3& point, const glm::vec3& split) {
	int x = point.x <= split.x ? 0 : 1;
	int y = point.y <= split.y ? 0 : 1;
	int z = point.z <= split.z ? 0 : 1;
	return x * 4 + y * 2 + z;
}

/// <summary>
/// Get the bounding box of the child octree by index.
/// The bounding box is represented by two diagonal corner points.
/// </summary>
void Octree::childBoundingBox(const glm::vec3& parentMin, const glm::vec3& parentMax,
	int index, glm::vec3& childMin, glm::vec3& childMax) {
	assert(0 <= index && index < 8);
	glm::vec3 split = (parentMin + parentMax) * 0.5f;
	int x = index & 4;
	int y = index & 2;
	int z = index & 1;
	childMin.x = x == 0 ? parentMin.x : split.x;
	childMin.y = y == 0 ? parentMin.y : split.y;
	childMin.z = z == 0 ? parentMin.z : split.z;
	childMax.x = x == 0 ? split.x : parentMax.x;
	childMax.y = y == 0 ? split.y : parentMax.y;
	childMax.z = z == 0 ? split.z : parentMax.z;
}