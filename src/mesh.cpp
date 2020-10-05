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
		
	}
}

void Octree::addGeoms(const std::vector<Geom>& geoms) {

}

void Octree::addHelper(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) {

}