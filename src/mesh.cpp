// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION

// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

#include "mesh.h"

bool Mesh::load(std::string filename) {
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