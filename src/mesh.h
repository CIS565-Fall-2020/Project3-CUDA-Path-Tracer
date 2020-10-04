#include "tiny_gltf.h"

#include <string>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <vector>

class MeshLoader {
public:
	bool load(std::string filename); // Load the file to model
	void pushTriangles(std::vector<glm::vec3> &triangles);

private:
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err, warn;
};