#include "tiny_gltf.h"

#include <string>
#include <iostream>


class Mesh {
public:
	tinygltf::Model model;
	bool load(std::string filename);
private:
	tinygltf::TinyGLTF loader;
	std::string err, warn;
};