#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <glm/gtx/normal.hpp>

#include "tiny_gltf.h"
#include "gltf-loader.h"

#include "image.h"
#include <unordered_map>

// Jack12 add gltf support 


using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    // jack12
    bool myGLTFloader(
        const std::string& file_path,
        float scale,
        std::vector<example::Mesh<float>>& out_mesh,
        std::vector<example::Material>& materials,
        std::vector<example::Texture>& textures);
    int loadGLTFMesh(
        const std::string& file_path,
        const Geom& parent);

    TextureDescriptor loadTexture(const string& path, bool normalize);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<GLTF_Model> gltf_models;

    std::vector<Material> materials;

    std::vector<Texture*> textures;
    std::unordered_map<string, Texture*> textureMap; // avoid repeated texture

    RenderState state;
};
