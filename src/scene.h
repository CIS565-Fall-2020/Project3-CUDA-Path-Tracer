#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "bvh.h"
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

    int maxPrimsInNode = 64;

    void buildAccelerationStructure();
    BVHBuildNode* recurBVHbuild(int start, int end, int& totalNodes, std::vector<Primitive>& orderedPrims);
    int flattenBVHTree(BVHBuildNode* node, int& offset);

    std::vector<Geom> geoms;
    std::vector<int> lightIDs; // each int represent a geom id that has emissive material
    int environmentLightID_idx = NULL_PRIMITIVE;

    void setDeviceEnvMap();

    std::vector<Triangle> triangles;
    std::vector<GLTF_Model> gltf_models;

    std::vector<Primitive> primitives;
    std::vector<BVHprimitiveInfo> primitivesInfo;
    std::vector<LinearBVHNode> LBVHnodes;

    std::vector<Material> materials;

    std::vector<Texture*> textures;
    std::unordered_map<string, Texture*> textureMap; // avoid repeated texture

    std::vector<Float> sampler_pdf;
    std::vector<Float> sampler_cdf;

    RenderState state;
};
