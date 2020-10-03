#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

class Scene {
private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    int loadGeom(std::string objectid);
    int loadCamera();

    static std::vector<Geom> loadObj(std::istream&);

    static void aabbForGeom(const Geom &geom, glm::vec3 *min, glm::vec3 *max);
public:
    Scene(std::string filename);

    void buildTree();

    std::vector<AABBTreeNode> aabbTree;
    int aabbTreeRoot;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
