#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>

#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

struct ObjFile {
    std::vector<GeomTriangle> triangles;
    std::map<std::size_t, std::string> materials;
};

class Scene {
private:
    std::ifstream fp_in;
    void loadMaterial(std::string materialid);
    void loadGeom();
    int loadCamera();

    static ObjFile loadObj(std::istream&, glm::mat4);

    static void aabbForGeom(const Geom &geom, glm::vec3 *min, glm::vec3 *max);
public:
    Scene(std::string filename);

    void buildTree();

    static void computeCameraParameters(Camera&);

    std::vector<AABBTreeNode> aabbTree;
    int aabbTreeRoot;

    std::vector<Geom> geoms;
    std::map<std::string, int> materialIdMapping;
    std::vector<Material> materials;
    RenderState state;
};
