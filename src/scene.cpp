#include <iostream>
#include <cstring>
#include <sstream>
#include <deque>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include "scene.h"

Scene::Scene(std::string filename) {
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }
    while (fp_in.good()) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                std::cout << " " << std::endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom();
                std::cout << " " << std::endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                std::cout << " " << std::endl;
            }
        }
    }
}

void Scene::loadGeom() {
    std::cout << "Loading Geom..." << std::endl;
    glm::vec3 translation, rotation, scale;
    std::string line;
    std::string meshPath;
    int materialId = -1;
    GeomType type = GeomType::INVALID;

    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (line == "sphere") {
            std::cout << "Creating new sphere..." << std::endl;
            type = GeomType::SPHERE;
        } else if (line == "cube") {
            std::cout << "Creating new cube..." << std::endl;
            type = GeomType::CUBE;
        } else if (line == "mesh") {
            type = GeomType::TRIANGLE;
            utilityCore::safeGetline(fp_in, meshPath);
        }
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens.size() >= 2) {
            if (tokens.size() != 2 || tokens[0] != "MATERIAL") {
                std::cout << "  Invalid material specification\n";
            } else {
                auto iter = materialIdMapping.find(tokens[1]);
                if (iter != materialIdMapping.end()) {
                    materialId = iter->second;
                    std::cout << "Connecting Geom to Material " << tokens[1] << " (" << materialId << ")...\n";
                } else {
                    std::cout << "Material not found: " << tokens[1] << "\n";
                }
            }
        }
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (tokens[0] == "TRANS") {
            translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (tokens[0] == "ROTAT") {
            rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (tokens[0] == "SCALE") {
            scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    glm::mat4 trans = utilityCore::buildTransformationMatrix(translation, rotation, scale);

    if (type == GeomType::TRIANGLE) { // load mesh
        std::ifstream fin(meshPath);
        ObjFile tris = loadObj(fin, trans);
        std::cout << "  Num triangles: " << tris.triangles.size() << "\n";
        int curMaterialId = materialId;
        auto matIter = tris.materials.begin();
        for (std::size_t i = 0; i < tris.triangles.size(); ++i) {
            if (matIter != tris.materials.end() && matIter->first == i) {
                auto it = materialIdMapping.find(matIter->second);
                if (it != materialIdMapping.end()) {
                    curMaterialId = it->second;
                } else {
                    std::cout << "  Material not found: " << matIter->second << ", using default material\n";
                    curMaterialId = materialId;
                }
                ++matIter;
            }

            Geom geom;
            geom.type = GeomType::TRIANGLE;
            geom.materialid = curMaterialId;
            geom.triangle = tris.triangles[i];
            geoms.push_back(geom);
        }
    } else {
        Geom newGeom;
        
        newGeom.type = type;
        newGeom.materialid = materialId;

        glm::mat4 invTrans = glm::inverse(trans);
        newGeom.implicit.transform = glm::mat4x3(trans);
        newGeom.implicit.inverseTransform = glm::mat4x3(invTrans);

        geoms.push_back(newGeom);
    }
}

void Scene::computeCameraParameters(Camera &cam) {
    float yscaled = tan(cam.fovy * (PI / 180)); // should divide by 360 here, but I'm leaving it this way for consistency
    float xscaled = (yscaled * cam.resolution.x) / cam.resolution.y;

    cam.right = glm::normalize(glm::cross(cam.view, cam.up));
    cam.pixelLength = 2.0f * glm::vec2(xscaled, yscaled);

    cam.view = glm::normalize(cam.lookAt - cam.position);
}

int Scene::loadCamera() {
    std::cout << "Loading Camera ..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;

    //load static properties
    for (int i = 0; i < 5; i++) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            camera.fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    std::string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    computeCameraParameters(camera);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    std::cout << "Loaded camera!" << std::endl;
    return 1;
}

void Scene::loadMaterial(std::string materialid) {
    std::cout << "Loading Material " << materialid << "..." << std::endl;
    Material newMaterial;

    //load static properties
    for (int i = 0; i < 7; i++) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RGB") == 0) {
            glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
            newMaterial.color = color;
        } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
            newMaterial.specular.exponent = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
            glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            newMaterial.specular.color = specColor;
        } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
            newMaterial.hasReflective = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
            newMaterial.hasRefractive = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
            newMaterial.indexOfRefraction = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
            newMaterial.emittance = atof(tokens[1].c_str());
        }
    }
    materialIdMapping[materialid] = materials.size();
    materials.push_back(newMaterial);
}

bool skipSeparator(std::istream &in) {
    int c;
    while ((c = in.get()) != std::char_traits<char>::eof()) {
        if (c == '/') {
            return true;
        }
        if (!std::isspace(c)) {
            in.unget();
            break;
        }
    }
    return false;
}
glm::ivec3 readVertex(std::istream &in) {
    glm::ivec3 res(0);
    in >> res.x;
    if (skipSeparator(in)) {
        in >> res.y;
        if (skipSeparator(in)) {
            in >> res.z;
        }
    }
    return res - 1;
}
ObjFile Scene::loadObj(std::istream &in, glm::mat4 trans) {
    struct _face {
        glm::ivec3 id[3];
    };

    ObjFile result;

    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<_face> faces;
    for (std::string line; std::getline(in, line); ) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "v") {
            glm::vec3 vert;
            ss >> vert.x >> vert.y >> vert.z;
            verts.emplace_back(vert);
        } else if (cmd == "vn") {
            glm::vec3 norm;
            ss >> norm.x >> norm.y >> norm.z;
            normals.emplace_back(norm);
        } else if (cmd == "vt") {
            glm::vec2 uv;
            ss >> uv.x >> uv.y;
            uvs.emplace_back(uv);
        } else if (cmd == "f") {
            _face face;
            for (std::size_t i = 0; i < 3; ++i) {
                face.id[i] = readVertex(ss);
            }
            faces.emplace_back(face);
        } else if (cmd == "usemtl") {
            std::string mtlName;
            ss >> mtlName;
            result.materials.try_emplace(faces.size(), mtlName);
        }
    }

    // apply transform
    for (glm::vec3 &pos : verts) {
        pos = glm::vec3(trans * glm::vec4(pos, 1.0f));
    }
    glm::mat4 invTrans = glm::inverseTranspose(trans);
    for (glm::vec3 &norm : normals) {
        norm = glm::vec3(invTrans * glm::vec4(norm, 0.0f));
    }

    for (const auto &face : faces) {
        result.triangles.emplace_back();
        GeomTriangle &tri = result.triangles.back();
        for (std::size_t i = 0; i < 3; ++i) {
            tri.vertices[i] = verts[face.id[i].x];
        }
        glm::vec3 flatNormal = glm::normalize(glm::cross(
            tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0]
        ));
        for (std::size_t i = 0; i < 3; ++i) {
            tri.uvs[i] = face.id[i].y < 0 ? glm::vec2(0.0f) : uvs[face.id[i].y];
            tri.normals[i] = face.id[i].z < 0 ? flatNormal : normals[face.id[i].z];
        }
    }
    return result;
}

struct _buildStep {
    _buildStep() = default;
    _buildStep(int *parent, std::size_t beg, std::size_t end) : parentPtr(parent), rangeBeg(beg), rangeEnd(end) {
    }

    int *parentPtr;
    std::size_t rangeBeg, rangeEnd;
};
struct _leaf {
    glm::vec3 centroid, aabbMin, aabbMax;
    int geomIndex;
    std::size_t bucket;
};
float surfaceAreaHeuristic(glm::vec3 min, glm::vec3 max) {
    glm::vec3 size = max - min;
    return size.x * size.y + size.x * size.z + size.y * size.z;
}
struct _bucket {
    glm::vec3
        aabbMin{ FLT_MAX },
        aabbMax{ -FLT_MAX };
    std::size_t count = 0;

    float heuristic() const {
        return count * surfaceAreaHeuristic(aabbMin, aabbMax);
    }

    inline static _bucket merge(_bucket lhs, _bucket rhs) {
        lhs.count += rhs.count;
        lhs.aabbMin = glm::min(lhs.aabbMin, rhs.aabbMin);
        lhs.aabbMax = glm::max(lhs.aabbMax, rhs.aabbMax);
        return lhs;
    }
};
void Scene::buildTree() {
    constexpr std::size_t numBuckets = 12;

    if (geoms.size() == 0) {
        return;
    }
    std::vector<_leaf> leaves(geoms.size());
    for (std::size_t i = 0; i < geoms.size(); ++i) {
        _leaf &cur = leaves[i];
        aabbForGeom(geoms[i], &cur.aabbMin, &cur.aabbMax);
        cur.geomIndex = i;
        cur.centroid = 0.5f * (cur.aabbMin + cur.aabbMax);
    }

    aabbTree.resize(geoms.size() - 1);
    int alloc = 0;
    std::deque<_buildStep> q;
    q.emplace_back(&aabbTreeRoot, 0, geoms.size());
    while (!q.empty()) {
        _buildStep step = q.front();
        q.pop_front();

        switch (step.rangeEnd - step.rangeBeg) {
        case 1:
            *step.parentPtr = ~leaves[step.rangeBeg].geomIndex;
            break;
        case 2:
            {
                _leaf &left = leaves[step.rangeBeg], &right = leaves[step.rangeBeg + 1];
                int nodeIndex = alloc++;
                *step.parentPtr = nodeIndex;
                AABBTreeNode &node = aabbTree[nodeIndex];
                node.leftChild = ~left.geomIndex;
                node.rightChild = ~right.geomIndex;
                node.leftAABBMin = left.aabbMin;
                node.leftAABBMax = left.aabbMax;
                node.rightAABBMin = right.aabbMin;
                node.rightAABBMax = right.aabbMax;
            }
            break;
        default:
            {
                // compute centroid & aabb bounds
                glm::vec3 centroidMin = leaves[step.rangeBeg].centroid;
                glm::vec3 centroidMax = centroidMin;
                float outerHeuristic;
                {
                    glm::vec3
                        aabbMin = leaves[step.rangeBeg].aabbMin,
                        aabbMax = leaves[step.rangeBeg].aabbMax;
                    for (std::size_t i = step.rangeBeg + 1; i < step.rangeEnd; ++i) {
                        glm::vec3 centroid = leaves[i].centroid;
                        _leaf &cur = leaves[i];
                        centroidMin = glm::min(centroidMin, centroid);
                        centroidMax = glm::max(centroidMax, centroid);
                        aabbMin = glm::min(aabbMin, cur.aabbMin);
                        aabbMax = glm::max(aabbMax, cur.aabbMax);
                    }
                    outerHeuristic = surfaceAreaHeuristic(aabbMin, aabbMax);
                }
                // find split direction
                glm::vec3 centroidSpan = centroidMax - centroidMin;
                std::size_t splitDim = centroidSpan.x > centroidSpan.y ? 0 : 1;
                if (centroidSpan.z > centroidSpan[splitDim]) {
                    splitDim = 2;
                }
                // bucket nodes
                _bucket buckets[numBuckets];
                float bucketRange = centroidSpan[splitDim] / numBuckets;
                for (std::size_t i = step.rangeBeg; i < step.rangeEnd; ++i) {
                    leaves[i].bucket = static_cast<std::size_t>(glm::clamp(
                        (leaves[i].centroid[splitDim] - centroidMin[splitDim]) / bucketRange, 0.5f, numBuckets - 0.5f
                    ));
                    _bucket &buck = buckets[leaves[i].bucket];
                    _leaf &cur = leaves[i];
                    buck.aabbMin = glm::min(buck.aabbMin, cur.aabbMin);
                    buck.aabbMax = glm::max(buck.aabbMax, cur.aabbMax);
                    ++buck.count;
                }
                // find optimal split point
                _bucket boundCache[numBuckets - 1];
                {
                    _bucket current = buckets[numBuckets - 1];
                    for (std::size_t i = numBuckets - 1; i > 0; ) {
                        boundCache[--i] = current;
                        current = _bucket::merge(current, buckets[i]);
                    }
                }
                std::size_t optSplitPoint = 0;
                glm::vec3 leftMin, leftMax, rightMin, rightMax;
                {
                    float minHeuristic = FLT_MAX;
                    _bucket sumLeft;
                    for (std::size_t splitPoint = 0; splitPoint < numBuckets - 1; ++splitPoint) {
                        sumLeft = _bucket::merge(sumLeft, buckets[splitPoint]);
                        _bucket sumRight = boundCache[splitPoint];
                        float heuristic =
                            0.125f + (sumLeft.heuristic() + sumRight.heuristic()) / outerHeuristic;
                        if (heuristic < minHeuristic) {
                            minHeuristic = heuristic;
                            optSplitPoint = splitPoint;
                            leftMin = sumLeft.aabbMin;
                            leftMax = sumLeft.aabbMax;
                            rightMin = sumRight.aabbMin;
                            rightMax = sumRight.aabbMax;
                        }
                    }
                }
                // split
                std::size_t pivot = step.rangeBeg;
                for (std::size_t i = step.rangeBeg; i < step.rangeEnd; ++i) {
                    if (leaves[i].bucket <= optSplitPoint) {
                        std::swap(leaves[i], leaves[pivot++]);
                    }
                }
                int nodeIndex = alloc++;
                *step.parentPtr = nodeIndex;
                AABBTreeNode &n = aabbTree[nodeIndex];
                n.leftAABBMin = leftMin;
                n.leftAABBMax = leftMax;
                n.rightAABBMin = rightMin;
                n.rightAABBMax = rightMax;
                // handle duplicate triangles
                if (pivot == step.rangeBeg || pivot == step.rangeEnd) {
                    pivot = (step.rangeBeg + step.rangeEnd) / 2;
                }
                q.emplace_back(&n.leftChild, step.rangeBeg, pivot);
                q.emplace_back(&n.rightChild, pivot, step.rangeEnd);
            }
            break;
        }
    }
}

template <std::size_t N> void aabbForVerts(const glm::vec3 (&verts)[N], glm::vec3 *min, glm::vec3 *max) {
    *min = *max = verts[0];
    for (std::size_t i = 1; i < N; ++i) {
        *min = glm::min(*min, verts[i]);
        *max = glm::max(*max, verts[i]);
    }
}

template <std::size_t N> void aabbForVertsWithTransform(
    glm::vec3 (&verts)[N], glm::mat4x3 trans, glm::vec3 *min, glm::vec3 *max
) {
    for (std::size_t i = 0; i < N; ++i) {
        verts[i] = trans * glm::vec4(verts[i], 1.0f);
    }
    aabbForVerts(verts, min, max);
}

void Scene::aabbForGeom(const Geom &geom, glm::vec3 *min, glm::vec3 *max) {
    switch (geom.type) {
    case GeomType::CUBE:
        {
            glm::vec3 verts[8]{
                {  0.5f,  0.5f,  0.5f },
                {  0.5f,  0.5f, -0.5f },
                {  0.5f, -0.5f,  0.5f },
                {  0.5f, -0.5f, -0.5f },
                { -0.5f,  0.5f,  0.5f },
                { -0.5f,  0.5f, -0.5f },
                { -0.5f, -0.5f,  0.5f },
                { -0.5f, -0.5f, -0.5f }
            };
            aabbForVertsWithTransform(verts, geom.implicit.transform, min, max);
        }
        break;
    case GeomType::SPHERE:
        { // TODO https://stackoverflow.com/questions/4368961/calculating-an-aabb-for-a-transformed-sphere
            glm::vec3 verts[8]{
                {  0.5f,  0.5f,  0.5f },
                {  0.5f,  0.5f, -0.5f },
                {  0.5f, -0.5f,  0.5f },
                {  0.5f, -0.5f, -0.5f },
                { -0.5f,  0.5f,  0.5f },
                { -0.5f,  0.5f, -0.5f },
                { -0.5f, -0.5f,  0.5f },
                { -0.5f, -0.5f, -0.5f }
            };
            aabbForVertsWithTransform(verts, geom.implicit.transform, min, max);
        }
        break;
    case GeomType::TRIANGLE:
        aabbForVerts(geom.triangle.vertices, min, max);
        break;
    }
}
