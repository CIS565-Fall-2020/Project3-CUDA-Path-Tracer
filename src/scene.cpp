#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
//#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

using namespace std;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "MESH") == 0) {
                loadMesh(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        obj_starts.push_back(geoms.size());
        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "GLOBAL") == 0) {
            this->globalLight = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
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
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadMesh(string objectid) {

    int id = atoi(objectid.c_str());
    
    cout << "Loading Mesh " << id << "..." << endl;
    string filename;
    string line;

    //load filename and get tinygltf model
    utilityCore::safeGetline(fp_in, filename);
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        printf("Warn %s \n", warn.c_str());
    }
    if (!err.empty()) {
        printf("Error %s \n", err.c_str());
    }
    if (!ret) {
        printf("Parsing Failed \n");
    }
    if (ret && warn.empty() && err.empty()) {
        printf("GLtf loading completed successfully \n");
    }

    //link material
    int materialId;
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        materialId = atoi(tokens[1].c_str());
        cout << "Connecting Mesh " << objectid << " to Material " << materialId << "..." << endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    glm::vec3 translate;
    glm::vec3 rotate;
    glm::vec3 scale;
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            translate = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            rotate = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    glm::mat4 transform = utilityCore::buildTransformationMatrix(translate, rotate, scale);
    obj_starts.push_back(geoms.size());
    for (tinygltf::Mesh& modelMesh : model.meshes) {
        for (tinygltf::Primitive& primitive : modelMesh.primitives) {
            // setup accessors for positions and normals
            const tinygltf::Accessor& accessorPos = model.accessors[primitive.attributes["POSITION"]];
            const tinygltf::BufferView& bufferViewPos = model.bufferViews[accessorPos.bufferView];
            const tinygltf::Buffer& bufferPos = model.buffers[bufferViewPos.buffer];

            const tinygltf::Accessor& accessorNor = model.accessors[primitive.attributes["NORMAL"]];
            const tinygltf::BufferView& bufferViewNor = model.bufferViews[accessorNor.bufferView];
            const tinygltf::Buffer& bufferNor = model.buffers[bufferViewNor.buffer];

            const tinygltf::Accessor& accessorIdx = model.accessors[primitive.attributes["INDEX"]];
            const tinygltf::BufferView& bufferViewIdx = model.bufferViews[accessorIdx.bufferView];
            const tinygltf::Buffer& bufferIdx = model.buffers[bufferViewIdx.buffer];

            const float* positions = reinterpret_cast<const float*>(&bufferPos.data[bufferViewPos.byteOffset + accessorPos.byteOffset]);
            const float* normals = reinterpret_cast<const float*>(&bufferNor.data[bufferViewNor.byteOffset + accessorNor.byteOffset]);
            const unsigned short* indices = reinterpret_cast<const unsigned short*>(&bufferIdx.data[bufferViewIdx.byteOffset + accessorIdx.byteOffset]);

            // From here, you choose what you wish to do with this position data. In this case, we  will display it out.
            for (size_t i = 0; i < accessorIdx.count; i += 3) {
                //int v1 = i * 3;
                //int v2 = (i + 1) * 3;
                //int v3 = (i + 2) * 3;
                int v1 = indices[i]*3;
                int v2 = indices[i + 1]*3;
                int v3 = indices[i + 2]*3;

                glm::vec3 pos1{ positions[v1], positions[v1 + 1], positions[v1 + 2] };
                glm::vec3 pos2{ positions[v2], positions[v2 + 1], positions[v2 + 2] };
                glm::vec3 pos3{ positions[v3], positions[v3 + 1], positions[v3 + 2] };

                glm::vec3 nor1{ normals[v1], normals[v1 + 1], normals[v1 + 2] };
                glm::vec3 nor2{ normals[v2], normals[v2 + 1], normals[v2 + 2] };
                glm::vec3 nor3{ normals[v3], normals[v3 + 1], normals[v3 + 2] };

                geoms.push_back(createTriangle(pos1, pos2, pos3, nor1, nor2, nor3, transform, materialId));
            }
        }
    }
    return 1;
}

Geom Scene::createTriangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3, glm::mat4& transform, int materialId) {
    glm::mat4 invTransposeTransform = glm::inverseTranspose(transform);
    
    Geom triangle;
    triangle.type = TRIANGLE;
    triangle.translation = glm::vec3(transform * glm::vec4(v1,1.f));
    triangle.rotation = glm::vec3(transform * glm::vec4(v2, 1.f));
    triangle.scale = glm::vec3(transform * glm::vec4(v3, 1.f));
    
    triangle.transform[0] = invTransposeTransform * glm::vec4(n1, 0.f);
    triangle.transform[1] = invTransposeTransform * glm::vec4(n2, 0.f);
    triangle.transform[2] = invTransposeTransform * glm::vec4(n3, 0.f);
    
    triangle.materialid = materialId;

    return triangle;
}