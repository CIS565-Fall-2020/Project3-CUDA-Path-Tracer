#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
#include "gltf-loader.h" // get example namespace

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
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
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
            // Jack12
            else if (strcmp(line.c_str(), "gltf_mesh") == 0) {
                cout << "Creating new gltf mesh..." << endl;
                newGeom.type = GLTF_MESH;
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
        std::string cur_path;
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
            // or path for gltf-mesh
            //PATH should put to last
            else if (strcmp(tokens[0].c_str(), "PATH") == 0) {
                cur_path = tokens[0];
                
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        if (newGeom.type != GLTF_MESH) {
            geoms.push_back(newGeom);
        }
        else {
            this->loadGLTFMesh(cur_path, newGeom);
        }
        
        return 1;
    }
}

int Scene::loadGLTFMesh(const std::string& file_path, const Geom& parent_geom) {
    // ref https://github.com/syoyo/tinygltf/blob/master/examples/glview/glview.cc
    std::cout << "read gltf mesh from " << file_path << std::endl;

    std::vector<Geom> trianges;
    // ty gktf-loader
    std::vector<example::Material> gltf_materials;
    std::vector<example::Mesh<float> > gltf_meshes;
    std::vector<example::Texture> gltf_textures;
    // ref https://github.com/syoyo/tinygltf/blob/master/examples/raytrace/main.cc, 
    // ref https://github.com/taylornelms15/Project3-CUDA-Path-Tracer/blob/master/src/scene.cpp
    bool flag = false;
    flag = example::LoadGLTF(file_path, 1.0f, &gltf_meshes, &gltf_materials,&gltf_textures);

    if (!flag) {
        std::cout << "Failed to load glTF file "
            << std::endl;
        return -1;
    }

    if (gltf_textures.size() > 0) {
        std::cout << "there has " << gltf_meshes.size() << " meshes." << std::endl;
        for (auto cur_mesh = gltf_meshes.begin(); cur_mesh != gltf_meshes.end(); cur_mesh++) {
            GLTF_Model cur_model;
            glm::vec3 maxVal_vec(-INFINITY, -INFINITY, -INFINITY);
            glm::vec3 minVal_vec(INFINITY, INFINITY, INFINITY);

            for (int i = 0; i < cur_mesh->faces.size(); i+=3) {
                
                Triangle cur_triangle;
                cur_model.self_geom.type = TRIANGLE;

                int idx_f0 = i;
                int idx_f1 = i + 1;
                int idx_f2 = i + 2;

                int idx_v0 = cur_mesh->faces[idx_f0];
                int idx_v1 = cur_mesh->faces[idx_f1];
                int idx_v2 = cur_mesh->faces[idx_f2];

                cur_triangle.v0 = glm::vec3(
                    cur_mesh -> vertices[3 * idx_v0],
                    cur_mesh -> vertices[3 * idx_v0 + 1],
                    cur_mesh -> vertices[3 * idx_v0 + 2]
                );

                cur_triangle.v1 = glm::vec3(
                    cur_mesh->vertices[3 * idx_v1],
                    cur_mesh->vertices[3 * idx_v1 + 1],
                    cur_mesh->vertices[3 * idx_v1 + 2]
                );

                cur_triangle.v2 = glm::vec3(
                    cur_mesh->vertices[3 * idx_v2],
                    cur_mesh->vertices[3 * idx_v2 + 1],
                    cur_mesh->vertices[3 * idx_v2 + 2]
                );

                cur_triangle.n0 = glm::vec3(
                    cur_mesh->facevarying_normals[3 * idx_v0],
                    cur_mesh->facevarying_normals[3 * idx_v0 + 1],
                    cur_mesh->facevarying_normals[3 * idx_v0 + 2]
                );

                cur_triangle.n1 = glm::vec3(
                    cur_mesh->facevarying_normals[3 * idx_v1],
                    cur_mesh->facevarying_normals[3 * idx_v1 + 1],
                    cur_mesh->facevarying_normals[3 * idx_v1 + 2]
                );

                cur_triangle.n2 = glm::vec3(
                    cur_mesh->facevarying_normals[3 * idx_v2],
                    cur_mesh->facevarying_normals[3 * idx_v2 + 1],
                    cur_mesh->facevarying_normals[3 * idx_v2 + 2]
                );

                //cur_triangle.norm = glm::triangleNormal()
                cur_model.triangles.emplace_back(cur_triangle);
                cur_model.self_geom = parent_geom;
                // assign bounding box
                //TODO
                minVal_vec = glm::min(minVal_vec, cur_triangle.v0);
                minVal_vec = glm::min(minVal_vec, cur_triangle.v1);
                minVal_vec = glm::min(minVal_vec, cur_triangle.v2);

                maxVal_vec = glm::max(maxVal_vec, cur_triangle.v0);
                maxVal_vec = glm::max(maxVal_vec, cur_triangle.v1);
                maxVal_vec = glm::max(maxVal_vec, cur_triangle.v2);
            }
            cur_model.bbox_geom = parent_geom;
            cur_model.bbox_geom.scale = maxVal_vec - minVal_vec;
            cur_model.bbox_geom.translation = maxVal_vec / 2.0f + minVal_vec / 2.0f;
            cur_model.bbox_geom.rotation = glm::vec3(0.0f);

            cur_model.bbox_geom.transform = utilityCore::buildTransformationMatrix(
                cur_model.bbox_geom.translation, cur_model.bbox_geom.rotation, cur_model.bbox_geom.scale);
            cur_model.bbox_geom.inverseTransform = glm::inverse(cur_model.bbox_geom.transform);
            cur_model.bbox_geom.invTranspose = glm::inverseTranspose(cur_model.bbox_geom.transform);

            this->gltf_models.emplace_back(cur_model);
        }
    }
    return 0;
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
        }
        // Jack12 add camera aperture radius and focusDistance
        else if (strcmp(tokens[0].c_str(), "ApRds") == 0) {
            camera.apertureRadius = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FD") == 0) {
            camera.focusDist = atof(tokens[1].c_str());
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
