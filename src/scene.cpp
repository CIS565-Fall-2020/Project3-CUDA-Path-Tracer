#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tinyobj/tiny_obj_loader.h"

#define BUILD_RANDOM_SCENE 0

Scene::Scene(string filename) {
#if BUILD_RANDOM_SCENE
    buildRandomScene();
    return;
#endif

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
            } else if (strcmp(line.c_str(), "mesh") == 0) {
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;
                utilityCore::safeGetline(fp_in, line);
                if (!line.empty() && fp_in.good()) {
                    loadObj(line, newGeom);
                }
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
        newGeom.moving = false;
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "MOTION") == 0) {
                newGeom.moving = true;
                newGeom.target = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}


bool Scene::loadObj(string filename, Geom& geom) {
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;

    string warn;
    string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
    
    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    geom.startIndex = triangles.size();
    geom.leftBottom = glm::vec3(1000000);
    geom.rightTop = glm::vec3(-1000000);

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            Triangle t;

            int fv = shapes[s].mesh.num_face_vertices[f];
            tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + 0];
            glm::vec3 p1(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]);
            t.p1 = p1;
            if (attrib.normals.size() > 0) {
                t.n1 = glm::vec3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]);
            }
            geom.leftBottom = glm::min(geom.leftBottom, p1);
            geom.rightTop = glm::max(geom.rightTop, p1);

            idx = shapes[s].mesh.indices[index_offset + 1];
            glm::vec3 p2(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]);
            t.p2 = p2;
            if (attrib.normals.size() > 0) {
                t.n2 = glm::vec3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]);
            }
            geom.leftBottom = glm::min(geom.leftBottom, p2);
            geom.rightTop = glm::max(geom.rightTop, p2);

            idx = shapes[s].mesh.indices[index_offset + 2];
            glm::vec3 p3(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]);
            t.p3 = p3;
            if (attrib.normals.size() > 0) {
                t.n3 = glm::vec3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]);
            }
            geom.leftBottom = glm::min(geom.leftBottom, p3);
            geom.rightTop = glm::max(geom.rightTop, p3);

            if (attrib.normals.size() <= 0) {
                glm::vec3 n1 = glm::normalize(glm::cross(t.p2 - t.p1, t.p3 - t.p2));
                t.n1 = n1;
                t.n2 = n1;
                t.n3 = n1;
            }

            index_offset += fv;

            triangles.push_back(t);
        }
    }
    geom.endIndex = triangles.size() - 1;
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

int clamp(int x, int n) {
    x = x > n - 1 ? n - 1 : x;
    x = x < 0 ? 0 : x;
    return x;
}

void Scene::addSphereByMaterial(Geom& geom, int id, glm::vec3 trans, float radius) {
    geom.type = SPHERE;
    geom.materialid = id;
    geom.translation = trans;
    geom.rotation = glm::vec3(0);
    geom.scale = glm::vec3(radius);
    geom.moving = false;

    geom.transform = utilityCore::buildTransformationMatrix(
        geom.translation, geom.rotation, geom.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
}


void Scene::buildRandomScene() {
    // camera
    cout << "Loading Camera ..." << endl;
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    camera.resolution.x = 800;
    camera.resolution.y = 600;
    fovy = 20;
    state.iterations = 5000;
    state.traceDepth = 8;
    camera.position = glm::vec3(12, 1.6, 5);
    camera.lookAt = glm::vec3(0, 1.6, 0);
    camera.up = glm::vec3(0, 1, 0);


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

    Material groundMaterial = { glm::vec3(0.5, 0.5, 0.5), {0, glm::vec3(0.5, 0.5, 0.5) }, 0, 0, 0, 0 };
    materials.push_back(groundMaterial);

    Geom groundSphere;
    addSphereByMaterial(groundSphere, 0, glm::vec3(0, -1000, 0), 2000);
    geoms.push_back(groundSphere);

    Material material1 = { glm::vec3(1), {0, glm::vec3(1) }, 0, 1, 1.5, 0 };
    materials.push_back(material1);
    Geom sphere1;
    addSphereByMaterial(sphere1, 1, glm::vec3(0, 1, 0), 2);
    geoms.push_back(sphere1);

    Material material2 = { glm::vec3(0.91, 0.91, 0.51), {0, glm::vec3(0) }, 0, 0, 0, 0 };
    materials.push_back(material2);
    Geom sphere2;
    addSphereByMaterial(sphere2, 2, glm::vec3(-4, 1, 0), 2);
    geoms.push_back(sphere2);

    Material material3 = { glm::vec3(0.7, 0.6, 0.5), {0, glm::vec3(0.7, 0.6, 0.5) }, 1, 0, 0, 0 };
    materials.push_back(material3);
    Geom sphere3;
    addSphereByMaterial(sphere3, 3, glm::vec3(4, 1, 0), 2);
    geoms.push_back(sphere3);

    Material material4 = { glm::vec3(0.85, 0.5, 0.67), {0, glm::vec3(0) }, 0, 0, 0, 0 };
    materials.push_back(material4);

    Geom bunny;
    bunny.type = MESH;
    bunny.materialid = 4;
    bunny.translation = glm::vec3(5, 0, 3);
    bunny.rotation = glm::vec3(0, 90, 0);
    bunny.scale = glm::vec3(5);
    bunny.moving = false;
    loadObj("../scenes/models/bunny.obj", bunny);

    bunny.transform = utilityCore::buildTransformationMatrix(
        bunny.translation, bunny.rotation, bunny.scale);
    bunny.inverseTransform = glm::inverse(bunny.transform);
    bunny.invTranspose = glm::inverseTranspose(bunny.transform);
    geoms.push_back(bunny);

    int matId = 5;
    for (int a = -9; a < 9; a++) {
        for (int b = -9; b < 9; b++) {
            float choose_mat = rand() / (RAND_MAX + 1.0);
            float choose_motion = rand() / (RAND_MAX + 1.0);
            glm::vec3 center(a * 1.5 + 0.9 * rand() / (RAND_MAX + 1.0), 0.2, b * 1.5 + 0.9 * rand() / (RAND_MAX + 1.0));

            Material material;
            if (glm::length(center - glm::vec3(0, 0.2, 0)) > 1 &&
                glm::length(center - glm::vec3(-4, 0.2, 0)) > 1 &&
                glm::length(center - glm::vec3(4, 0.2, 0)) > 1) {
                if (choose_mat < 0.75) {
                    // diffuse
                    glm::vec3 color(rand() / (RAND_MAX + 1.0), rand() / (RAND_MAX + 1.0), rand() / (RAND_MAX + 1.0));
                    Material material = { color, {0, color}, 0, 0, 0, 0 };
                    materials.push_back(material);

                }
                else if (choose_mat < 0.82) {
                    // emmision
                    glm::vec3 color(rand() / (RAND_MAX + 1.0) * 0.1 + 0.9, rand() / (RAND_MAX + 1.0) * 0.1 + 0.9, rand() / (RAND_MAX + 1.0) * 0.1 + 0.9);
                    material = { color, {0, color}, 0, 0, 0, 1 };
                    materials.push_back(material);
                }
                else if (choose_mat < 0.95) {
                    // metal
                    glm::vec3 color(rand() / (RAND_MAX + 1.0) * 0.5 + 0.5, rand() / (RAND_MAX + 1.0) * 0.5 + 0.5, rand() / (RAND_MAX + 1.0) * 0.5 + 0.5);
                    material = { color, {0, color}, 1, 0, 0, 0 };
                    materials.push_back(material);
                }
                else {
                    // glass
                    glm::vec3 color(rand() / (RAND_MAX + 1.0) * 0.1 + 0.9, rand() / (RAND_MAX + 1.0) * 0.1 + 0.9, rand() / (RAND_MAX + 1.0) * 0.1 + 0.9);
                    material = { color, {0, color}, 0, 1, 1.5, 0 };
                    materials.push_back(material);
                }
                Geom sphere;
                addSphereByMaterial(sphere, matId++, center, 0.4);
                if (choose_motion < 0.2 && choose_mat < 0.95) {
                    sphere.moving = true;
                    sphere.target = sphere.translation + glm::vec3(0, 0.2, 0);
                }
                geoms.push_back(sphere);
            }
        }
    }
}
