#include "scene.h"
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

Scene::Scene(string filename) 
    : total_faces(0), total_vertices(0)
{
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

Scene::~Scene()
{}

int Scene::getMeshesSize() const
{
    return meshes.size();
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size())
    {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } 
    else
    {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        std::vector<Geom> gltfGeoms;
        bool geomFromGltf = false;

        string line;

        // load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0)
            {
                cout << "Creating new sphere..." << endl;
                newGeom.type = GeomType::SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0)
            {
                cout << "Creating new cube..." << endl;
                newGeom.type = GeomType::CUBE;
            }
            else if (strcmp(line.c_str(), "mesh") == 0)
            {
                cout << "Creating new mesh..." << endl;
                geomFromGltf = true;

                utilityCore::safeGetline(fp_in, line);
                const std::string gltf_file = line;
                bool ret = LoadGLTF(gltf_file, 1, &(this->meshes), nullptr, nullptr,
                    &this->total_faces, &this->total_vertices);
                if (!ret)
                {
                    std::cerr << "Failed to load glTF file [ " << gltf_file << " ]" << std::endl;
                    return -1;
                }
                
                for (int i = 0; i < meshes.size(); i++)
                {
                    Geom meshGeom;
                    meshGeom.type = GeomType::MESH;

                    meshGeom.transform[0][0] = meshes[i].transform[0][0];
                    meshGeom.transform[0][1] = meshes[i].transform[0][1];
                    meshGeom.transform[0][2] = meshes[i].transform[0][2];
                    meshGeom.transform[0][3] = meshes[i].transform[0][3];

                    meshGeom.transform[1][0] = meshes[i].transform[1][0];
                    meshGeom.transform[1][1] = meshes[i].transform[1][1];
                    meshGeom.transform[1][2] = meshes[i].transform[1][2];
                    meshGeom.transform[1][3] = meshes[i].transform[1][3];

                    meshGeom.transform[2][0] = meshes[i].transform[2][0];
                    meshGeom.transform[2][1] = meshes[i].transform[2][1];
                    meshGeom.transform[2][2] = meshes[i].transform[2][2];
                    meshGeom.transform[2][3] = meshes[i].transform[2][3];

                    meshGeom.transform[3][0] = meshes[i].transform[3][0];
                    meshGeom.transform[3][1] = meshes[i].transform[3][1];
                    meshGeom.transform[3][2] = meshes[i].transform[3][2];
                    meshGeom.transform[3][3] = meshes[i].transform[3][3];

                    gltfGeoms.push_back(meshGeom);
                }
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good())
        {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (geomFromGltf)
            {
                for (Geom& gltfGeom : gltfGeoms)
                {
                    gltfGeom.materialid = atoi(tokens[1].c_str());
                }
            }
            else
            {
                newGeom.materialid = atoi(tokens[1].c_str());
                cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
            }
        }

        //load transformations
        glm::vec3 tempTranslate(0);
        glm::vec3 tempRotate(0);
        glm::vec3 tempScale(1);

        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                tempTranslate = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                tempRotate = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                tempScale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        glm::mat4 totalTransform = utilityCore::buildTransformationMatrix(tempTranslate, tempRotate, tempScale);
        if (geomFromGltf)
        {
            for (Geom& gltfGeom : gltfGeoms)
            {
                gltfGeom.translation = tempTranslate;
                gltfGeom.rotation = tempRotate;
                gltfGeom.scale = tempScale;
                setGeomTransform(&gltfGeom, totalTransform * gltfGeom.transform);
                geoms.push_back(gltfGeom);
            }
        }
        else
        {
            newGeom.translation = tempTranslate;
            newGeom.rotation = tempRotate;
            newGeom.scale = tempScale;
            setGeomTransform(&newGeom, totalTransform);
            geoms.push_back(newGeom);
        }

        
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
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
            }
            else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            }
            else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}