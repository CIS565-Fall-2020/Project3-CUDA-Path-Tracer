#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include "gltf-loader.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }

    faceCount = 0;
    meshCount = 0;
    posCount = 0;

    octree = Octree();

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

    octree.pointerize();

    for (int i = 0; i < geoms.size(); i++) 
    {
        int mId = geoms.at(i).materialid;
        if (mId < materials.size() && materials[mId].emittance > 0.0f) 
        {
            lights.push_back(geoms.at(i));
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
        std::vector<Geom> meshGeoms;
        std::vector<example::Mesh<float>> curMesh;
        bool isMesh = false;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
                //newGeom.modelId = -1;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
                //newGeom.modelId = -1;
            }
            else if (strcmp(line.c_str(), "mesh") == 0) 
            {
                isMesh = true;
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;

                // Load Mesh
                cout << "Load new mesh..." << endl;
                utilityCore::safeGetline(fp_in, line);

                cout << "Loading new mesh..." << endl;

                bool isLoaded = example::LoadGLTF(line, 1.0f, &curMesh, &gltfMaterials, &gltfTextures);

                if (!isLoaded) 
                {
                    std::cout << "Load mesh failed!" << std::endl;
                    return -1;
                }

               
                //newGeom.offset = faceCount;

                meshes.push_back(curMesh);

                int curFaceNum = 0;
                int curPosNum = 0;

                for (int i = 0; i < curMesh.size(); i++) 
                {
                    Geom curGeom;
                    curGeom.type = MESH;
                    curGeom.offset = curFaceNum;
                    curGeom.posOffset = curPosNum;
                    curGeom.faceNum = curMesh.at(i).faces.size() / 3;
                    curGeom.posNum = curMesh.at(i).vertices.size();

                    curGeom.materialid = curMesh.at(i).material_ids + materials.size();

                    glm::vec3 curTranslation = glm::vec3(curMesh.at(i).localTranslate[0], 
                                                         curMesh.at(i).localTranslate[1],
                                                         curMesh.at(i).localTranslate[2]);
                    glm::vec3 curRotation = glm::vec3(curMesh.at(i).localRotation[0],
                                                      curMesh.at(i).localRotation[1],
                                                      curMesh.at(i).localRotation[2]);
                    glm::vec3 curScale = glm::vec3(curMesh.at(i).localScale[0],
                                                   curMesh.at(i).localScale[1],
                                                   curMesh.at(i).localScale[2]);

                    curGeom.transform = utilityCore::buildTransformationMatrix(
                        curTranslation, curRotation, curScale);
                    BoundingBox bb;

                    bb.boundingCenter = glm::vec3(curMesh.at(i).center[0], curMesh.at(i).center[1], curMesh.at(i).center[2]);
                    bb.boundingScale = glm::vec3(curMesh.at(i).scale[0], curMesh.at(i).scale[1], curMesh.at(i).scale[2]);

                    curGeom.boundingIdx = boundingBoxes.size();

                    boundingBoxes.push_back(bb);

                    curFaceNum += curMesh.at(i).faces.size() / 3;
                    curPosNum += curMesh.at(i).vertices.size();

                    meshGeoms.push_back(curGeom);
                }
                faceCount += curFaceNum;
                posCount += curPosNum;
                //newGeom.faceNum = curFaceNum;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (!isMesh) 
            {
                newGeom.materialid = atoi(tokens[1].c_str());
                cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
                /*for (int i = 0; i < meshGeoms.size(); i++) 
                {
                    meshGeoms.at(i).materialid = atoi(tokens[1].c_str());
                    cout << "Connecting Geom " << objectid << "  Mesh " << i << " to Material " << newGeom.materialid << "..." << endl;
                }*/
            }
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (isMesh) 
            {
                // Load Mesh Transforms
                for (int i = 0; i < meshGeoms.size(); i++) 
                {
                    if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                        meshGeoms.at(i).translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                    }
                    else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                        meshGeoms.at(i).rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                    }
                    else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                        meshGeoms.at(i).scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                    }
                }
            }
            else 
            {
                //load tranformations
                if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                    newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                }
                else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                    newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                }
                else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                    newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                }
            }
            

            utilityCore::safeGetline(fp_in, line);
        }

        if (isMesh) 
        {
            for (int i = 0; i < meshGeoms.size(); i++) 
            {
                meshGeoms.at(i).transform *= utilityCore::buildTransformationMatrix(
                    meshGeoms.at(i).translation, meshGeoms.at(i).rotation, meshGeoms.at(i).scale);
                meshGeoms.at(i).inverseTransform = glm::inverse(meshGeoms.at(i).transform);
                meshGeoms.at(i).invTranspose = glm::inverseTranspose(meshGeoms.at(i).transform);

                Geom curGeom = meshGeoms.at(i);


                // Add into octree
                for (int j = 0; j < curGeom.faceNum; j++)
                {
                    int xIndex = curMesh.at(i).faces.at(3 * j);
                    int yIndex = curMesh.at(i).faces.at(3 * j + 1);
                    int zIndex = curMesh.at(i).faces.at(3 * j + 2);

                    glm::vec3 posX = glm::vec3(curGeom.transform *
                        glm::vec4(curMesh.at(i).vertices.at(xIndex),
                            curMesh.at(i).vertices.at(xIndex + 1),
                            curMesh.at(i).vertices.at(xIndex + 2), 1.0f));

                    glm::vec3 posY = glm::vec3(curGeom.transform *
                        glm::vec4(curMesh.at(i).vertices.at(yIndex),
                            curMesh.at(i).vertices.at(yIndex + 1),
                            curMesh.at(i).vertices.at(yIndex + 2), 1.0f));

                    glm::vec3 posZ = glm::vec3(curGeom.transform *
                        glm::vec4(curMesh.at(i).vertices.at(zIndex),
                            curMesh.at(i).vertices.at(zIndex + 1),
                            curMesh.at(i).vertices.at(zIndex + 2), 1.0f));

                    int faceIndex = curGeom.offset + j;

                    MeshTri curTri;
                    curTri.x = posX;
                    curTri.y = posY;
                    curTri.z = posZ;
                    curTri.faceIndex = faceIndex;
                    curTri.transform = curGeom.transform;
                    curTri.inverseTransform = curGeom.inverseTransform;
                    curTri.invTranspose = curGeom.invTranspose;

                    octree.insertMeshTri(curTri);
                }


                geoms.push_back(meshGeoms.at(i));
            }
        }
        else 
        {
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
        
        //Insert into octree
        octree.insertPrim(geoms.size() - 1, newGeom);

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
