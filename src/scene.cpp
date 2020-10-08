#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

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

void handleFace(string& token, vector<int> &posIndices) {
    char* tokenC = new char[token.length() + 1];
    strcpy(tokenC, token.c_str());
    char* c = strtok(tokenC, "/");
    posIndices.push_back(atoi(c));
    c = strtok(tokenC, "/");
    c = strtok(tokenC, "/");
}

// Based off of example code from https://github.com/tinyobjloader/tinyobjloader
int Scene::loadObj(std::vector<Geom>& triangles, int materialId, glm::mat4 transform, string inputName) {

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputName.c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        return -1;
    }

    glm::mat4 invTransform = glm::inverse(transform);
    glm::mat4 invTranspose = glm::inverseTranspose(transform);

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            // Loop over vertices in the face.
            std::vector<glm::vec3> face_pos;
            std::vector<glm::vec3> face_nor;
            std::vector<glm::vec2> face_uvs;
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                face_pos.push_back(glm::vec3(vx, vy, vz));

                tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
                face_nor.push_back(glm::vec3(nx, ny, nz));

                tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                face_uvs.push_back(glm::vec2(tx, ty));
            }

            index_offset += fv;
            // Make triangles out of these vertices
            for (int i = 1; i < face_pos.size() - 1; i++) {
                Geom triangle;
                triangle.type = TRIANGLE;
                triangle.tri.point1.pos = face_pos[0];
                triangle.tri.point1.nor = face_nor[0];
                triangle.tri.point1.uv = face_uvs[0];

                triangle.tri.point2.pos = face_pos[i];
                triangle.tri.point2.nor = face_nor[i];
                triangle.tri.point2.uv = face_uvs[i];

                triangle.tri.point3.pos = face_pos[i + 1];
                triangle.tri.point3.nor = face_nor[i + 1];
                triangle.tri.point3.uv = face_uvs[i + 1];

                triangle.materialid = materialId;
                triangle.transform = transform;
                triangle.inverseTransform = invTransform;
                triangle.invTranspose = invTranspose;

                triangles.push_back(triangle);
            }
        }
    }

    return 1;
}

#define LOADOBJ 1

#if LOADOBJ

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
	cout << "Loading Geom " << id << "..." << endl;
	Geom newGeom;
    std::vector<Geom> triangles;

    string line;
    string objFileName = "";
    int materialId = 0;
    glm::mat4 transform;
    bool loadingOBJ = false;

	//load object type
	utilityCore::safeGetline(fp_in, line);
	if (!line.empty() && fp_in.good()) {
		if (strcmp(line.c_str(), "sphere") == 0) {
			cout << "Creating new sphere..." << endl;
			newGeom.type = SPHERE;
		}
		else if (strcmp(line.c_str(), "cube") == 0) {
			cout << "Creating new cube..." << endl;
			newGeom.type = CUBE;
		}
        else if (strcmp(line.c_str(), "tanglecube") == 0) {
            cout << "Creating new tanglecube..." << endl;
            newGeom.type = IMPLICIT;
            newGeom.implicit.type = TANGLECUBE;
            newGeom.implicit.sdf = false;
            newGeom.implicit.shadowEpsilon = 0.07f;
        }
        else if (strcmp(line.c_str(), "torus") == 0) {
            cout << "Creating new torus..." << endl;
            newGeom.type = IMPLICIT;
            newGeom.implicit.type = TORUS;
            newGeom.implicit.sdf = true;
            newGeom.implicit.shadowEpsilon = 0.01f;
        }
		else if (strstr(line.c_str(), ".obj") != NULL) {
			cout << "Creating OBJ..." << endl;
            loadingOBJ = true;
            objFileName = line;
		}
	}

	//link material
	utilityCore::safeGetline(fp_in, line);
	if (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		newGeom.materialid = atoi(tokens[1].c_str());
        materialId = atoi(tokens[1].c_str());
		cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
	}

	//load transformations
	utilityCore::safeGetline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);

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

		utilityCore::safeGetline(fp_in, line);
	}

	newGeom.transform = utilityCore::buildTransformationMatrix(
		newGeom.translation, newGeom.rotation, newGeom.scale);
	newGeom.inverseTransform = glm::inverse(newGeom.transform);
	newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    transform = newGeom.transform;

    if (loadingOBJ) {
        loadObj(triangles, materialId, transform, objFileName);
        for (int i = 0; i < triangles.size(); i++) {
            geoms.push_back(triangles[i]);
        }
    }
    else {
        geoms.push_back(newGeom);
    }
	return 1;

}

#else 
int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strstr(line.c_str(), ".obj") != NULL) {
                cout << "Trying to read obj..." << endl;

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
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
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
#endif

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
        else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOCALDIST") == 0) {
            camera.focalDist = atof(tokens[1].c_str());
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

#define numProperties 8

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < numProperties; i++) {
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
            } else if (strcmp(tokens[0].c_str(), "DIFF") == 0) {
                newMaterial.hasDiffuse = atof(tokens[1].c_str());
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
