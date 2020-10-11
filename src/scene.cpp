#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <stb_image_write.h>
#include <stb_image.h>
#include "tiny_obj_loader.h"
#include <glm/gtx/transform.hpp> 

#define DEBUG 0

/*
 * Case Sensitive Implementation of endsWith()
 * It checks if the string 'mainStr' ends with given string 'toMatch'
 * Helper Function from https://thispointer.com/c-how-to-check-if-a-string-ends-with-an-another-given-string/#:~:text=To%20check%20if%20a%20main,%E2%80%93%20size%20of%20given%20string).
 */
bool endsWith(const std::string& mainStr, const std::string& toMatch)
{
    if (mainStr.size() >= toMatch.size() &&
        mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
        return true;
    else
        return false;
}

// This function is modified from the one we were given in CIS 460
BoundingBox Scene::loadOBJ(string filename, int material_id)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string error;
    std::string base_dir = "";
    if (filename.find_last_of("/\\") != std::string::npos) {
        base_dir = filename.substr(0, filename.find_last_of("/\\"));
    }
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &error, filename.c_str(), base_dir.c_str());
    // if no errors, we continue...
    if (error.empty()) {

        // initialize bounding box with default values
        BoundingBox bb;
        bb.min = glm::vec3(std::numeric_limits<float>::max());
        bb.max = glm::vec3(std::numeric_limits<float>::min());

        for (size_t s = 0; s < shapes.size(); s++) {

            // for debugging
            int triangle_count = 0;
            // loop over every triangle
            int triangle_index = 3;
            size_t index_offset = 0; // this is the current count. We are counting in 3s
            for (size_t i = 0; i < shapes[s].mesh.num_face_vertices.size(); i++) {

                triangle_count++;
#if DEBUG
                cout << "Generating Triangle #" << triangle_count << endl;
#endif

                Geom new_geom;
                new_geom.type = TRIANGLE;
                new_geom.materialid = material_id;

                // loop through the three vertices of each triangle
                for (size_t j = 0; j < triangle_index; j++) {
                    // get vertex index
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + j];

                    // vertex's position
                    glm::vec3 idx_pos(attrib.vertices[3 * idx.vertex_index + 0], 
                                      attrib.vertices[3 * idx.vertex_index + 1], 
                                      attrib.vertices[3 * idx.vertex_index + 2]);
                    idx_pos *= 4.0f;
                    idx_pos.z -= 2;
                    idx_pos.y += 2;
                    idx_pos.x -= 2;


                    // vertex's normal
                    glm::vec3 idx_nor(0, 0, 0);
                    if (attrib.normals.size() > 0) {
                        idx_nor = glm::vec3(attrib.normals[3 * idx.normal_index + 0], 
                                            attrib.normals[3 * idx.normal_index + 1], 
                                            attrib.normals[3 * idx.normal_index + 2]);
                    }
                    // vertex's uv
                    glm::vec2 idx_uv(0, 0);
                    if (attrib.texcoords.size() > 0) {
                        new_geom.t.has_texture = true;
                        idx_uv = glm::vec2(attrib.texcoords[2 * idx.texcoord_index + 0], 
                                           attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }

                    // fill the triangle with this index's info
                    new_geom.t.pos[j] = idx_pos;
                    new_geom.t.nor[j] = idx_nor;
                    new_geom.t.uv[j] = idx_uv;

                    // check for bounding box values
                    for (int k = 0; k < 3; k++) {
                        if (idx_pos[k] < bb.min[k]) {
                            bb.min[k] = idx_pos[k];
                        }
                        if (idx_pos[k] > bb.max[k]) {
                            bb.max[k] = idx_pos[k];
                        }
                    }

#if DEBUG
                    cout << "= index #" << j << endl;
                    cout << "=== pos: " << idx_pos.x << ", " << idx_pos.y << ", " << idx_pos.z << endl;
                    cout << "=== nor: " << idx_nor.x << ", " << idx_nor.y << ", " << idx_nor.z << endl;
                    cout << "=== uv: " << idx_uv.x << ", " << idx_uv.y << endl;
#endif
                }
                index_offset += triangle_index;

                // add this new triangle
                geoms.push_back(new_geom);
            }
        }
        return bb;
#if false
        cout << "FINAL BOUNDING BOX" << endl;
        cout << "= min pos: " << bb.min.x << ", " << bb.min.y << ", " << bb.min.z << endl;
        cout << "= max pos: " << bb.max.x << ", " << bb.max.y << ", " << bb.max.z << endl;
#endif
    }

}

bool Scene::loadTexture(Material& newMaterial, string path, bool bump) {
    int w = 0; // width of texture
    int h = 0; // height of texure
    int comp = 0; //
    float* rawPixels = stbi_loadf(path.c_str(), &w, &h, &comp, 3);
    // we only want to operate on the texture if it is rgb or rgba
    if (comp == 3 || comp == 4)
    {
        // since we are storing all our texture pixel colors in one vector
        // we have to index them, in case another material has a different
        // texture that it is linked to
        if (bump) {
            newMaterial.tex_bump_index = texture.size();
            newMaterial.tex_bump_height = h;
            newMaterial.tex_bump_width = w;
        }
        else {
            newMaterial.tex_index = texture.size();
            newMaterial.tex_height = h;
            newMaterial.tex_width = w;
        }
        newMaterial.has_bump_map = bump;
        
        // loop through and push the color of each pixel into the vector
        for (int i = 0; i < w * h; i++)
        {
            glm::vec3 color (rawPixels[i * comp], 
                rawPixels[i * comp + 1], 
                rawPixels[i * comp + 2]);
#if DEBUG
            //cout << "color @ index:" << i << ", r: " << color.x << ", g: " << color.y << ", b:" << color.z << endl;
#endif
            texture.push_back(color);
        }
        std::cout << "Loaded texture! Texture Path:" << path << "\" width: " << w << ", height: " << h << std::endl;
        stbi_image_free(rawPixels);
        return true;
    }
    std::cout << "Error: Could not load texture" << path << std::endl;
    stbi_image_free(rawPixels);
    return false;
}

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    if (endsWith(filename.c_str(), ".obj")) {
        cout << "Loading Obj File" << endl;
        this->bounding_box = loadOBJ(filename, 0);

        // make material
        Material newMaterial;
        newMaterial.color = glm::vec3(1.f, 1.f, 1.f);
        newMaterial.specular.exponent = 1.f;
        newMaterial.specular.color = glm::vec3(1.f, 1.f, 1.f);
        newMaterial.hasReflective = 0.f;
        newMaterial.hasRefractive = 0.f;
        newMaterial.indexOfRefraction = 1.f;
        newMaterial.emittance = 0.f;

        // load color texture
        //loadTexture(newMaterial, "../scenes/default.png", false);
        loadTexture(newMaterial, "../scenes/bump2.png", true);
        //cout << "bump h and w: " << newMaterial.tex_bump_width << ", " << newMaterial.tex_bump_height << endl;
        materials.push_back(newMaterial);

        // make camera
        RenderState& state = this->state;
        Camera& camera = state.camera;
        camera.resolution.x = 800;
        camera.resolution.y = 800;
        float fovy = 45;
        state.iterations = 5000;
        state.traceDepth = 8;
        state.imageName = "obj";
        camera.position = glm::vec3(0.0f, 0.f, 10.f);
        camera.lookAt = glm::vec3(0.f, 1.f, 0.f);
        camera.up = glm::vec3(0.f, 1.0f, 0.f);
        float yscaled = tan(fovy * (PI / 180));
        float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
        float fovx = (atan(xscaled) * 180) / PI;
        camera.fov = glm::vec2(fovx, fovy);
        camera.right = glm::normalize(glm::cross(camera.view, camera.up));
        camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
            , 2 * yscaled / (float)camera.resolution.y);

        camera.view = glm::normalize(camera.lookAt - camera.position);
        int arraylen = camera.resolution.x * camera.resolution.y;
        state.image.resize(arraylen);
        std::fill(state.image.begin(), state.image.end(), glm::vec3());

        cout << "Loaded camera!" << endl;

        // add additional cornell box
        filename = "../scenes/environment.txt";
    }
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
                }
                else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                    loadGeom(tokens[1]);
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

        geoms.push_back(newGeom);
        return 1;
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
            if (id == 2) {
                //loadTexture(newMaterial, "../scenes/default.png", false);
                newMaterial.is_procedural = true;
            }
            if (id == 3) {
                loadTexture(newMaterial, "../scenes/floral.jpg", false);
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
