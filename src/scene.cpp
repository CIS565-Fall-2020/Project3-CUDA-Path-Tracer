#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include <tiny_gltf.h>


Scene::Scene(string filename) {
    std::cout << "Reading scene from " << filename << " ..." << endl;
    std::cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                std::cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                std::cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                std::cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        std::cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        std::cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                std::cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                std::cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            // Jack12
            else if (strcmp(line.c_str(), "gltf_mesh") == 0) {
                std::cout << "Creating new gltf mesh..." << endl;
                newGeom.type = GLTF_MESH;
            }

        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
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
                cur_path = tokens[1];
                
            }
            // motion blur
            else if (strcmp(tokens[0].c_str(), "VELO") == 0) {
                newGeom.velocity = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        
        if (newGeom.type == GLTF_MESH) {
            this->loadGLTFMesh(cur_path, newGeom);
        }
        else {
            // gltf would push back its bbox
            geoms.push_back(newGeom);
        }

        return 1;
    }
}

static std::string GetFilePathExtension(const std::string& FileName) {
    if (FileName.find_last_of(".") != std::string::npos)
        return FileName.substr(FileName.find_last_of(".") + 1);
    return "";
}

//copy from gltf-loader.cc stupid gltf
bool Scene::myGLTFloader(
    const std::string& file_path, 
    float scale,
    std::vector<example::Mesh<float>>& meshes,
    std::vector<example::Material>& materials,
    std::vector<example::Texture>& textures) {
    using namespace example;
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    const std::string ext =  GetFilePathExtension(file_path);
    
    bool ret = false;
    if (ext.compare("glb") == 0) {
        // assume binary glTF.
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, file_path.c_str());
    }
    else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, file_path.c_str());
    }

    if (!warn.empty()) {
        std::cout << "glTF parse warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "glTF parse error: " << err << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load glTF: " << file_path << std::endl;
        return false;
    }

    std::cout << "loaded glTF file has:\n"
        << model.accessors.size() << " accessors\n"
        << model.animations.size() << " animations\n"
        << model.buffers.size() << " buffers\n"
        << model.bufferViews.size() << " bufferViews\n"
        << model.materials.size() << " materials\n"
        << model.meshes.size() << " meshes\n"
        << model.nodes.size() << " nodes\n"
        << model.textures.size() << " textures\n"
        << model.images.size() << " images\n"
        << model.skins.size() << " skins\n"
        << model.samplers.size() << " samplers\n"
        << model.cameras.size() << " cameras\n"
        << model.scenes.size() << " scenes\n"
        << model.lights.size() << " lights\n";

    // Iterate through all the meshes in the glTF file
    for (const auto& gltfMesh : model.meshes) {
        std::cout << "Current mesh has " << gltfMesh.primitives.size()
            << " primitives:\n";

        // Create a mesh object
        Mesh<float> loadedMesh(sizeof(float) * 3);

        // To store the min and max of the buffer (as 3D vector of floats)
        v3f pMin = {}, pMax = {};

        // Store the name of the glTF mesh (if defined)
        loadedMesh.name = gltfMesh.name;

        // For each primitive
        for (const auto& meshPrimitive : gltfMesh.primitives) {
            // Boolean used to check if we have converted the vertex buffer format
            bool convertedToTriangleList = false;
            // This permit to get a type agnostic way of reading the index buffer
            std::unique_ptr<intArrayBase> indicesArrayPtr = nullptr;
            {
                const auto& indicesAccessor = model.accessors[meshPrimitive.indices];
                const auto& bufferView = model.bufferViews[indicesAccessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                const auto dataAddress = buffer.data.data() + bufferView.byteOffset +
                    indicesAccessor.byteOffset;
                const auto byteStride = indicesAccessor.ByteStride(bufferView);
                const auto count = indicesAccessor.count;

                // Allocate the index array in the pointer-to-base declared in the
                // parent scope
                switch (indicesAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_BYTE:
                    indicesArrayPtr =
                        std::unique_ptr<intArray<char> >(new intArray<char>(
                            arrayAdapter<char>(dataAddress, count, byteStride)));
                    break;

                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                    indicesArrayPtr = std::unique_ptr<intArray<unsigned char> >(
                        new intArray<unsigned char>(arrayAdapter<unsigned char>(
                            dataAddress, count, byteStride)));
                    break;

                case TINYGLTF_COMPONENT_TYPE_SHORT:
                    indicesArrayPtr =
                        std::unique_ptr<intArray<short> >(new intArray<short>(
                            arrayAdapter<short>(dataAddress, count, byteStride)));
                    break;

                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                    indicesArrayPtr = std::unique_ptr<intArray<unsigned short> >(
                        new intArray<unsigned short>(arrayAdapter<unsigned short>(
                            dataAddress, count, byteStride)));
                    break;

                case TINYGLTF_COMPONENT_TYPE_INT:
                    indicesArrayPtr = std::unique_ptr<intArray<int> >(new intArray<int>(
                        arrayAdapter<int>(dataAddress, count, byteStride)));
                    break;

                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                    indicesArrayPtr = std::unique_ptr<intArray<unsigned int> >(
                        new intArray<unsigned int>(arrayAdapter<unsigned int>(
                            dataAddress, count, byteStride)));
                    break;
                default:
                    break;
                }
            }
            const auto& indices = *indicesArrayPtr;

            if (indicesArrayPtr) {
                std::cout << "indices: omit by annotaion by Jack12";
                for (size_t i(0); i < indicesArrayPtr->size(); ++i) {
                    //std::cout << indices[i] << " ";
                    loadedMesh.faces.push_back(indices[i]);
                }
                std::cout << '\n';
            }

            switch (meshPrimitive.mode) {
                // We re-arrange the indices so that it describe a simple list of
                // triangles
            case TINYGLTF_MODE_TRIANGLE_FAN:
                if (!convertedToTriangleList) {
                    std::cout << "TRIANGLE_FAN\n";
                    // This only has to be done once per primitive
                    convertedToTriangleList = true;

                    // We steal the guts of the vector
                    auto triangleFan = std::move(loadedMesh.faces);
                    loadedMesh.faces.clear();

                    // Push back the indices that describe just one triangle one by one
                    for (size_t i{ 2 }; i < triangleFan.size(); ++i) {
                        loadedMesh.faces.push_back(triangleFan[0]);
                        loadedMesh.faces.push_back(triangleFan[i - 1]);
                        loadedMesh.faces.push_back(triangleFan[i]);
                    }
                }
            case TINYGLTF_MODE_TRIANGLE_STRIP:
                if (!convertedToTriangleList) {
                    std::cout << "TRIANGLE_STRIP\n";
                    // This only has to be done once per primitive
                    convertedToTriangleList = true;

                    auto triangleStrip = std::move(loadedMesh.faces);
                    loadedMesh.faces.clear();

                    for (size_t i{ 2 }; i < triangleStrip.size(); ++i) {
                        loadedMesh.faces.push_back(triangleStrip[i - 2]);
                        loadedMesh.faces.push_back(triangleStrip[i - 1]);
                        loadedMesh.faces.push_back(triangleStrip[i]);
                    }
                }
            case TINYGLTF_MODE_TRIANGLES:  // this is the simpliest case to handle

            {
                std::cout << "TRIANGLES\n";

                for (const auto& attribute : meshPrimitive.attributes) {
                    const auto attribAccessor = model.accessors[attribute.second];
                    const auto& bufferView =
                        model.bufferViews[attribAccessor.bufferView];
                    const auto& buffer = model.buffers[bufferView.buffer];
                    const auto dataPtr = buffer.data.data() + bufferView.byteOffset +
                        attribAccessor.byteOffset;
                    const auto byte_stride = attribAccessor.ByteStride(bufferView);
                    const auto count = attribAccessor.count;

                    std::cout << "current attribute has count " << count
                        << " and stride " << byte_stride << " bytes\n";

                    std::cout << "attribute string is : " << attribute.first << '\n';
                    if (attribute.first == "POSITION") {
                        std::cout << "found position attribute\n";

                        // get the position min/max for computing the boundingbox
                        pMin.x = attribAccessor.minValues[0];
                        pMin.y = attribAccessor.minValues[1];
                        pMin.z = attribAccessor.minValues[2];
                        pMax.x = attribAccessor.maxValues[0];
                        pMax.y = attribAccessor.maxValues[1];
                        pMax.z = attribAccessor.maxValues[2];

                        switch (attribAccessor.type) {
                        case TINYGLTF_TYPE_VEC3: {
                            switch (attribAccessor.componentType) {
                            case TINYGLTF_COMPONENT_TYPE_FLOAT:
                                std::cout << "Type is FLOAT\n";
                                // 3D vector of float
                                v3fArray positions(
                                    arrayAdapter<v3f>(dataPtr, count, byte_stride));

                                std::cout << "positions's size : " << positions.size()
                                    << '\n';

                                for (size_t i{ 0 }; i < positions.size(); ++i) {
                                    const auto v = positions[i];
                                    //std::cout << "positions[" << i << "]: (" << v.x << ", "
                                        //<< v.y << ", " << v.z << ")\n";

                                    loadedMesh.vertices.push_back(v.x * scale);
                                    loadedMesh.vertices.push_back(v.y * scale);
                                    loadedMesh.vertices.push_back(v.z * scale);
                                }
                            }
                            break;
                        case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                            std::cout << "Type is DOUBLE\n";
                            switch (attribAccessor.type) {
                            case TINYGLTF_TYPE_VEC3: {
                                v3dArray positions(
                                    arrayAdapter<v3d>(dataPtr, count, byte_stride));
                                for (size_t i{ 0 }; i < positions.size(); ++i) {
                                    const auto v = positions[i];

                                    /*std::cout << "positions[" << i << "]: (" << v.x
                                        << ", " << v.y << ", " << v.z << ")\n";*/

                                    loadedMesh.vertices.push_back(v.x * scale);
                                    loadedMesh.vertices.push_back(v.y * scale);
                                    loadedMesh.vertices.push_back(v.z * scale);
                                }
                            } break;
                            default:
                                // TODO Handle error
                                break;
                            }
                            break;
                        default:
                            break;
                        }
                        } break;
                        }
                    }

                    if (attribute.first == "NORMAL") {
                        std::cout << "found normal attribute\n";

                        switch (attribAccessor.type) {
                        case TINYGLTF_TYPE_VEC3: {
                            std::cout << "Normal is VEC3\n";
                            switch (attribAccessor.componentType) {
                            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                                std::cout << "Normal is FLOAT\n";
                                v3fArray normals(
                                    arrayAdapter<v3f>(dataPtr, count, byte_stride));

                                // IMPORTANT: We need to reorder normals (and texture
                                // coordinates into "facevarying" order) for each face

                                // For each triangle :
                                for (size_t i{ 0 }; i < indices.size() / 3; ++i) {
                                    // get the i'th triange's indexes
                                    auto f0 = indices[3 * i + 0];
                                    auto f1 = indices[3 * i + 1];
                                    auto f2 = indices[3 * i + 2];

                                    // get the 3 normal vectors for that face
                                    v3f n0, n1, n2;
                                    n0 = normals[f0];
                                    n1 = normals[f1];
                                    n2 = normals[f2];

                                    // Put them in the array in the correct order
                                    loadedMesh.facevarying_normals.push_back(n0.x);
                                    loadedMesh.facevarying_normals.push_back(n0.y);
                                    loadedMesh.facevarying_normals.push_back(n0.z);

                                    loadedMesh.facevarying_normals.push_back(n1.x);
                                    loadedMesh.facevarying_normals.push_back(n1.y);
                                    loadedMesh.facevarying_normals.push_back(n1.z);

                                    loadedMesh.facevarying_normals.push_back(n2.x);
                                    loadedMesh.facevarying_normals.push_back(n2.y);
                                    loadedMesh.facevarying_normals.push_back(n2.z);
                                }
                            } break;
                            case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                                std::cout << "Normal is DOUBLE\n";
                                v3dArray normals(
                                    arrayAdapter<v3d>(dataPtr, count, byte_stride));

                                // IMPORTANT: We need to reorder normals (and texture
                                // coordinates into "facevarying" order) for each face

                                // For each triangle :
                                for (size_t i{ 0 }; i < indices.size() / 3; ++i) {
                                    // get the i'th triange's indexes
                                    auto f0 = indices[3 * i + 0];
                                    auto f1 = indices[3 * i + 1];
                                    auto f2 = indices[3 * i + 2];

                                    // get the 3 normal vectors for that face
                                    v3d n0, n1, n2;
                                    n0 = normals[f0];
                                    n1 = normals[f1];
                                    n2 = normals[f2];

                                    // Put them in the array in the correct order
                                    loadedMesh.facevarying_normals.push_back(n0.x);
                                    loadedMesh.facevarying_normals.push_back(n0.y);
                                    loadedMesh.facevarying_normals.push_back(n0.z);

                                    loadedMesh.facevarying_normals.push_back(n1.x);
                                    loadedMesh.facevarying_normals.push_back(n1.y);
                                    loadedMesh.facevarying_normals.push_back(n1.z);

                                    loadedMesh.facevarying_normals.push_back(n2.x);
                                    loadedMesh.facevarying_normals.push_back(n2.y);
                                    loadedMesh.facevarying_normals.push_back(n2.z);
                                }
                            } break;
                            default:
                                std::cerr << "Unhandeled componant type for normal\n";
                            }
                        } break;
                        default:
                            std::cerr << "Unhandeled vector type for normal\n";
                        }

                        // Face varying comment on the normals is also true for the UVs
                        
                    }

                    if (attribute.first == "TEXCOORD_0") {
                        std::cout << "Found texture coordinates\n";

                        switch (attribAccessor.type) {
                        case TINYGLTF_TYPE_VEC2: {
                            std::cout << "TEXTCOORD is VEC2\n";
                            switch (attribAccessor.componentType) {
                            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                                std::cout << "TEXTCOORD is FLOAT\n";
                                v2fArray uvs(
                                    arrayAdapter<v2f>(dataPtr, count, byte_stride));

                                for (size_t i{ 0 }; i < indices.size() / 3; ++i) {
                                    // get the i'th triange's indexes
                                    auto f0 = indices[3 * i + 0];
                                    auto f1 = indices[3 * i + 1];
                                    auto f2 = indices[3 * i + 2];

                                    // get the texture coordinates for each triangle's
                                    // vertices
                                    v2f uv0, uv1, uv2;
                                    uv0 = uvs[f0];
                                    uv1 = uvs[f1];
                                    uv2 = uvs[f2];

                                    // push them in order into the mesh data
                                    loadedMesh.facevarying_uvs.push_back(uv0.x);
                                    loadedMesh.facevarying_uvs.push_back(uv0.y);

                                    loadedMesh.facevarying_uvs.push_back(uv1.x);
                                    loadedMesh.facevarying_uvs.push_back(uv1.y);

                                    loadedMesh.facevarying_uvs.push_back(uv2.x);
                                    loadedMesh.facevarying_uvs.push_back(uv2.y);
                                }

                            } break;
                            case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                                std::cout << "TEXTCOORD is DOUBLE\n";
                                v2dArray uvs(
                                    arrayAdapter<v2d>(dataPtr, count, byte_stride));

                                for (size_t i{ 0 }; i < indices.size() / 3; ++i) {
                                    // get the i'th triange's indexes
                                    auto f0 = indices[3 * i + 0];
                                    auto f1 = indices[3 * i + 1];
                                    auto f2 = indices[3 * i + 2];

                                    v2d uv0, uv1, uv2;
                                    uv0 = uvs[f0];
                                    uv1 = uvs[f1];
                                    uv2 = uvs[f2];

                                    loadedMesh.facevarying_uvs.push_back(uv0.x);
                                    loadedMesh.facevarying_uvs.push_back(uv0.y);

                                    loadedMesh.facevarying_uvs.push_back(uv1.x);
                                    loadedMesh.facevarying_uvs.push_back(uv1.y);

                                    loadedMesh.facevarying_uvs.push_back(uv2.x);
                                    loadedMesh.facevarying_uvs.push_back(uv2.y);
                                }
                            } break;
                            default:
                                std::cerr << "unrecognized vector type for UV";
                            }
                        } break;
                        default:
                            std::cerr << "unreconized componant type for UV";
                        }
                    }
                }
                break;

            default:
                std::cerr << "primitive mode not implemented";
                break;

                // These aren't triangles:
            case TINYGLTF_MODE_POINTS:
            case TINYGLTF_MODE_LINE:
            case TINYGLTF_MODE_LINE_LOOP:
                std::cerr << "primitive is not triangle based, ignoring";
            }
            }

            // bbox :
            v3f bCenter;
            bCenter.x = 0.5f * (pMax.x - pMin.x) + pMin.x;
            bCenter.y = 0.5f * (pMax.y - pMin.y) + pMin.y;
            bCenter.z = 0.5f * (pMax.z - pMin.z) + pMin.z;

            /*for (size_t v = 0; v < loadedMesh.vertices.size() / 3; v++) {
                loadedMesh.vertices[3 * v + 0] -= bCenter.x;
                loadedMesh.vertices[3 * v + 1] -= bCenter.y;
                loadedMesh.vertices[3 * v + 2] -= bCenter.z;
            }*/

            loadedMesh.pivot_xform[0][0] = 1.0f;
            loadedMesh.pivot_xform[0][1] = 0.0f;
            loadedMesh.pivot_xform[0][2] = 0.0f;
            loadedMesh.pivot_xform[0][3] = 0.0f;

            loadedMesh.pivot_xform[1][0] = 0.0f;
            loadedMesh.pivot_xform[1][1] = 1.0f;
            loadedMesh.pivot_xform[1][2] = 0.0f;
            loadedMesh.pivot_xform[1][3] = 0.0f;

            loadedMesh.pivot_xform[2][0] = 0.0f;
            loadedMesh.pivot_xform[2][1] = 0.0f;
            loadedMesh.pivot_xform[2][2] = 1.0f;
            loadedMesh.pivot_xform[2][3] = 0.0f;

            loadedMesh.pivot_xform[3][0] = bCenter.x;
            loadedMesh.pivot_xform[3][1] = bCenter.y;
            loadedMesh.pivot_xform[3][2] = bCenter.z;
            loadedMesh.pivot_xform[3][3] = 1.0f;

            // TODO handle materials
            for (size_t i{ 0 }; i < loadedMesh.faces.size(); ++i)
                loadedMesh.material_ids.push_back(materials.at(0).id);

            meshes.push_back(loadedMesh);
            ret = true;
        }
    }

    // Iterate through all texture declaration in glTF file
    for (const auto& gltfTexture : model.textures) {
        std::cout << "Found texture!";
        example::Texture loadedTexture;
        const auto& image = model.images[gltfTexture.source];
        loadedTexture.components = image.component;
        loadedTexture.width = image.width;
        loadedTexture.height = image.height;

        const auto size =
            image.component * image.width * image.height * sizeof(unsigned char);
        loadedTexture.image = new unsigned char[size];
        memcpy(loadedTexture.image, image.image.data(), size);
        textures.push_back(loadedTexture);
    }

    return ret;
}

int Scene::loadGLTFMesh(const std::string& file_path, const Geom& parent_geom) {
    // ref https://github.com/syoyo/tinygltf/blob/master/examples/glview/glview.cc
    std::cout << "read gltf mesh from " << file_path << std::endl;

    // ty gktf-loader
    std::vector<example::Material> gltf_materials;
    std::vector<example::Mesh<float> > gltf_meshes;
    std::vector<example::Texture> gltf_textures;
    // ref https://github.com/syoyo/tinygltf/blob/master/examples/raytrace/main.cc, 
    // ref https://github.com/taylornelms15/Project3-CUDA-Path-Tracer/blob/master/src/scene.cpp
    bool flag = false;

    example::Material default_material;

    // tigra: set default material to 95% white diffuse
    default_material.diffuse[0] = 0.95f;
    default_material.diffuse[1] = 0.95f;
    default_material.diffuse[2] = 0.95f;

    default_material.specular[0] = 0;
    default_material.specular[1] = 0;
    default_material.specular[2] = 0;

    // Material pushed as first material on the list
    gltf_materials.push_back(default_material);
    flag = this->myGLTFloader(file_path, 1.0, gltf_meshes, gltf_materials, gltf_textures);

    if (!flag) {
        std::cout << "Failed to load glTF file "
            << std::endl;
        return -1;
    }

    if (gltf_meshes.size() > 0) {
        std::cout << "there has " << gltf_meshes.size() << " meshes." << std::endl;
        for (auto cur_mesh = gltf_meshes.begin(); cur_mesh != gltf_meshes.end(); cur_mesh++) {
            GLTF_Model cur_model;
            std::vector<Triangle> cur_triangles;
            glm::vec3 maxVal_vec(-INFINITY, -INFINITY, -INFINITY);
            glm::vec3 minVal_vec(INFINITY, INFINITY, INFINITY);
            std::cout << cur_mesh->faces.size() << " faces." << std::endl;
            for (int i = 0; i < cur_mesh->faces.size(); i+=3) {
                
                Triangle cur_triangle;

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
                    cur_mesh->facevarying_normals[3 * i],
                    cur_mesh->facevarying_normals[3 * i + 1],
                    cur_mesh->facevarying_normals[3 * i + 2]
                );

                cur_triangle.n1 = glm::vec3(
                    cur_mesh->facevarying_normals[3 * i + 3],
                    cur_mesh->facevarying_normals[3 * i + 4],
                    cur_mesh->facevarying_normals[3 * i + 5]
                );

                cur_triangle.n2 = glm::vec3(
                    cur_mesh->facevarying_normals[3 * i + 6],
                    cur_mesh->facevarying_normals[3 * i + 7],
                    cur_mesh->facevarying_normals[3 * i + 8]
                );

                cur_triangle.norm = glm::triangleNormal(
                    cur_triangle.v0,
                    cur_triangle.v1,
                    cur_triangle.v2
                );

                if (cur_mesh->facevarying_uvs.size() > 0) {
                    auto uvs = cur_mesh->facevarying_uvs;
                    cur_triangle.uv0 = glm::vec2(uvs[2 * i + 0], uvs[2 * i + 1]);
                    cur_triangle.uv1 = glm::vec2(uvs[2 * i + 2], uvs[2 * i + 3]);
                    cur_triangle.uv2 = glm::vec2(uvs[2 * i + 4], uvs[2 * i + 5]);
                }
                else {
                    cur_triangle.uv0 = glm::vec2(-1.0f);
                    cur_triangle.uv1 = glm::vec2(-1.0f);
                    cur_triangle.uv2 = glm::vec2(-1.0f);
                }

                //cur_triangle.norm = glm::triangleNormal()
                cur_triangles.emplace_back(cur_triangle);
                // store geom info from .txt
                
                // assign bounding box
                //TODO check correctness for this

                glm::vec3 w_v0 = glm::vec3(parent_geom.transform * glm::vec4(cur_triangle.v0, 1.f));
                glm::vec3 w_v1 = glm::vec3(parent_geom.transform * glm::vec4(cur_triangle.v1, 1.f));
                glm::vec3 w_v2 = glm::vec3(parent_geom.transform * glm::vec4(cur_triangle.v2, 1.f));

                minVal_vec = glm::min(minVal_vec, w_v0);
                minVal_vec = glm::min(minVal_vec, w_v1);
                minVal_vec = glm::min(minVal_vec, w_v2);

                maxVal_vec = glm::max(maxVal_vec, w_v0);
                maxVal_vec = glm::max(maxVal_vec, w_v1);
                maxVal_vec = glm::max(maxVal_vec, w_v2);
            }
            cur_model.self_geom.type = GLTF_MESH;
            cur_model.self_geom = parent_geom;
            // insert cur triangeles
            cur_model.triangle_idx = this->triangles.size();
            cur_model.triangle_count = cur_triangles.size();
            //this->gltf_models.emplace_back(cur_model);
            this->triangles.insert(this -> triangles.end(), cur_triangles.begin(), cur_triangles.end());

            // create bbox
            Geom cur_bbox;
            cur_bbox = parent_geom;
            cur_bbox.type = BBOX;
            
           /* cur_bbox.scale = (maxVal_vec - minVal_vec) * parent_geom.scale;
            cur_bbox.translation = maxVal_vec / 2.0f + minVal_vec / 2.0f;
            cur_bbox.translation = cur_bbox.translation + parent_geom.translation;*/
            //cur_bbox.rotation = parent_geom.rotation;
            cur_bbox.scale = maxVal_vec - minVal_vec;
            cur_bbox.translation = maxVal_vec / 2.0f + minVal_vec / 2.0f;
            cur_bbox.rotation = glm::vec3(0);

            cur_bbox.transform = utilityCore::buildTransformationMatrix(
                cur_bbox.translation,
                cur_bbox.rotation,
                cur_bbox.scale);

            cur_bbox.inverseTransform = glm::inverse(cur_bbox.transform);
            cur_bbox.invTranspose = glm::inverseTranspose(cur_bbox.transform);
            // use this to index gltf_models
            cur_bbox.mesh_idx = this->gltf_models.size();
            this->geoms.emplace_back(cur_bbox);

            this->gltf_models.emplace_back(cur_model);
        }
    }
    return 0;
}

TextureDescriptor Scene::loadTexture(const std::string& path, bool normalize)
{
    TextureDescriptor desc;

    Texture* tex = nullptr;
    if (textureMap.find(path) == textureMap.end()) {
        tex = new Texture(path, 1.f, normalize);
        this->textures.push_back(tex);
    }
    else {
        tex = textureMap[path];
    }

    desc.type = 0;
    desc.index = textures.size() - 1;
    desc.width = tex->xSize;
    desc.height = tex->ySize;
    desc.valid = 1;

    return desc;
}

int Scene::loadCamera() {
    std::cout << "Loading Camera ..." << endl;
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

    std::cout << "Loaded camera!" << std::endl;
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

        string line;
        utilityCore::safeGetline(fp_in, line);
        //load static properties
        while (!line.empty() && fp_in.good()) {
            /*string line;
            utilityCore::safeGetline(fp_in, line);*/
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
            else if (strcmp(tokens[0].c_str(), "TEX_DIFFUSE") == 0)
            {
                newMaterial.diffuseTexture = loadTexture(tokens[1], false);
            }
            else if (strcmp(tokens[0].c_str(), "TEX_SPECULAR") == 0)
            {
                newMaterial.specularTexture = loadTexture(tokens[1], false);
            }
            else if (strcmp(tokens[0].c_str(), "TEX_NORMAL") == 0)
            {
                newMaterial.normalTexture = loadTexture(tokens[1], false);
            }

            utilityCore::safeGetline(fp_in, line);
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
