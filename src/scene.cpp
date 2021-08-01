#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <map>

#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION

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
	// set up bounding box
	pMin = glm::vec3(FLT_MAX);
	pMax = glm::vec3(FLT_MIN);
}

void Scene::traverseNode(const tinygltf::Model &model, const tinygltf::Node &node, glm::mat4 pTran) {
#if 1
	glm::mat4 tran = glm::mat4(1.0f);
	glm::mat4 rot = glm::mat4(1.0f);
	glm::mat4 scale = glm::mat4(1.0f);
	// Translation
	if (node.translation.size() > 0) {
		tran = glm::translate(glm::mat4(), glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
	}

	// Rotation
	if (node.rotation.size() > 0) {
		glm::quat q = glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]);
		rot = glm::toMat4(q);
	}

	// Scaling
	if (node.scale.size() > 0) {
		scale = glm::scale(glm::mat4(), glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
	}

	glm::mat4 gTran = pTran * tran * rot * scale;
	// Check if matrix specified
	if (node.matrix.size() > 0) {
		float arr[16];
		for (int i = 0; i < 16; i++) {
			arr[i] = (float)node.matrix[i];
		}
		glm::mat4 local = glm::make_mat4(arr);
		gTran = pTran * local;
	}

	glm::mat4 gTranNorm = glm::inverseTranspose(gTran);

	if (node.mesh >= 0) {
		const tinygltf::Mesh &mesh = model.meshes[node.mesh];

		int existingMats = materials.size();
		// For each primitive
		for (const auto &prim : mesh.primitives) {
			// check for indices
			vector<int> indices;
			if (prim.indices >= 0) {
				const auto &indicesAccessor = model.accessors[prim.indices];
				const auto &bufferView = model.bufferViews[indicesAccessor.bufferView];
				const auto &buffer = model.buffers[bufferView.buffer];
				const unsigned char* dataAddress = &buffer.data[bufferView.byteOffset + indicesAccessor.byteOffset];
				const auto byteStride = indicesAccessor.ByteStride(bufferView);
				const auto count = indicesAccessor.count;
				for (int i = 0; i < count; i++) {
					const unsigned short *idx = reinterpret_cast<const unsigned short*>(dataAddress + i * byteStride);
					indices.push_back(idx[0]);
				}
			}
			
			// fetch positions and normals
			vector<glm::vec3> vertex_positions;
			vector<glm::vec3> vertex_normals;

			for (const auto &attr : prim.attributes) {
				const auto attrAccessor = model.accessors[attr.second];
				const auto &bufferView =
					model.bufferViews[attrAccessor.bufferView];
				const auto &buffer = model.buffers[bufferView.buffer];
				const auto dataPtr = buffer.data.data() + bufferView.byteOffset +
					attrAccessor.byteOffset;
				const auto byte_stride = attrAccessor.ByteStride(bufferView);
				const auto count = attrAccessor.count;
				const unsigned char* data_ptr = &buffer.data[bufferView.byteOffset + attrAccessor.byteOffset];

				if (attr.first == "POSITION") {
					for (int i = 0; i < count; i++) {
						const float* p = reinterpret_cast<const float*>(data_ptr + i * byte_stride);
						glm::vec4 vpos = glm::vec4(p[0], p[1], p[2], 1.0f);
						vpos = gTran * vpos;
						vertex_positions.push_back(glm::vec3(vpos));
					}
					// compute and update bounding box
					glm::vec3 p_min = glm::vec3(attrAccessor.minValues[0], attrAccessor.minValues[1], attrAccessor.minValues[2]);
					glm::vec3 p_max = glm::vec3(attrAccessor.maxValues[0], attrAccessor.maxValues[1], attrAccessor.maxValues[2]);
					updateBoundingBox(p_min, p_max, gTran);
				}
				else if (attr.first == "NORMAL") {
					for (int i = 0; i < count; i++) {
						const float* n = reinterpret_cast<const float*>(data_ptr + i * byte_stride);
						glm::vec4 norm = glm::vec4(n[0], n[1], n[2], 1.0f);
						norm = gTranNorm * norm;
						vertex_normals.push_back(glm::vec3(norm));
					}
				}
			}
			if (prim.indices < 0) {
				for (int i = 0; i < vertex_positions.size(); i++) {
					indices.push_back(i);
				}
			}
			for (int k = 0; k < indices.size(); k = k + 3) {
				Geom newGeom;
				newGeom.type = TRIANGLE;
				newGeom.v0 = vertex_positions[indices[k]];
				newGeom.v1 = vertex_positions[indices[k + 1]];
				newGeom.v2 = vertex_positions[indices[k + 2]];
				newGeom.normal = vertex_normals[indices[k]];
				newGeom.materialid = prim.material + existingMats;
				geoms.push_back(newGeom);
			}
		}
	}
	
	// Traverse children
	for (const auto &c : node.children) {
		const auto &child = model.nodes[c];
		traverseNode(model, child, gTran);
	}
#endif
}

void Scene::updateBoundingBox(const glm::vec3 &v0, const glm::vec3 &v1, const glm::mat4 &tMat) {
	glm::mat2x3 vals = glm::mat2x3(v0, v1);
	for (int i : {0, 1}) {
		for (int j : {0, 1}) {
			for (int k : {0, 1}) {
				glm::vec4 v = glm::vec4(vals[i][0], vals[j][1], vals[k][2], 1.f);
				glm::vec3 vt = glm::vec3(tMat * v);
				pMin = glm::min(pMin, vt);
				pMax = glm::max(pMax, vt);
			}
		}
	}
}

int Scene::loadGltf(string filename) {
	
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	string err;
	string warn;
	
	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
	if (!ret) {
		cout << "Enable to load gltf file. " << endl;
		return -1;
	}
	
	// Iterate through all nodes (load triangles)
	for (const auto &sc : model.scenes) {
		for (const auto &nodeIdx : sc.nodes) {
			traverseNode(model, model.nodes[nodeIdx], glm::mat4(1.f));
		}
	}
	// Load Materials (after traversing nodes)
	for (const auto &gltfMat : model.materials) {
		Material mat;
		tinygltf::PbrMetallicRoughness pbr = gltfMat.pbrMetallicRoughness;
		mat.color = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
		mat.hasReflective = pbr.metallicFactor;
		mat.specular.color = mat.color;
		materials.push_back(mat);
	}

	// Load Lights (after other materials)
	for (const auto &light : model.lights) {
		Material eMat;
		eMat.color = glm::vec3(light.color[0], light.color[1], light.color[2]);
		eMat.emittance = (float)light.intensity;
		materials.push_back(eMat);
	}

	return 0;
}

void Scene::buildOctreeNode(OctreeNode &node, int depth) {
	if (depth >= MAX_DEPTH || node.geom_idx_end == node.geom_idx_start) {
		return;
	}
	int startNodeIdx = octree.size();
	
	for (int i = 0; i < 8; i++) {
		node.childrenIndices[i] = startNodeIdx + i;
	}
	glm::vec3 c = node.center;
	glm::vec3 v0 = node.bp0;
	float dx = c[0] - v0[0];
	float dy = c[1] - v0[1];
	float dz = c[2] - v0[2];
	int l = node.geom_idx_start;
	int r = node.geom_idx_end;

	// Add 8 child nodes
	for (int i : {0, 1}) {
		for (int j : {0, 1}) {
			for (int k : {0, 1}) {
				glm::vec3 p_min = v0 + glm::vec3(i*dx, j*dy, k*dz);
				glm::vec3 p_max = p_min + glm::vec3(dx, dy, dz);
				OctreeNode c(p_min, p_max);
				// add intersecting mesh
				int start = geom_indices.size();
				for (int x = l; x < r; x++) {
					int gIdx = geom_indices[x];
					if (c.intersectTriangle(geoms[gIdx])) {
						//c.geomIndices.push_back(gIdx);
						geom_indices.push_back(gIdx);
					}
				}
				int end = geom_indices.size();
				if (start < end) {
					c.geom_idx_start = start;
					c.geom_idx_end = end;
				}
				octree.push_back(c);
			}
		}
	}

	// iterate
	for (int i = 0; i < 8; i++) {
		buildOctreeNode(octree[startNodeIdx + i], depth + 1);
	}
}

void Scene::buildOctree() {
	OctreeNode root(pMin, pMax);
	for (int i = 0; i < geoms.size(); i++) {
		if (geoms[i].type == TRIANGLE) {
			// only handle triangle mesh for now
			geom_indices.push_back(i);
		}
	}
	root.geom_idx_start = 0;
	root.geom_idx_end = geom_indices.size();
	octree.push_back(root);
	buildOctreeNode(octree[0], 1);
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
