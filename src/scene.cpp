#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"

#define FOCAL_DIST 5.f
#define LENS_RADIUS 1.5f

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

Scene::Scene(string filename, string obj_filename)
{
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		throw;
	}

	mesh.filename = obj_filename;
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

//bool Scene::loadMesh() {
//    tinyobj::attrib_t attrib;
//    std::vector<tinyobj::shape_t> shapes;
//    std::vector<tinyobj::material_t> materials;
//    std::string warn, err;
//
//    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, mesh.filename.c_str());
//
//    if (!warn.empty()) {
//        std::cout << warn << std::endl;
//    }
//
//    if (!err.empty()) {
//        std::cerr << err << std::endl;
//        return false; 
//    }
//
//    if (!ret) {
//        return false;
//        //exit(1);
//    }
//
//    //Loop over shapes
//    for (size_t s = 0; s < shapes.size(); s++) {
//        // Loop over faces(polygon)
//        size_t index_offset = 0;
//        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
//            int fv = shapes[s].mesh.num_face_vertices[f];
//
//            // Loop over vertices in the face.
//            if (fv == 3)
//            {
//                Triangle t;
//
//                // INDICES 
//                tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset];
//                tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset + 1];
//                tinyobj::index_t idx3 = shapes[s].mesh.indices[index_offset + 2];
//
//                //VERTICES 
//                //V1 
//                tinyobj::real_t v1x = attrib.vertices[3 * idx1.vertex_index];
//                tinyobj::real_t v1y = attrib.vertices[3 * idx1.vertex_index + 1];
//                tinyobj::real_t v1z = attrib.vertices[3 * idx1.vertex_index + 2];
//                glm::vec3 vert0 = glm::vec3(float(v1x), float(v1y), float(v1z));
//                t.vertices[0] = vert0;
//                //V2 
//                tinyobj::real_t v2x = attrib.vertices[3 * idx2.vertex_index];
//                tinyobj::real_t v2y = attrib.vertices[3 * idx2.vertex_index + 1];
//                tinyobj::real_t v2z = attrib.vertices[3 * idx2.vertex_index + 2];
//                glm::vec3 vert1 = glm::vec3(float(v2x), float(v2y), float(v2z));
//                t.vertices[1] = vert1;
//                //V3 
//                tinyobj::real_t v3x = attrib.vertices[3 * idx3.vertex_index];
//                tinyobj::real_t v3y = attrib.vertices[3 * idx3.vertex_index + 1];
//                tinyobj::real_t v3z = attrib.vertices[3 * idx3.vertex_index + 2];
//                glm::vec3 vert2 = glm::vec3(float(v3x), float(v3y), float(v3z));
//                t.vertices[2] = vert2;
//
//                //NORMALS 
//                if (attrib.normals.size() > 0)
//                {
//                    //N1
//                    tinyobj::real_t n1x = attrib.normals[3 * idx1.normal_index + 0];
//                    tinyobj::real_t n1y = attrib.normals[3 * idx1.normal_index + 1];
//                    tinyobj::real_t n1z = attrib.normals[3 * idx1.normal_index + 2];
//                    glm::vec3 norm1 = glm::vec3(float(n1x), float(n1y), float(n1z));
//                    t.normals[0] = norm1;
//                    //N2
//                    tinyobj::real_t n2x = attrib.normals[3 * idx2.normal_index + 0];
//                    tinyobj::real_t n2y = attrib.normals[3 * idx2.normal_index + 1];
//                    tinyobj::real_t n2z = attrib.normals[3 * idx2.normal_index + 2];
//                    glm::vec3 norm2 = glm::vec3(float(n2x), float(n2y), float(n2z));
//                    t.normals[1] = norm2;
//                    //N3 
//                    tinyobj::real_t n3x = attrib.normals[3 * idx3.normal_index + 0];
//                    tinyobj::real_t n3y = attrib.normals[3 * idx3.normal_index + 1];
//                    tinyobj::real_t n3z = attrib.normals[3 * idx3.normal_index + 2];
//                    glm::vec3 norm3 = glm::vec3(float(n3x), float(n3y), float(n3z));
//                    t.normals[2] = norm3;
//                }
//
//                if (attrib.texcoords.size() > 0)
//                {
//                    //UVS1
//                    tinyobj::real_t t1u = attrib.texcoords[2 * idx1.texcoord_index + 0];
//                    tinyobj::real_t t1v = attrib.texcoords[2 * idx1.texcoord_index + 1];
//                    glm::vec2 uv1 = glm::vec2(float(t1u), float(t1v));
//                    t.uvs[0] = uv1;
//                    //UVS2
//                    tinyobj::real_t t2u = attrib.texcoords[2 * idx2.texcoord_index + 0];
//                    tinyobj::real_t t2v = attrib.texcoords[2 * idx2.texcoord_index + 1];
//                    glm::vec2 uv2 = glm::vec2(float(t2u), float(t2v));
//                    t.uvs[1] = uv2;
//                    //UVS3 
//                    tinyobj::real_t t3u = attrib.texcoords[2 * idx3.texcoord_index + 0];
//                    tinyobj::real_t t3v = attrib.texcoords[2 * idx3.texcoord_index + 1];
//                    glm::vec2 uv3 = glm::vec2(float(t3u), float(t3v));
//                    t.uvs[2] = uv3;
//                }
//
//                mesh.num_triangles++;
//                mesh.triangles.push_back(t);
//                index_offset += 3;
//            }
//        }
//    }
//    std::cout << "Num triangles in this mesh" << mesh.num_triangles << std::endl; 
//    return true; 
//}

bool Scene::loadMesh() {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, mesh.filename.c_str());

	if (!warn.empty()) {
		std::cout << warn << std::endl;
	}

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return false;
	}

	if (!ret) {
		return false;
		//exit(1);
	}
	mesh.minCorner = glm::vec3(FLT_MAX);
	//mesh.maxCorner = glm::vec3(0.f);
	mesh.maxCorner = glm::vec3(-FLT_MAX); 
	//Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];

			// Loop over vertices in the face.
			if (fv == 3)
			{
				Triangle t;

				// INDICES 
				tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset];
				tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset + 1];
				tinyobj::index_t idx3 = shapes[s].mesh.indices[index_offset + 2];

				//VERTICES 
				//V1 
				t.vert1 = glm::vec3(float(attrib.vertices[3 * (idx1.vertex_index)]),
					float(attrib.vertices[(3 * (idx1.vertex_index)) + 1]),
					float(attrib.vertices[(3 * (idx1.vertex_index)) + 2]));

				//V2 
				t.vert2 = glm::vec3(float(attrib.vertices[3 * (idx2.vertex_index)]),
					float(attrib.vertices[(3 * (idx2.vertex_index)) + 1]),
					float(attrib.vertices[(3 * (idx2.vertex_index)) + 2]));

				//V3 
				t.vert3 = glm::vec3(float(attrib.vertices[3 * (idx3.vertex_index)]),
					float(attrib.vertices[(3 * (idx3.vertex_index)) + 1]),
					float(attrib.vertices[(3 * (idx3.vertex_index)) + 2]));

				//NORMALS 
				if (attrib.normals.size() > 0)
				{
					//N1
					tinyobj::real_t n1x = attrib.normals[3 * idx1.normal_index + 0];
					tinyobj::real_t n1y = attrib.normals[3 * idx1.normal_index + 1];
					tinyobj::real_t n1z = attrib.normals[3 * idx1.normal_index + 2];
					glm::vec3 norm1 = glm::vec3(float(n1x), float(n1y), float(n1z));
					t.norm1 = norm1;
					//N2
					tinyobj::real_t n2x = attrib.normals[3 * idx2.normal_index + 0];
					tinyobj::real_t n2y = attrib.normals[3 * idx2.normal_index + 1];
					tinyobj::real_t n2z = attrib.normals[3 * idx2.normal_index + 2];
					glm::vec3 norm2 = glm::vec3(float(n2x), float(n2y), float(n2z));
					t.norm2 = norm2;
					//N3 
					tinyobj::real_t n3x = attrib.normals[3 * idx3.normal_index + 0];
					tinyobj::real_t n3y = attrib.normals[3 * idx3.normal_index + 1];
					tinyobj::real_t n3z = attrib.normals[3 * idx3.normal_index + 2];
					glm::vec3 norm3 = glm::vec3(float(n3x), float(n3y), float(n3z));
					t.norm3 = norm3;
				}

				//Find the min corner 
				mesh.minCorner.x = glm::min(t.vert1.x, glm::min(t.vert2.x, glm::min(t.vert3.x, mesh.minCorner.x))); 
				mesh.minCorner.y = glm::min(t.vert1.y, glm::min(t.vert2.y, glm::min(t.vert3.y, mesh.minCorner.y)));
				mesh.minCorner.z = glm::min(t.vert1.z, glm::min(t.vert2.z, glm::min(t.vert3.z, mesh.minCorner.z)));

				//Find the max corner 
				mesh.maxCorner.x = glm::max(t.vert1.x, glm::max(t.vert2.x, glm::max(t.vert3.x, mesh.maxCorner.x)));
				mesh.maxCorner.y = glm::max(t.vert1.y, glm::max(t.vert2.y, glm::max(t.vert3.y, mesh.maxCorner.y)));
				mesh.maxCorner.z = glm::max(t.vert1.z, glm::max(t.vert2.z, glm::max(t.vert3.z, mesh.maxCorner.z)));

				mesh.num_triangles++;
				mesh.triangles.push_back(t);
				index_offset += 3;
			}
		}
	}
	std::cout << "Num triangles in this mesh" << mesh.num_triangles << std::endl;
	return true;
}

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
			else if (strcmp(line.c_str(), "mesh") == 0) {
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
				try
				{
					bool success = loadMesh();
					newGeom.geomMinCorner = mesh.minCorner; 
					newGeom.geomMaxCorner = mesh.maxCorner; 
					if (!success)
					{
						throw(1);
					}
				}
				catch (int err)
				{
					std::cout << "Failed to load mesh.." << std::endl;
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

		//Store lights in the scene 
		if (materials[newGeom.materialid].emittance > 0.f)
		{
			lights.push_back(newGeom); 
		}

		geoms.push_back(newGeom);
		return 1;
	}
}

int Scene::loadCamera() {
	cout << "Loading Camera ..." << endl;
	RenderState& state = this->state;
	Camera& camera = state.camera;
	float fovy;
	camera.focalDistance = FOCAL_DIST;
	camera.lensRadius = LENS_RADIUS; 

	//load static properties
	for (int i = 0; i < 5; i++) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "RES") == 0) {
			camera.resolution.x = atoi(tokens[1].c_str());
			camera.resolution.y = atoi(tokens[2].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
			fovy = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
			state.iterations = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
			state.traceDepth = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
			state.imageName = tokens[1];
		}
		//else if (strcmp(tokens[0].c_str(), "FOCAL_DIST") == 0) {
		//	camera.focalDistance = atoi(tokens[1].c_str());
		//}
		//else if (strcmp(tokens[0].c_str(), "LENS_RADIUS") == 0) {
		//	camera.lensRadius = atoi(tokens[1].c_str());
		//}
	}

	string line;
	utilityCore::safeGetline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "EYE") == 0) {
			camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
			camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "UP") == 0) {
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

