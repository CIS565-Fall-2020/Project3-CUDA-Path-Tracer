#include "scene.h"

#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobj.h"

using namespace std;

Scene::Scene(string filename) {
  std::cout << "Reading scene from " << filename << " ..." << endl;
  std::cout << " " << endl;
  char *fname = (char *)filename.c_str();
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
        if (loadGeom(tokens[1]) < 0) {
          std::cout << "Load Object " << tokens[1] << " failed" << endl;
          throw;
        }
        std::cout << " " << endl;
      } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
        loadCamera();
        std::cout << " " << endl;
      } else if (strcmp(tokens[0].c_str, "MESH") == 0) {
        if (loadMesh(tokens[1]) < 0) {
          std::cout << "Load Object " << tokens[1] << " failed" << endl;
          throw;
        }
        std::cout << " " << endl;
      }
    }
  }
}

int Scene::loadGeom(std::string objectid) {
  int id = atoi(objectid.c_str());
  if (id != geoms.size()) {
    cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
    return -1;
  } else {
    cout << "Loading Geom " << id << "..." << endl;
    Geom newGeom;
    string line;

    // load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
      vector<string> tokens = utilityCore::tokenizeString(line);
      if (strcmp(line.c_str(), "sphere") == 0) {
        cout << "Creating new sphere..." << endl;
        newGeom.type = SPHERE;
      } else if (strcmp(line.c_str(), "cube") == 0) {
        cout << "Creating new cube..." << endl;
        newGeom.type = CUBE;
      } else if (strcmp(tokens[0].c_str(), "mesh") == 0) {
        cout << "Creating new mesh..." << endl;
        int meshIdx = atoi(tokens[1].c_str());
        if (meshIdx >= meshes.size) {
          cout << "Mesh idx " << meshIdx << " out of bounds" << endl;
          return -1;
        }
        newGeom.type = MESH;
        utilityCore::safeGetline(fp_in, line);
        vector<string> meshTokens = utilityCore::tokenizeString(line);
        if (strcmp(meshTokens[0].c_str(), "shapeIdx") == 0) {
          int shapeIdx = atoi(meshTokens[1].c_str());
          if (shapeIdx >= meshes[meshIdx].shapes.size()) {
            cout << "Shape idx " << shapeIdx << " out of bounds" << endl;
            return -1;
          }
          int triStart = 0;
          for (int i = 0; i < meshIdx; i++) {
            triStart += meshes[i].triangleCnt;
          }
          for (int i = 0; i < shapeIdx; i++) {
            triStart += meshes[meshIdx].shapes[i].mesh.num_face_vertices.size();
          }
          newGeom.triStart = triStart;
          newGeom.triEnd = triStart + meshes[meshIdx].shapes[shapeIdx].mesh.num_face_vertices.size();
        }
      }
    }

    // link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
      vector<string> tokens = utilityCore::tokenizeString(line);
      newGeom.materialid = atoi(tokens[1].c_str());
      cout << "Connecting Geom " << objectid << " to Material "
           << newGeom.materialid << "..." << endl;
    }

    // load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
      vector<string> tokens = utilityCore::tokenizeString(line);

      // load tranformations
      if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
        newGeom.translation =
            glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                      atof(tokens[3].c_str()));
      } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
        newGeom.rotation =
            glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                      atof(tokens[3].c_str()));
      } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
        newGeom.scale =
            glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                      atof(tokens[3].c_str()));
        if (newGeom.type == MESH) {
          newGeom.min_bound = triangles[newGeom.triStart].v[0];
          newGeom.max_bound = triangles[newGeom.triStart].v[0];
          for (int i = newGeom.triStart; i < newGeom.triEnd; i++) {
            Triangle &tri = triangles[i];
            for (int i = 0; i < 3; i++) {
              tri.v[i].x *= newGeom.scale[0];
              tri.v[i].y *= newGeom.scale[1];
              tri.v[i].z *= newGeom.scale[2];
              newGeom.min_bound = glm::min(newGeom.min_bound, tri.v[i]);
              newGeom.max_bound = glm::max(newGeom.max_bound, tri.v[i]);
            }
          }
          // Reset geometry scale to 1.0
          newGeom.scale = glm::vec3(1.0f);
        }
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
  return 0;
}

// Return the number of triangles of the mesh, or -1 if failed
int Scene::loadMesh(string meshId) {
  int id = atoi(meshId.c_str());
  if (id != meshes.size()) {
    cout << "ERROR: MESH ID does not match expected number of geoms" << endl;
    return -1;
  }
  std::string warn, err;
  Mesh mesh;
  std::string filename;
  utilityCore::safeGetline(fp_in, filename);

  // load obj
  if (!tinyobj::LoadObj(&mesh.attrib, &mesh.shapes, &mesh.materials, &warn, &err,
                        filename.c_str())) {
    std::cout << "tinyobj load " << filename << " failed " << std::endl;
    return -1;
  }

  if (!warn.empty()) {
    std::cout << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  int startIdx = triangles.size();

  // Loop over shapes
  for (const tinyobj::shape_t &shape : shapes) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
      int fv = shape.mesh.num_face_vertices[f];
      // Loop over vertices in the face.
      Triangle t;

      for (size_t v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        t.v[v] = glm::vec3(vx, vy, vz);
      }

      index_offset += fv;
      // compute normal
      t.n = glm::normalize(glm::cross(t.v[1] - t.v[0], t.v[2] - t.v[0]));
      triangles.push_back(t);
    }
  }
  mesh.triangleCnt = triangles.size() - startIdx;
  meshes.push_back(mesh);
  return mesh.triangleCnt;
}

int Scene::loadCamera() {
  cout << "Loading Camera ..." << endl;
  RenderState &state = this->state;
  Camera &camera = state.camera;
  float fovy;

  // load static properties
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
      camera.position =
          glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                    atof(tokens[3].c_str()));
    } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
      camera.lookAt =
          glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                    atof(tokens[3].c_str()));
    } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
      camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                            atof(tokens[3].c_str()));
    } else if (strcmp(tokens[0].c_str(), "MOTION") == 0) {
      camera.motion =
          glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                    atof(tokens[3].c_str()));
    } else if ((tokens[0].c_str(), "DOF") == 0) {
      camera.depth_of_field = atoi(tokens[1].c_str());
    } else if (strcmp(tokens[0].c_str(), "LENSR") == 0) {
      camera.lens_radius = atof(tokens[1].c_str());
    } else if (strcmp(tokens[0].c_str(), "FD") == 0) {
      camera.focal_distance = atof(tokens[1].c_str());
    }

    utilityCore::safeGetline(fp_in, line);
  }

  // calculate fov based on resolution
  float yscaled = tan(fovy * (PI / 180));
  float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(xscaled) * 180) / PI;
  camera.fov = glm::vec2(fovx, fovy);

  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                 2 * yscaled / (float)camera.resolution.y);

  camera.view = glm::normalize(camera.lookAt - camera.position);

  // set up render camera stuff
  int arraylen = camera.resolution.x * camera.resolution.y;
  state.image.resize(arraylen);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());

  cout << "Loaded camera!" << endl;
  return 1;
}

int Scene::loadMaterial(string materialid) {
  int id = atoi(materialid.c_str());
  if (id != materials.size()) {
    cout << "ERROR: MATERIAL ID does not match expected number of materials"
         << endl;
    return -1;
  } else {
    cout << "Loading Material " << id << "..." << endl;
    Material newMaterial;

    // load static properties
    for (int i = 0; i < 7; i++) {
      string line;
      utilityCore::safeGetline(fp_in, line);
      vector<string> tokens = utilityCore::tokenizeString(line);
      if (strcmp(tokens[0].c_str(), "RGB") == 0) {
        glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                        atof(tokens[3].c_str()));
        newMaterial.color = color;
      } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
        newMaterial.specular.exponent = atof(tokens[1].c_str());
      } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
        glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                            atof(tokens[3].c_str()));
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
