#include "main.h"
#include "preview.h"
#include <cstring>
#include <memory>


// --------------------------------------------
// -------------gltf loading-------------------
// --------------------------------------------
//#define TINYGLTF_IMPLEMENTATION
//#include "tiny_gltf.h"
//#include <glm/gtc/matrix_inverse.hpp>
//#include "gltf-loader.h"
//#include "mesh.h"
//#include "material.h"
//using namespace example;
// -------------------------------------------------------------------------
// ----------------------- END GLTF LOADING --------------------------------
// -------------------------------------------------------------------------





static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
RenderState* renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	  startTimeString = currentTimeString();

	  if (argc < 2) {
			printf("Usage: %s SCENEFILE.txt\n", argv[0]);
			return 1;
	  }

	  const char* sceneFile = argv[1];

	  // Load scene file
	  scene = new Scene(sceneFile);

#define GLTF 0
#if GLTF
	  //  ---------------------- test load gltf ------------------------------
	  const char* gltfFile = argv[2];

	  

	  tinygltf::Model model;
	  tinygltf::TinyGLTF loader;
	  std::string err;
	  std::string warn;

	  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfFile);

	  if (!warn.empty()) {
			printf("Warn: %s\n", warn.c_str());
	  }
	  if (!err.empty()) {
			printf("Err: %s\n", err.c_str());
	  }
	  if (!ret) {
			printf("Failed to parse glTF\n");
			return -1;
	  }
	  std::cout << "no warnings: load gltf successful" << std::endl;

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

	  int currTrigIdx = 0;

	  // Iterate through all the meshes in the glTF file
	  for (const auto& gltfMesh : model.meshes) {
			std::cout << "Current mesh has " << gltfMesh.primitives.size() << " primitives\n";

			Geom newGeom;
			newGeom.type = MESH;
			newGeom.materialid = 6;
			newGeom.translation = glm::vec3(0.f, 4.f, 3.f); // right, up, forward(to camera)
			newGeom.rotation = glm::vec3(0.f, 180.f, 0.f); // ?, y?, left/right
			newGeom.scale = glm::vec3(0.2f, 0.2f, 0.2f);
			newGeom.transform = utilityCore::buildTransformationMatrix(
						newGeom.translation, newGeom.rotation, newGeom.scale);
			newGeom.inverseTransform = glm::inverse(newGeom.transform);
			newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
			newGeom.trigStartIdx = currTrigIdx;
			// later modify the scene txt to code translation etc.

			// For each primitive -- should only have triangles
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

				  std::cout << "TRIANGLES\n";
				  std::cout << "mode = " << meshPrimitive.mode << std::endl;

				  std::vector<glm::vec3> trigPositions; // same as num of indices
				  std::vector<std::vector<int>> trigVertIdx; // same as num of triangles
				  std::vector<std::vector<glm::vec3>> trigNormals; // same as num of triangles
				  std::vector<std::vector<glm::vec2>> trigUVs; // same as num of triangles

				  // set up trig vertex indexes
				  for (size_t i{ 0 }; i < indices.size() / 3; ++i) {
						// get the i'th triange's indexes
						int f0 = indices[3 * i + 0];
						int f1 = indices[3 * i + 1];
						int f2 = indices[3 * i + 2];
						std::vector<int> indices;
						indices.push_back(f0);
						indices.push_back(f1);
						indices.push_back(f2);
						trigVertIdx.push_back(indices);
				  }

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
							  /*pMin.x = attribAccessor.minValues[0];
							  pMin.y = attribAccessor.minValues[1];
							  pMin.z = attribAccessor.minValues[2];
							  pMax.x = attribAccessor.maxValues[0];
							  pMax.y = attribAccessor.maxValues[1];
							  pMax.z = attribAccessor.maxValues[2];*/

							  switch (attribAccessor.type) {
							  case TINYGLTF_TYPE_VEC3: {
									switch (attribAccessor.componentType) {
									case TINYGLTF_COMPONENT_TYPE_FLOAT:
										  std::cout << "Type is FLOAT\n";
										  // 3D vector of float
										  v3fArray positions(
												arrayAdapter<v3f>(dataPtr, count, byte_stride));

										  /*std::cout << "positions's size : " << positions.size()
												<< '\n';*/

										  for (size_t i{ 0 }; i < positions.size(); ++i) {
												const auto v = positions[i];
												/*std::cout << "positions[" << i << "]: (" << v.x << ", "
													  << v.y << ", " << v.z << ")\n";*/

												trigPositions.push_back(glm::vec3(v.x, v.y, v.z));
												// might need to times the v.x, v.y v.z by a scale? -- haven't set scale yet
										  }

										  //std::cout << "trigPositions.size = " << trigPositions.size() << std::endl;
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

												trigPositions.push_back(glm::vec3(v.x, v.y, v.z));
												// might need to multiply by scale later

												/*loadedMesh.vertices.push_back(v.x * scale);
												loadedMesh.vertices.push_back(v.y * scale);
												loadedMesh.vertices.push_back(v.z * scale);*/
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

												/*std::cout << "indices.size = " << indices.size() << std::endl;

												std::cout << "for this triangle, i = " << i << ", f0 = "
													  << f0 << ", f1 = " << f1 << ", f2 = " << f2
													  << std::endl;*/

												std::vector<glm::vec3> currTriNormals;
												currTriNormals.push_back(glm::vec3(n0.x, n0.y, n0.z));
												currTriNormals.push_back(glm::vec3(n1.x, n1.y, n1.z));
												currTriNormals.push_back(glm::vec3(n2.x, n2.y, n2.z));
												trigNormals.push_back(currTriNormals);

												// Put them in the array in the correct order
												/*loadedMesh.facevarying_normals.push_back(n0.x);
												loadedMesh.facevarying_normals.push_back(n0.y);
												loadedMesh.facevarying_normals.push_back(n0.z);

												loadedMesh.facevarying_normals.push_back(n1.x);
												loadedMesh.facevarying_normals.push_back(n1.y);
												loadedMesh.facevarying_normals.push_back(n1.z);

												loadedMesh.facevarying_normals.push_back(n2.x);
												loadedMesh.facevarying_normals.push_back(n2.y);
												loadedMesh.facevarying_normals.push_back(n2.z);*/
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

												std::vector<glm::vec3> currTriNormals;
												currTriNormals.push_back(glm::vec3(n0.x, n0.y, n0.z));
												currTriNormals.push_back(glm::vec3(n1.x, n1.y, n1.z));
												currTriNormals.push_back(glm::vec3(n2.x, n2.y, n2.z));
												trigNormals.push_back(currTriNormals);

												// Put them in the array in the correct order
												/*loadedMesh.facevarying_normals.push_back(n0.x);
												loadedMesh.facevarying_normals.push_back(n0.y);
												loadedMesh.facevarying_normals.push_back(n0.z);

												loadedMesh.facevarying_normals.push_back(n1.x);
												loadedMesh.facevarying_normals.push_back(n1.y);
												loadedMesh.facevarying_normals.push_back(n1.z);

												loadedMesh.facevarying_normals.push_back(n2.x);
												loadedMesh.facevarying_normals.push_back(n2.y);
												loadedMesh.facevarying_normals.push_back(n2.z);*/
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

													  std::vector<glm::vec2> currTriUVs;
													  currTriUVs.push_back(glm::vec2(uv0.x, uv0.y));
													  currTriUVs.push_back(glm::vec2(uv1.x, uv1.y));
													  currTriUVs.push_back(glm::vec2(uv2.x, uv2.y));
													  trigUVs.push_back(currTriUVs);

													  // push them in order into the mesh data
													  /*loadedMesh.facevarying_uvs.push_back(uv0.x);
													  loadedMesh.facevarying_uvs.push_back(uv0.y);

													  loadedMesh.facevarying_uvs.push_back(uv1.x);
													  loadedMesh.facevarying_uvs.push_back(uv1.y);

													  loadedMesh.facevarying_uvs.push_back(uv2.x);
													  loadedMesh.facevarying_uvs.push_back(uv2.y);*/
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

													  std::vector<glm::vec2> currTriUVs;
													  currTriUVs.push_back(glm::vec2(uv0.x, uv0.y));
													  currTriUVs.push_back(glm::vec2(uv1.x, uv1.y));
													  currTriUVs.push_back(glm::vec2(uv2.x, uv2.y));
													  trigUVs.push_back(currTriUVs);

													  /*loadedMesh.facevarying_uvs.push_back(uv0.x);
													  loadedMesh.facevarying_uvs.push_back(uv0.y);

													  loadedMesh.facevarying_uvs.push_back(uv1.x);
													  loadedMesh.facevarying_uvs.push_back(uv1.y);

													  loadedMesh.facevarying_uvs.push_back(uv2.x);
													  loadedMesh.facevarying_uvs.push_back(uv2.y);*/
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
						} // attribute (pos, normal etc.) casing ends
				  } // for attributes loop end


				  // setup triangles in scene
				  for (int i = 0; i < trigVertIdx.size(); i++) {
						//std::cout << "trig number " << i << std::endl;

						std::vector<int>& idx = trigVertIdx[i];
						Triangle newTrig;

						newTrig.vertices[0] = trigPositions[idx[0]];
						newTrig.vertices[1] = trigPositions[idx[1]];
						newTrig.vertices[2] = trigPositions[idx[2]];

						/*std::cout << "Positions are: ["
							  << trigPositions[idx[0]].x << ", " << trigPositions[idx[0]].y << ", " << trigPositions[idx[0]].z << "], ["
							  << trigPositions[idx[1]].x << ", " << trigPositions[idx[1]].y << ", " << trigPositions[idx[1]].z << "], ["
							  << trigPositions[idx[2]].x << ", " << trigPositions[idx[2]].y << ", " << trigPositions[idx[2]].z << "], "
							  << std::endl;*/

						if (trigNormals.size() == trigVertIdx.size()) {
							  newTrig.normal[0] = (trigNormals[i])[0];
							  newTrig.normal[1] = (trigNormals[i])[1];
							  newTrig.normal[2] = (trigNormals[i])[2];
						}

						/*std::cout << "Normals are: ["s
							  << (trigNormals[i])[0].x << ", " << (trigNormals[i])[0].y << ", " << (trigNormals[i])[0].z << "], ["
							  << (trigNormals[i])[1].x << ", " << (trigNormals[i])[1].y << ", " << (trigNormals[i])[1].z << "], ["
							  << (trigNormals[i])[2].x << ", " << (trigNormals[i])[2].y << ", " << (trigNormals[i])[2].z << "], "
							  << std::endl;*/

						

						if (trigUVs.size() == trigVertIdx.size()) {
							  newTrig.uv[0] = (trigUVs[i])[0];
							  newTrig.uv[1] = (trigUVs[i])[1];
							  newTrig.uv[2] = (trigUVs[i])[2];
						}

						scene->triangles.push_back(newTrig);

						currTrigIdx++;
				  }


			} // for primitives loop end -- there should only be the triangle primitive


			newGeom.trigEndIdx = currTrigIdx; // not sure -- need to double check
			scene->geoms.push_back(newGeom);
	  }

	  // testing
	  for (auto& g : scene->geoms) {
			std::cout << "type is " << g.type << ", trigStart: " << g.trigStartIdx << ", trigEnd: " << g.trigEndIdx << std::endl;
	  }


	  // ------------------------- end of testing loading gltf ---------------------------
#endif 

	  // testing
	  for (auto& g : scene->geoms) {
			std::cout << "type is " << g.type << ", trigStart: " << g.trigStartIdx << ", trigEnd: " << g.trigEndIdx << std::endl;
	  }

	  //for (auto& t : scene->triangles) {
			//for (int i = 0; i < 3; i++) {
			//	  std::cout << "vertex: " << t.vertices[i].x << ", " << t.vertices[i].y << ", " << t.vertices[i].z << std::endl;
			//	  std::cout << "normal: " << t.normal[i].x << ", " << t.normal[i].y  << "," << t.normal[i].z << std::endl;
			//}
	  //}




	  // Set up camera stuff from loaded path tracer settings
	  iteration = 0;
	  renderState = &scene->state;
	  Camera& cam = renderState->camera;
	  width = cam.resolution.x;
	  height = cam.resolution.y;

	  glm::vec3 view = cam.view;
	  glm::vec3 up = cam.up;
	  glm::vec3 right = glm::cross(view, up);
	  up = glm::cross(right, view);

	  cameraPosition = cam.position;

	  // compute phi (horizontal) and theta (vertical) relative 3D axis
	  // so, (0 0 1) is forward, (0 1 0) is up
	  glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	  glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	  phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	  theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	  ogLookAt = cam.lookAt;
	  zoom = glm::length(cam.position - ogLookAt);

	  // Initialize CUDA and GL components
	  init();

	  // GLFW main loop
	  mainLoop();

	  return 0;
}

void saveImage() {
	  float samples = iteration;
	  // output image file
	  image img(width, height);

	  for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				  int index = x + (y * width);
				  glm::vec3 pix = renderState->image[index];
				  img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
			}
	  }

	  std::string filename = renderState->imageName;
	  std::ostringstream ss;
	  ss << filename << "." << startTimeString << "." << samples << "samp";
	  filename = ss.str();

	  // CHECKITOUT
	  img.savePNG(filename);
	  //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	  if (camchanged) {
			iteration = 0;
			Camera& cam = renderState->camera;
			cameraPosition.x = zoom * sin(phi) * sin(theta);
			cameraPosition.y = zoom * cos(theta);
			cameraPosition.z = zoom * cos(phi) * sin(theta);

			cam.view = -glm::normalize(cameraPosition);
			glm::vec3 v = cam.view;
			glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
			glm::vec3 r = glm::cross(v, u);
			cam.up = glm::cross(r, v);
			cam.right = r;

			cam.position = cameraPosition;
			cameraPosition += cam.lookAt;
			cam.position = cameraPosition;
			camchanged = false;
	  }

	  // Map OpenGL buffer object for writing from CUDA on a single GPU
	  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	  if (iteration == 0) {
			pathtraceFree();
			pathtraceInit(scene);
	  }

	  if (iteration < renderState->iterations) {
			uchar4* pbo_dptr = NULL;
			iteration++;
			cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

			// execute the kernel
			int frame = 0;
			pathtrace(pbo_dptr, frame, iteration);

			// unmap buffer object
			cudaGLUnmapBufferObject(pbo);
	  } else {
			saveImage();
			pathtraceFree();
			cudaDeviceReset();
			exit(EXIT_SUCCESS);
	  }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	  if (action == GLFW_PRESS) {
			switch (key) {
			case GLFW_KEY_ESCAPE:
				  saveImage();
				  glfwSetWindowShouldClose(window, GL_TRUE);
				  break;
			case GLFW_KEY_S:
				  saveImage();
				  break;
			case GLFW_KEY_SPACE:
				  camchanged = true;
				  renderState = &scene->state;
				  Camera& cam = renderState->camera;
				  cam.lookAt = ogLookAt;
				  break;
			}
	  }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	  if (leftMousePressed) {
			// compute new camera parameters
			phi -= (xpos - lastX) / width;
			theta -= (ypos - lastY) / height;
			theta = std::fmax(0.001f, std::fmin(theta, PI));
			camchanged = true;
	  } else if (rightMousePressed) {
			zoom += (ypos - lastY) / height;
			zoom = std::fmax(0.1f, zoom);
			camchanged = true;
	  } else if (middleMousePressed) {
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			glm::vec3 forward = cam.view;
			forward.y = 0.0f;
			forward = glm::normalize(forward);
			glm::vec3 right = cam.right;
			right.y = 0.0f;
			right = glm::normalize(right);

			cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
			cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
			camchanged = true;
	  }
	  lastX = xpos;
	  lastY = ypos;
}
