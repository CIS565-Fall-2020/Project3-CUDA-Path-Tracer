#include "main.h"
#include "preview.h"
#include <cstring>
#include <chrono>

// Jacky added
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

#include <glm/gtc/matrix_inverse.hpp>

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

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

// Jacky added
std::string inputfile = "../scenes/pigHead.obj";
tinyobj::attrib_t attrib;
std::vector<tinyobj::shape_t> shapes;
std::vector<tinyobj::material_t> materials;

std::string warn;
std::string err;

std::vector<int> faces;
std::vector<glm::vec3> vertices;
std::vector<glm::vec3> normals;
int numVertices;

Geom meshBB;
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    // Load obj file (Jacky added)
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    Geom newGeom;
    newGeom.type = MESH;
    newGeom.materialid = 5;
    scene->geoms.push_back(newGeom);

    glm::vec3 minCorner(FLT_MAX);
    glm::vec3 maxCorner(FLT_MIN);

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                vertices.push_back(glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0], 
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]));

                normals.push_back(glm::vec3(
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2]));

                minCorner[0] = glm::min(minCorner[0], attrib.vertices[3 * idx.vertex_index + 0]);
                minCorner[1] = glm::min(minCorner[1], attrib.vertices[3 * idx.vertex_index + 1]);
                minCorner[2] = glm::min(minCorner[2], attrib.vertices[3 * idx.vertex_index + 2]);

                maxCorner[0] = glm::max(maxCorner[0], attrib.vertices[3 * idx.vertex_index + 0]);
                maxCorner[1] = glm::max(maxCorner[1], attrib.vertices[3 * idx.vertex_index + 1]);
                maxCorner[2] = glm::max(maxCorner[2], attrib.vertices[3 * idx.vertex_index + 2]);
            }
            index_offset += fv;
        }
    }

    numVertices = vertices.size();

    meshBB.translation = minCorner - glm::vec3(glm::scale(maxCorner - minCorner) * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f));
    meshBB.scale = maxCorner - minCorner;
    meshBB.rotation = glm::vec3(0.f);

    meshBB.transform = utilityCore::buildTransformationMatrix(meshBB.translation, meshBB.rotation, meshBB.scale);
    meshBB.inverseTransform = glm::inverse(meshBB.transform);
    meshBB.invTranspose = glm::inverseTranspose(meshBB.transform);

    std::cout << numVertices << std::endl;
    std::cout << "meshBB.translation: (" << meshBB.translation.x << ", " << meshBB.translation.y << ", " << meshBB.translation.z << ")" << std::endl;
    std::cout << "meshBB.scale: (" << meshBB.scale.x << ", " << meshBB.scale.y << ", " << meshBB.scale.z << ")" << std::endl;
    for (int i = 0; i < numVertices; i++) {
        glm::vec3 v = vertices[i];
        glm::vec3 n = normals[i];
        // std::cout << "Position: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
        // std::cout << "Normal: (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;
    }

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
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
        Camera &cam = renderState->camera;
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

    // Performance Timing Variables (Jacky added)
    std::chrono::high_resolution_clock::time_point startTime, endTime;

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene, vertices, normals, numVertices, meshBB);

        // Jacky added code for timing purposes
        startTime = std::chrono::high_resolution_clock::now();
    }

    if (iteration < renderState->iterations) {
        uchar4 *pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    } else {
        // Jacky added code for timing purposes
        endTime = std::chrono::high_resolution_clock::now();
        std::cout << "Time Duration: " <<
            std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count() << std::endl;

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
        Camera &cam = renderState->camera;
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
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
