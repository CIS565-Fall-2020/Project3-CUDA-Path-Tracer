#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <map>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define MAX_QUEUE_DEPTH 32

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;

// Store currently active and terminated paths for efficiency
static PathSegment* dev_current_paths = NULL;
static PathSegment* dev_terminated_paths = NULL;

static ShadeableIntersection * dev_intersections = NULL;

// Cache for first bounce information to be used at subsequent iterations
static ShadeableIntersection* dev_first_bounce = NULL;
static PathSegment* dev_first_bounce_paths = NULL;

static std::vector<OctNode> hst_octnodes;
static std::map<int, std::vector<int>> hst_octnode_geoms;
static std::vector<int> hst_octnode_geom_indices;
static OctNode* dev_octnodes = NULL;
static int* dev_octnode_geoms = NULL;
static int* dev_octnode_queues = NULL;
static Geom* dev_triangles = NULL;
static int octNodeId = 1; // id counter for octNode
static bool useOctree = false;
static int octreeDepth = 0;
static int maxGeom = 0;
static int totalIterations = 0;

/*
* Update the OctNode bounding box if necessary
*/
void updateBounds(glm::vec3& maxBound, glm::vec3& minBound, int geomIdx) {
    Geom geom = hst_scene->geoms[geomIdx];
    if (geom.max_point.x > maxBound.x) maxBound.x = geom.max_point.x;
    if (geom.max_point.y > maxBound.y) maxBound.y = geom.max_point.y;
    if (geom.max_point.z > maxBound.z) maxBound.z = geom.max_point.z;
    if (geom.min_point.x < minBound.x) minBound.x = geom.min_point.x;
    if (geom.min_point.y < minBound.y) minBound.y = geom.min_point.y;
    if (geom.min_point.z < minBound.z) minBound.z = geom.min_point.z;
}

/*
* Check if given geometry has an overlap with given bounds
*/
bool doesGeomOverlap(int geomIdx, glm::vec3 &minCorner, glm::vec3 &maxCorner) {
    Geom geom = hst_scene->geoms[geomIdx];
    if (geom.min_point.x >= minCorner.x && geom.min_point.x <= maxCorner.x) {
        if (geom.min_point.y > maxCorner.y) return false;
        if (geom.min_point.z > maxCorner.z) return false;
        if (geom.max_point.y < minCorner.y) return false;
        if (geom.max_point.z < minCorner.z) return false;
        return true;
    }
    else if (geom.min_point.y >= minCorner.y && geom.min_point.y <= maxCorner.y) {
        if (geom.min_point.x > maxCorner.x) return false;
        if (geom.min_point.z > maxCorner.z) return false;
        if (geom.max_point.x < minCorner.x) return false;
        if (geom.max_point.z < minCorner.z) return false;
        return true;
    }
    else if (geom.min_point.z >= minCorner.z && geom.min_point.z <= maxCorner.z) {
        if (geom.min_point.x > maxCorner.x) return false;
        if (geom.min_point.y > maxCorner.y) return false;
        if (geom.max_point.x < minCorner.x) return false;
        if (geom.max_point.y < minCorner.y) return false;
        return true;
    }
    return false;
}

void setupOctreeRecursive(int depth, int maxGeom, OctNode &root, std::vector<int> &geoms) {

    root.upFarLeft, root.upFarRight, root.upNearLeft, root.upNearRight, root.downFarLeft, root.downFarRight, root.downNearLeft, root.downNearRight = -1, -1, -1, -1, -1, -1, -1, -1;

    if (depth == 0 || geoms.size() <= maxGeom) {
        // Base case - no need to create children nodes, set this node as leaf
        root.numGeoms = geoms.size();
        hst_octnode_geoms[root.id] = geoms;
        return;
    }

    // Setup potential children nodes
    OctNode upFarLeft, upFarRight, upNearLeft, upNearRight, downFarLeft, downFarRight, downNearLeft, downNearRight;
    std::vector<int> upFarLeftGeoms, upFarRightGeoms, upNearLeftGeoms, upNearRightGeoms, downFarLeftGeoms, downFarRightGeoms, downNearLeftGeoms, downNearRightGeoms;

    // Compute bounds for potential children nodes
    glm::vec3 boundingBoxSizes = (root.maxCorner - root.minCorner) / 2.f;

    glm::vec3 upFarLeftMin(root.minCorner.x, root.minCorner.y + boundingBoxSizes.y, root.minCorner.z);
    glm::vec3 upFarLeftMax(root.minCorner.x + boundingBoxSizes.x, root.maxCorner.y, root.minCorner.z + boundingBoxSizes.z);

    glm::vec3 upFarRightMin(root.minCorner.x + boundingBoxSizes.x, root.minCorner.y + boundingBoxSizes.y, root.minCorner.z);
    glm::vec3 upFarRightMax(root.maxCorner.x, root.maxCorner.y, root.minCorner.z + boundingBoxSizes.z);

    glm::vec3 upNearLeftMin(root.minCorner.x, root.minCorner.y + boundingBoxSizes.y, root.minCorner.z + boundingBoxSizes.z);
    glm::vec3 upNearLeftMax(root.minCorner.x + boundingBoxSizes.x, root.maxCorner.y, root.maxCorner.z);

    glm::vec3 upNearRightMin(root.minCorner.x + boundingBoxSizes.x, root.minCorner.y + boundingBoxSizes.y, root.minCorner.z + boundingBoxSizes.z);
    glm::vec3 upNearRightMax(root.maxCorner);

    glm::vec3 downFarLeftMin(root.minCorner);
    glm::vec3 downFarLeftMax(upNearRightMin);

    glm::vec3 downFarRightMin(root.minCorner.x + boundingBoxSizes.x, root.minCorner.y, root.minCorner.z);
    glm::vec3 downFarRightMax(root.maxCorner.x, root.minCorner.y + boundingBoxSizes.y, root.minCorner.z + boundingBoxSizes.z);

    glm::vec3 downNearLeftMin(root.minCorner.x, root.minCorner.y, root.minCorner.z + boundingBoxSizes.z);
    glm::vec3 downNearLeftMax(root.minCorner.x + boundingBoxSizes.x, root.minCorner.y + boundingBoxSizes.y, root.maxCorner.z);

    glm::vec3 downNearRightMin(root.minCorner.x + boundingBoxSizes.x, root.minCorner.y, root.minCorner.z + boundingBoxSizes.z);
    glm::vec3 downNearRightMax(root.maxCorner.x, root.minCorner.y + boundingBoxSizes.y, root.maxCorner.z);

    // Iterate through geometry within the node and determine where they belong
    for (int geom : geoms) {
        if (doesGeomOverlap(geom, upFarLeftMin, upFarLeftMax)) {
            upFarLeftGeoms.push_back(geom);
            if (upFarLeftGeoms.size() == 1) {
                root.upFarLeft = octNodeId++;
            }
        }
        if (doesGeomOverlap(geom, upFarRightMin, upFarRightMax)) {
            upFarRightGeoms.push_back(geom);
            if (upFarRightGeoms.size() == 1) {
                root.upFarRight = octNodeId++;
            }
        }
        if (doesGeomOverlap(geom, upNearLeftMin, upNearLeftMax)) {
            upNearLeftGeoms.push_back(geom);
            if (upNearLeftGeoms.size() == 1) {
                root.upNearLeft = octNodeId++;
            }
        }
        if (doesGeomOverlap(geom, upNearRightMin, upNearRightMax)) {
            upNearRightGeoms.push_back(geom);
            if (upNearRightGeoms.size() == 1) {
                root.upNearRight = octNodeId++;
            }
        }
        if (doesGeomOverlap(geom, downFarLeftMin, downFarLeftMax)) {
            downFarLeftGeoms.push_back(geom);
            if (downFarLeftGeoms.size() == 1) {
                root.downFarLeft = octNodeId++;
            }
        }
        if (doesGeomOverlap(geom, downFarRightMin, downFarRightMax)) {
            downFarRightGeoms.push_back(geom);
            if (downFarRightGeoms.size() == 1) {
                root.downFarRight = octNodeId++;
            }
        }
        if (doesGeomOverlap(geom, downNearLeftMin, downNearLeftMax)) {
            downNearLeftGeoms.push_back(geom);
            if (downNearLeftGeoms.size() == 1) {
                root.downNearLeft = octNodeId++;
            }
        }
        if (doesGeomOverlap(geom, downNearRightMin, downNearRightMax)) {
            downNearRightGeoms.push_back(geom);
            if (downNearRightGeoms.size() == 1) {
                root.downNearRight = octNodeId++;
            }
        }
    }

    // Recursive calls - handle potential splitting
    if (root.upFarLeft > -1) {
        upFarLeft.id = root.upFarLeft;
        upFarLeft.maxCorner = upFarLeftMax;
        upFarLeft.minCorner = upFarLeftMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, upFarLeft, upFarLeftGeoms);
        hst_octnodes.push_back(upFarLeft);
    }
    if (root.upFarRight > -1) {
        upFarRight.id = root.upFarRight;
        upFarRight.maxCorner = upFarRightMax;
        upFarRight.minCorner = upFarRightMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, upFarRight, upFarRightGeoms);
        hst_octnodes.push_back(upFarRight);
    }
    if (root.upNearLeft > -1) {
        upNearLeft.id = root.upNearLeft;
        upNearLeft.maxCorner = upNearLeftMax;
        upNearLeft.minCorner = upNearLeftMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, upNearLeft, upNearLeftGeoms);
        hst_octnodes.push_back(upNearLeft);
    }
    if (root.upNearRight > -1) {
        upNearRight.id = root.upNearRight;
        upNearRight.maxCorner = upNearRightMax;
        upNearRight.minCorner = upNearRightMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, upNearRight, upNearRightGeoms);
        hst_octnodes.push_back(upNearRight);
    }
    if (root.downFarLeft > -1) {
        downFarLeft.id = root.downFarLeft;
        downFarLeft.maxCorner = downFarLeftMax;
        downFarLeft.minCorner = downFarLeftMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, downFarLeft, downFarLeftGeoms);
        hst_octnodes.push_back(downFarLeft);
    }
    if (root.downFarRight > -1) {
        downFarRight.id = root.downFarRight;
        downFarRight.maxCorner = downFarRightMax;
        downFarRight.minCorner = downFarRightMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, downFarRight, downFarRightGeoms);
        hst_octnodes.push_back(downFarRight);
    }
    if (root.downNearLeft > 1) {
        downNearLeft.id = root.downNearLeft;
        downNearLeft.maxCorner = downNearLeftMax;
        downNearLeft.minCorner = downNearLeftMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, downNearLeft, downNearLeftGeoms);
        hst_octnodes.push_back(downNearLeft);
    }
    if (root.downNearRight > -1) {
        downNearRight.id = root.downNearRight;
        downNearRight.maxCorner = downNearRightMax;
        downNearRight.minCorner = downNearRightMin;
        root.numGeoms = 0;
        setupOctreeRecursive(depth - 1, maxGeom, downNearRight, downNearRightGeoms);
        hst_octnodes.push_back(downNearRight);
    }
}

void setupOctree() {
    
    if (hst_scene->geoms.size() > 0 && octreeDepth > 1) {

        // Setup root
        OctNode root;
        glm::vec3 maxPos(std::numeric_limits<float>::min());
        glm::vec3 minPos(std::numeric_limits<float>::max());
        std::vector<int> rootGeomIndices;
        for (int i = 0; i < hst_scene->geoms.size(); ++i) {
            updateBounds(maxPos, minPos, i);
            rootGeomIndices.push_back(i);
        }
        root.id = 0;
        root.maxCorner = glm::vec3(maxPos.x + 0.005f, maxPos.y + 0.005f, maxPos.z + 0.005f);
        root.minCorner = glm::vec3(minPos.x - 0.005f, minPos.y - 0.005f, minPos.z - 0.005f);
        setupOctreeRecursive(octreeDepth - 1, maxGeom, root, rootGeomIndices);
        hst_octnodes.push_back(root);

        // Sort octnodes
        std::sort(hst_octnodes.begin(), hst_octnodes.end(), [](OctNode const& n1, OctNode const& n2) {
                                                               return n1.id < n2.id;
                                                               });
        // Setup geom indices
        int numGeomsAdded = 0;
        for (int i = 0; i < hst_octnodes.size(); ++i) {
            if (hst_octnodes[i].numGeoms > 0) {
                hst_octnodes[i].geomStartIdx = numGeomsAdded;
                for (int geom : hst_octnode_geoms[i]) {
                    hst_octnode_geom_indices.push_back(geom);
                    numGeomsAdded++;
                }
            }
            else {
                hst_octnodes[i].geomStartIdx = -1;
            }
        }
        // For testing purposes
        /*std::cout << "UpFarLeft: " << root.upFarLeft << std::endl;
        std::cout << "UpFarRight: " << root.upFarRight << std::endl;
        std::cout << "UpNearLeft: " << root.upNearLeft << std::endl;
        std::cout << "UpNearRight: " << root.upNearRight << std::endl;
        std::cout << "DownFarLeft: " << root.downFarLeft << std::endl;
        std::cout << "DownFarRight: " << root.downFarRight << std::endl;
        std::cout << "DownNearLeft: " << root.downNearLeft << std::endl;
        std::cout << "DownNearRight: " << root.downNearRight << std::endl;*/
        for (int i = 0; i < hst_octnodes.size(); ++i) {
            std::cout << "-----------------" << std::endl;
            std::cout << "Node: " << hst_octnodes[i].id << ", " << hst_octnodes[i].numGeoms << " geoms" << std::endl;
            std::cout << "Geom starts at: " << hst_octnodes[i].geomStartIdx << std::endl;
            for (int j = 0; j < hst_octnode_geoms[i].size(); ++j) {
                std::cout << "Geom id: " << hst_octnode_geoms[i][j] << std::endl;
            }
        }

        std::cout << "Geom indices" << std::endl;
        for (int i = 0; i < hst_octnode_geom_indices.size(); ++i) {
            std::cout << i << ": " << hst_octnode_geom_indices[i] << std::endl;
        }
    }
    else {
        // Do not use octree
        useOctree = false;
    }
}

void pathtraceInit(Scene *scene, bool octree, int treeDepth, int geomNumber, int totalIter) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    totalIterations = totalIter;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_current_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_terminated_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_first_bounce, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));
    cudaMemset(dev_first_bounce_paths, 0, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_triangles, hst_scene->num_triangles * sizeof(Geom));
    int triangles_added = 0;
    for (std::map<int, std::vector<Geom>>::iterator iter = hst_scene->meshes.begin(); iter != hst_scene->meshes.end(); ++iter) {
        int k = iter->first;
        hst_scene->geoms[k].triangleStart = triangles_added;
        hst_scene->geoms[k].numTriangles = hst_scene->meshes[k].size();
        cudaMemcpy(dev_triangles + triangles_added, hst_scene->meshes[k].data(), hst_scene->meshes[k].size() * sizeof(Geom), cudaMemcpyHostToDevice);
        triangles_added += hst_scene->meshes[k].size();
    }

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    // Setup the Octree
    useOctree = octree;
    octreeDepth = treeDepth;
    maxGeom = geomNumber;
    if (octree) {
        setupOctree();

        cudaMalloc(&dev_octnodes, hst_octnodes.size() * sizeof(OctNode));
        cudaMemcpy(dev_octnodes, hst_octnodes.data(), hst_octnodes.size() * sizeof(OctNode), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_octnode_geoms, hst_octnode_geom_indices.size() * sizeof(int));
        cudaMemcpy(dev_octnode_geoms, hst_octnode_geom_indices.data(), hst_octnode_geom_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

        int queue_size = MAX_QUEUE_DEPTH * sizeof(int);
        cudaMalloc(&dev_octnode_queues, pixelcount * queue_size);
        cudaMemset(dev_octnode_queues, 0, pixelcount * queue_size);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree(bool octree) {
    octNodeId = 1; // reset octnode id counter
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
    cudaFree(dev_current_paths);
    cudaFree(dev_terminated_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_first_bounce);
    cudaFree(dev_first_bounce_paths);
    cudaFree(dev_triangles);
    if (octree) {
        cudaFree(dev_octnodes);
        cudaFree(dev_octnode_geoms);
        hst_octnodes.clear();
        hst_octnode_geoms.clear();
        hst_octnode_geom_indices.clear();
        cudaFree(dev_octnode_queues);
    }
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool cacheFirstBounce)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        glm::vec2 s(0.f);

        if (!cacheFirstBounce) {
            thrust::default_random_engine rng1 = makeSeededRandomEngine(iter, x, 0);
            thrust::default_random_engine rng2 = makeSeededRandomEngine(iter, y, 0);
            thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
            s = glm::vec2(cam.pixelLength.x * u01(rng1), cam.pixelLength.y * u01(rng2));
        }

		segment.ray.direction = glm::normalize(cam.view
			- (cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f) - s[0])
			- (cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f) - s[1])
			);

        if (cam.lensRadius > 0) {
            glm::vec3 focalP = segment.ray.direction * cam.focalDist + segment.ray.origin;
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
            glm::vec3 sample(u01(rng), u01(rng), 0.f);
            segment.ray.origin += (sample * cam.lensRadius);
            segment.ray.direction = glm::normalize(focalP - segment.ray.origin);
        }

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__host__ __device__ bool MeshBoundsTest(
    Geom& mesh,
    Ray& ray,
    glm::vec3& invDir
)
{
    glm::vec3 min_pointPad(mesh.min_point.x, mesh.min_point.y, mesh.min_point.z);
    glm::vec3 max_pointPad(mesh.max_point.x, mesh.max_point.y, mesh.max_point.z);
    float t0x = (min_pointPad.x - ray.origin.x) * invDir.x;
    float t1x = (max_pointPad.x - ray.origin.x) * invDir.x;
    if (t0x > t1x) {
        float temp = t0x;
        t0x = t1x;
        t1x = temp;
    }
    float t0y = (min_pointPad.y - ray.origin.y) * invDir.y;
    float t1y = (max_pointPad.y - ray.origin.y) * invDir.y;
    if (t0y > t1y) {
        float temp = t0y;
        t0y = t1y;
        t1y = temp;
    }
    if ((t0x > t1y) || (t0y > t1x)) return false;
    if (t0y > t0x) t0x = t0y;
    if (t1y < t1x) t1x = t1y;
    float t0z = (min_pointPad.z - ray.origin.z) * invDir.z;
    float t1z = (max_pointPad.z - ray.origin.z) * invDir.z;
    if (t0z > t1z) {
        float temp = t0z;
        t0z = t1z;
        t1z = temp;
    }
    if ((t0x > t1z) || (t0z > t1x)) return false;
    return true;
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
    , Geom* triangles
    , bool useBound
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
            else if (geom.type == MESH) 
            {
                // check if ray would intersect this mesh
                float xdir = pathSegment.ray.direction.x;
                float ydir = pathSegment.ray.direction.y;

                // Find set of geometry to test against from octree
                float invDir_x = xdir != 0 ? 1.f / xdir : 0.f;
                float invDir_y = ydir != 0 ? 1.f / ydir : 0.f;
                float invDir_z = pathSegment.ray.direction.z != 0 ? 1.f / pathSegment.ray.direction.z : 0.f;
                glm::vec3 invDir(invDir_x, invDir_y, invDir_z);
                if (useBound) {
                    if (MeshBoundsTest(geom, pathSegment.ray, invDir)) {
                        // check for intersection against each triangle
                        for (int j = geom.triangleStart; j < geom.numTriangles + geom.triangleStart; ++j) {
                            t = triangleIntersectionTest(triangles[j], pathSegment.ray, tmp_intersect, tmp_normal, outside);
                            if (t > 0.0f && t_min > t)
                            {
                                t_min = t;
                                hit_geom_index = i;
                                intersect_point = tmp_intersect;
                                normal = tmp_normal;
                            }
                        }
                    }
                }
                else {
                    // check for intersection against each triangle
                    for (int j = geom.triangleStart; j < geom.numTriangles + geom.triangleStart; ++j) {
                        t = triangleIntersectionTest(triangles[j], pathSegment.ray, tmp_intersect, tmp_normal, outside);
                        if (t > 0.0f && t_min > t)
                        {
                            t_min = t;
                            hit_geom_index = i;
                            intersect_point = tmp_intersect;
                            normal = tmp_normal;
                        }
                    }
                }
            }
            else if (geom.type == TANGLECUBE) {
                t = tanglecubeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == BOUND_BOX) {
                t = boundBoxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__host__ __device__ bool octNodeBoundsTest(
    OctNode& node,
    Ray& ray,
    glm::vec3& invDir
)
{
    glm::vec3 minCornerPad(node.minCorner.x, node.minCorner.y, node.minCorner.z);
    glm::vec3 maxCornerPad(node.maxCorner.x, node.maxCorner.y, node.maxCorner.z);
    float t0x = (minCornerPad.x - ray.origin.x) * invDir.x;
    float t1x = (maxCornerPad.x - ray.origin.x) * invDir.x;
    if (t0x > t1x) {
        float temp = t0x;
        t0x = t1x;
        t1x = temp;
    }
    float t0y = (minCornerPad.y - ray.origin.y) * invDir.y;
    float t1y = (maxCornerPad.y - ray.origin.y) * invDir.y;
    if (t0y > t1y) {
        float temp = t0y;
        t0y = t1y;
        t1y = temp;
    }
    if ((t0x > t1y) || (t0y > t1x)) return false;
    if (t0y > t0x) t0x = t0y;
    if (t1y < t1x) t1x = t1y;
    float t0z = (minCornerPad.z - ray.origin.z) * invDir.z;
    float t1z = (maxCornerPad.z - ray.origin.z) * invDir.z;
    if (t0z > t1z) {
        float temp = t0z;
        t0z = t1z;
        t1z = temp;
    }
    if ((t0x > t1z) || (t0z > t1x)) return false;
    return true;
}

__global__ void computeOctreeIntersections(
    int num_paths
    , PathSegment* pathSegments
    , ShadeableIntersection* intersections
    , OctNode* octreeNodes
    , Geom* geoms
    , int* geomIndices
    , int treeDepth
    , int* node_queue
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        float xdir = pathSegment.ray.direction.x;
        float ydir = pathSegment.ray.direction.y;

        // Find set of geometry to test against from octree
        float invDir_x = xdir != 0 ? 1.f / xdir : 0.f;
        float invDir_y = ydir != 0 ? 1.f / ydir : 0.f;
        float invDir_z = pathSegment.ray.direction.z != 0 ? 1.f / pathSegment.ray.direction.z : 0.f;
        glm::vec3 invDir(invDir_x, invDir_y, invDir_z);

        int current_node = 0;
        int offset = path_index * MAX_QUEUE_DEPTH;
        node_queue[offset] = 0; // set root as starting node
        int total_nodes_left = 1;
        int queue_ptr = 1; // add next node id tp this index in queue
        while (total_nodes_left > 0) {
            // get current node
            OctNode node = octreeNodes[node_queue[offset + current_node]];
            total_nodes_left--;
            // Check if you are at leaf node
            if (node.numGeoms > 0 && node.geomStartIdx >= 0) {
                // At leaf node - get the geometry and do the intersection test
                int start = node.geomStartIdx;
                for (int j = 0; j < node.numGeoms; ++j) {
                    Geom geom = geoms[geomIndices[start + j]];

                    if (geom.type == CUBE)
                    {
                        t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                    }
                    else if (geom.type == SPHERE)
                    {
                        t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                    }
                    else if (geom.type == TRIANGLE)
                    {
                        t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                    }

                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                        hit_geom_index = geomIndices[start + j];
                        intersect_point = tmp_intersect;
                        float xi = intersect_point.x;
                        float yi = intersect_point.y;
                        float zi = intersect_point.z;
                        float xn = node.maxCorner.x;
                        float yn = node.maxCorner.y;
                        float yz = node.maxCorner.z;
                        int temp = xi + yi + zi;
                        temp++;
                        normal = tmp_normal;
                    }
                }
                // move to next node in queue
                current_node++;
                if (current_node == MAX_QUEUE_DEPTH) current_node = 0;
            }
            else {
                if (node.upFarLeft > 0 && octNodeBoundsTest(octreeNodes[node.upFarLeft], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.upFarLeft;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                    total_nodes_left++;
                }
                if (node.upFarRight > 0 && octNodeBoundsTest(octreeNodes[node.upFarRight], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.upFarRight;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                    total_nodes_left++;
                }
                if (node.upNearLeft > 0 && octNodeBoundsTest(octreeNodes[node.upNearLeft], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.upNearLeft;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                    total_nodes_left++;
                }
                if (node.upNearRight > 0 && octNodeBoundsTest(octreeNodes[node.upNearRight], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.upNearRight;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                    total_nodes_left++;
                }
                if (node.downFarLeft > 0 && octNodeBoundsTest(octreeNodes[node.downFarLeft], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.downFarLeft;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                    total_nodes_left++;
                }
                if (node.downFarRight > 0 && octNodeBoundsTest(octreeNodes[node.downFarRight], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.downFarRight;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                   total_nodes_left++;
                }
                if (node.downNearLeft > 0 && octNodeBoundsTest(octreeNodes[node.downNearLeft], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.downNearLeft;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                    total_nodes_left++;
                }
                if (node.downNearRight > 0 && octNodeBoundsTest(octreeNodes[node.downNearRight], pathSegment.ray, invDir)) {
                    // add this child to queue
                    node_queue[offset + queue_ptr] = node.downNearRight;
                    queue_ptr++;
                    if (queue_ptr == MAX_QUEUE_DEPTH) queue_ptr = 0;
                   total_nodes_left++;
                }
            }
        }
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
    , int totalIters
    , Camera cam
    , int depth
    , int totalDepth
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;
      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0; // terminate this path
      }
      // Otherwise, compute BSDF
      else {
          float ior1 = 1.f;
          float ior2 = material.indexOfRefraction;
          //pathSegments[idx].color = material.color;
          switch (material.type) {
            case DIFFUSE:
                diffuseScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng, iter, totalIters, depth, totalDepth);
                break;
            case MIRROR:
                mirrorScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
                break;
            case GLOSSY:
                glossyScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng, iter, totalIters, depth, totalDepth);
                break;
            case DIELECTRIC:
                dielectricScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, ior1, ior2, rng);
                break;
            case GLASS:
                glassScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, ior1, ior2, rng);
                break;
            default:
                break;
          }
        pathSegments[idx].remainingBounces -= 1;
        if (pathSegments[idx].remainingBounces == 0) {
            // This was the last bounce, since a non-emissive material was hit set the ray color to black
            pathSegments[idx].color = glm::vec3(0.0f);
        }
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0; // terminate this path
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter, bool cacheFirstBounce, bool sortByMaterial, bool useMeshBounds) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.


    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, cacheFirstBounce);
    checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
    int num_paths_initial = num_paths;
    //std::cout << "initial: " << num_paths_initial << std::endl;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

    if (iter > 1 && depth == 0) {
        if (cacheFirstBounce) {
            cudaMemcpy(dev_intersections, dev_first_bounce, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else {
            if (useOctree) {
                cudaMemset(dev_octnode_queues, 0, num_paths * MAX_QUEUE_DEPTH * sizeof(int));
                checkCUDAError("trace one bounce from octree before");
                computeOctreeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    num_paths
                    , dev_paths
                    , dev_intersections
                    , dev_octnodes
                    , dev_geoms
                    , dev_octnode_geoms
                    , octreeDepth
                    , dev_octnode_queues
                    );
                checkCUDAError("trace one bounce from octree after");
                cudaDeviceSynchronize();
            }
            else {
                computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    depth
                    , num_paths
                    , dev_paths
                    , dev_geoms
                    , hst_scene->geoms.size()
                    , dev_intersections
                    , dev_triangles
                    , useMeshBounds
                    );
                checkCUDAError("trace one bounce");
                cudaDeviceSynchronize();
            }
        }
    }
    else {
        if (useOctree) {
            cudaMemset(dev_octnode_queues, 0, num_paths * MAX_QUEUE_DEPTH * sizeof(int));
            checkCUDAError("trace one bounce from octree before");
            computeOctreeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths
                , dev_paths
                , dev_intersections
                , dev_octnodes
                , dev_geoms
                , dev_octnode_geoms
                , octreeDepth
                , dev_octnode_queues
                );
            checkCUDAError("trace one bounce from octree after");
            cudaDeviceSynchronize();
        }
        else {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                , dev_triangles
                , useMeshBounds
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }
        if (iter == 1 && depth == 0 && cacheFirstBounce) {
            // cache first bounce
            cudaMemcpy(dev_first_bounce, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
    }
	depth++;


	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

    // Sort paths by material
    if (sortByMaterial) {
        thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_sort());
    }

  shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials,
    totalIterations,
    cam,
    depth,
    hst_scene->state.traceDepth
  );

  // Perform stream compaction
  cudaMemcpy(dev_current_paths, dev_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
  thrust::pair<PathSegment*, PathSegment*> res = thrust::stable_partition_copy(thrust::device, dev_paths, dev_paths + num_paths, dev_current_paths, dev_terminated_paths, keep_path());
  num_paths = res.first - dev_current_paths;
  dev_terminated_paths = res.second;
  cudaMemcpy(dev_paths, dev_current_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
  //std::cout << "Depth: " << depth << " , Num paths: " << num_paths << std::endl;

  if (num_paths <= 0) {
      iterationComplete = true;
      dev_terminated_paths -= num_paths_initial; // set dev_terminated_paths to first entry
  }
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths_initial, dev_image, dev_terminated_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
