#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


constexpr bool
	sortByMaterial = false,
	cacheFirstBounce = false;


#define ERRORCHECK 1

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

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
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

static Scene *hst_scene = nullptr;
static glm::vec3 *dev_image = nullptr;
static Geom *dev_geoms = nullptr;
static Material *dev_materials = nullptr;
static PathSegment *dev_paths = nullptr;
static ShadeableIntersection *dev_intersections = nullptr;
static AABBTreeNode *dev_aabbTree = nullptr;
static int aabbTreeRoot;

// static variables for device memory, any extra info you need, etc

static bool firstBounceCached = false;
static ShadeableIntersection *dev_firstBounceIntersections = nullptr;

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// initialize any extra device memeory you need
	if (cacheFirstBounce) {
		cudaMalloc(&dev_firstBounceIntersections, pixelcount * sizeof(ShadeableIntersection));
	}
	firstBounceCached = false;

	cudaMalloc(&dev_aabbTree, scene->aabbTree.size() * sizeof(AABBTreeNode));
	cudaMemcpy(dev_aabbTree, scene->aabbTree.data(), scene->aabbTree.size() * sizeof(AABBTreeNode), cudaMemcpyHostToDevice);

	aabbTreeRoot = scene->aabbTreeRoot;

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

	// clean up any extra device memory you created
	if (cacheFirstBounce) {
		cudaFree(dev_firstBounceIntersections);
	}
	cudaFree(dev_aabbTree);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment *pathSegments) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= cam.resolution.x && y >= cam.resolution.y) {
		return;
	}


	int index = x + (y * cam.resolution.x);
	PathSegment &segment = pathSegments[index];

	thrust::default_random_engine rand = makeSeededRandomEngine(iter, index, -1);
	thrust::uniform_real_distribution<float> dist(-0.5f, 0.5f);

	segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

	// implement antialiasing by jittering the ray
	glm::vec3 dir =
		cam.view -
		cam.right * (cam.pixelLength.x * ((static_cast<float>(x) + dist(rand)) / cam.resolution.x - 0.5f)) -
		cam.up * (cam.pixelLength.y * ((static_cast<float>(y) + dist(rand)) / cam.resolution.x - 0.5f));

	// depth of field
	dir *= cam.focalDistance;
	glm::vec2 aperture = sampleUnitDiskUniform(rand) * cam.aperture;
	glm::vec3 dofOffset = aperture.x * cam.right + aperture.y * cam.up;

	segment.ray.origin = cam.position + dofOffset;
	segment.ray.direction = glm::normalize(dir - dofOffset);

	segment.pixelIndex = index;
	segment.remainingBounces = traceDepth;
}

__device__ float max3(glm::vec3 xyz) {
	return glm::max(xyz.x, glm::max(xyz.y, xyz.z));
}
__device__ float min3(glm::vec3 xyz) {
	return glm::min(xyz.x, glm::min(xyz.y, xyz.z));
}
__device__ bool rayBoxIntersection(const Ray &ray, glm::vec3 min, glm::vec3 max, float far) {
	min = (min - ray.origin) / ray.direction;
	max = (max - ray.origin) / ray.direction;
	float rmin = max3(glm::min(min, max)), rmax = min3(glm::max(min, max));
	return rmin < far && rmax >= rmin && rmax > 0.0f;
}

__device__ bool rayGeomIntersection(const Ray &ray, const Geom &geom, float *dist, glm::vec3 *normal) {
	float t = -1.0f;
	glm::vec3 norm;
	if (geom.type == GeomType::CUBE) {
		t = boxIntersectionTest(geom.implicit, ray, norm);
	} else if (geom.type == GeomType::SPHERE) {
		t = sphereIntersectionTest(geom.implicit, ray, norm);
	} else if (geom.type == GeomType::TRIANGLE) {
		glm::vec2 bary;
		t = triangleIntersectionTest(geom.triangle, ray, &bary);
		if (t >= 0.0f) {
			norm =
				geom.triangle.normals[0] * (1.0f - bary.x - bary.y) +
				geom.triangle.normals[1] * bary.x + geom.triangle.normals[2] * bary.y;
			norm = glm::normalize(norm);
		}
	}
	if (t > 0.0f && t < *dist) {
		*dist = t;
		*normal = norm;
		return true;
	}
	return false;
}

__device__ int traverseAABBTree(
	const Ray &ray, const AABBTreeNode *tree, int root, const Geom *geoms, float *dist, glm::vec3 *normal
) {
	constexpr int geomTestInterval = 4;

	int stack[64], top = 1;
	stack[0] = root;
	int candidates[geomTestInterval * 2], numCandidates = 0;
	int counter = 0, resIndex = -1;
	while (top > 0) {
		const AABBTreeNode &node = tree[stack[--top]];
		bool
			leftIsect = rayBoxIntersection(ray, node.leftAABBMin, node.leftAABBMax, *dist),
			rightIsect = rayBoxIntersection(ray, node.rightAABBMin, node.rightAABBMax, *dist);
		if (leftIsect) {
			if (node.leftChild < 0) {
				candidates[numCandidates++] = ~node.leftChild;
			} else {
				stack[top++] = node.leftChild;
			}
		}
		if (rightIsect) {
			if (node.rightChild < 0) {
				candidates[numCandidates++] = ~node.rightChild;
			} else {
				stack[top++] = node.rightChild;
			}
		}

		if (++counter == geomTestInterval) {
			for (int i = 0; i < numCandidates; ++i) {
				if (rayGeomIntersection(ray, geoms[candidates[i]], dist, normal)) {
					resIndex = candidates[i];
				}
			}
			numCandidates = 0;
			counter = 0;
		}
	}
	for (int i = 0; i < numCandidates; ++i) {
		if (rayGeomIntersection(ray, geoms[candidates[i]], dist, normal)) {
			resIndex = candidates[i];
		}
	}
	return resIndex;
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth, int num_paths, PathSegment *pathSegments,
	const Geom *geoms, int geoms_size, const AABBTreeNode *aabbTree, int aabbTreeRoot,
	ShadeableIntersection *intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths) {
		return;
	}

	PathSegment pathSegment = pathSegments[path_index];

	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;

	hit_geom_index = traverseAABBTree(pathSegment.ray, aabbTree, aabbTreeRoot, geoms, &t_min, &normal);

	if (hit_geom_index == -1)
	{
		intersections[path_index].t = -1.0f;
		intersections[path_index].materialId = -1;
	}
	else
	{
		//The ray hits something
		intersections[path_index].t = t_min;
		intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].surfaceNormal = normal;
	}
}

__global__ void shade(
	int iter, int depth, int num_paths,
	ShadeableIntersection *intersections, PathSegment *paths, Material *materials
) {
	int iSelf = blockIdx.x * blockDim.x + threadIdx.x;
	if (iSelf >= num_paths) {
		return;
	}

	ShadeableIntersection intersection = intersections[iSelf];
	PathSegment path = paths[iSelf];

	if (intersection.materialId != -1) {
		/*path.color = (intersection.surfaceNormal + 1.0f) * 0.5f;
		path.remainingBounces = -1;*/

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, iSelf, depth);
		scatterRay(
			path, path.ray.origin + path.ray.direction * intersection.t, intersection.surfaceNormal,
			materials[intersection.materialId], rng
		);
	} else {
		path.color = glm::vec3(0.0f);
		path.remainingBounces = 0;
	}
	paths[iSelf] = path;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths) {
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct IsRayTravelling {
	__host__ __device__ bool operator()(const PathSegment &path) {
		return path.remainingBounces > 0;
	}
};

struct NoContribution {
	__host__ __device__ bool operator()(const PathSegment &path) {
		return path.remainingBounces != -1;
	}
};

struct MaterialCompare {
	__host__ __device__ bool operator()(const ShadeableIntersection &lhs, const ShadeableIntersection &rhs) {
		return lhs.materialId > rhs.materialId;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
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
	//   * Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	for (int depth = 0; num_paths > 0; ++depth) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		int numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		if (depth == 0 && firstBounceCached) {
			cudaMemcpy(
				dev_intersections, dev_firstBounceIntersections,
				pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice
			);
		} else {
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
				depth, num_paths, dev_paths,
				dev_geoms, hst_scene->geoms.size(), dev_aabbTree, aabbTreeRoot,
				dev_intersections
			);
			checkCUDAError("trace one bounce");

			if (cacheFirstBounce && depth == 0) {
				cudaMemcpy(
					dev_firstBounceIntersections, dev_intersections,
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice
				);
				firstBounceCached = true;
			}
		}

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		if (sortByMaterial) {
			thrust::sort_by_key(
				thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, MaterialCompare()
			);
		}

		shade<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter, depth, num_paths, dev_intersections, dev_paths, dev_materials
		);

		num_paths = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, IsRayTravelling()) - dev_paths;
	}

	// Assemble this iteration and apply it to the image
	int gather_paths = thrust::remove_if(thrust::device, dev_paths, dev_path_end, NoContribution()) - dev_paths;
	if (gather_paths > 0) {
		dim3 numBlocksPixels = (gather_paths + blockSize1d - 1) / blockSize1d;
		finalGather<<<numBlocksPixels, blockSize1d>>>(gather_paths, dev_image, dev_paths);
	}

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
			pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
