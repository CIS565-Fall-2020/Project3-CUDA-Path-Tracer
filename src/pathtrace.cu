#include <cstdio>
#include <cuda.h>
#include <cmath>
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

static Octree octree;
static OctNode* dev_octnodes = NULL;
static OctNode* hst_octnodes = NULL;
static int octNodeId = 1; // id counter for octNode

/*
* Update the OctNode bounding box if necessary
*/
void updateBounds(glm::vec3& maxBound, glm::vec3& minBound, Geom &geom) {
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
bool doesGeomOverlap(Geom& geom, glm::vec3 &minCorner, glm::vec3 &maxCorner) {
    return (geom.min_point.x >= minCorner.x) ||
        (geom.min_point.y >= minCorner.y) ||
        (geom.min_point.z >= minCorner.z) ||
        (geom.max_point.x <= maxCorner.x) ||
        (geom.max_point.y <= maxCorner.y) ||
        (geom.max_point.z <= maxCorner.z);
}

void setupOctreeRecursive(int depth, glm::vec3 maxRoot, glm::vec3 minRoot, OctNode &root) {

    // Setup potential children nodes
    root.upFarLeft, root.upFarRight, root.upNearLeft, root.upNearRight, root.downFarLeft, root.downFarRight, root.downNearLeft, root.downNearRight = -1;
    OctNode upFarLeft, upFarRight, upNearLeft, upNearRight, downFarLeft, downFarRight, downNearLeft, downNearRight;
    std::vector<Geom> upFarLeftGeoms, upFarRightGeoms, upNearLeftGeoms, upNearRightGeoms, downFarLeftGeoms, downFarRightGeoms, downNearLeftGeoms, downNearRightGeoms;

    // Compute bounds for potential children nodes
    glm::vec3 boundingBoxSizes = (maxRoot - minRoot) / 2.f;

    glm::vec3 upFarLeftMin(minRoot.x, minRoot.y + boundingBoxSizes.y, minRoot.z);
    glm::vec3 upFarLeftMax(minRoot.x + boundingBoxSizes.x, maxRoot.y, minRoot.z + boundingBoxSizes.z);

    glm::vec3 upFarRightMin(minRoot.x + boundingBoxSizes.x, minRoot.y + boundingBoxSizes.y, minRoot.z);
    glm::vec3 upFarRightMax(maxRoot.x, maxRoot.y, minRoot.z + boundingBoxSizes.z);

    glm::vec3 upNearLeftMin(minRoot.x, minRoot.y + boundingBoxSizes.y, minRoot.z + boundingBoxSizes.z);
    glm::vec3 upNearLeftMax(minRoot.x + boundingBoxSizes.x, maxRoot.y, maxRoot.z);

    glm::vec3 upNearRightMin(minRoot.x + boundingBoxSizes.x, minRoot.y + boundingBoxSizes.y, minRoot.z + boundingBoxSizes.z);
    glm::vec3 upNearRightMax(maxRoot);

    glm::vec3 downFarLeftMin(minRoot);
    glm::vec3 downFarLeftMax(upNearRightMin);

    glm::vec3 downFarRightMin(minRoot.x + boundingBoxSizes.x, minRoot.y, minRoot.z);
    glm::vec3 downFarRightMax(maxRoot.x, minRoot.y + boundingBoxSizes.y, minRoot.z + boundingBoxSizes.z);

    glm::vec3 downNearLeftMin(minRoot.x, minRoot.y, minRoot.z + boundingBoxSizes.z);
    glm::vec3 downNearLeftMax(minRoot.x + boundingBoxSizes.x, minRoot.y + boundingBoxSizes.y, maxRoot.z);

    glm::vec3 downNearRightMin(minRoot.x + boundingBoxSizes.x, minRoot.y, minRoot.z + boundingBoxSizes.z);
    glm::vec3 downNearRightMax(maxRoot.x, minRoot.y + boundingBoxSizes.y, maxRoot.z);

    // Iterate through geometry within the node and determine where they belong
    for (Geom geom : root.geoms) {
        // Up far left
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
    }
}

void setupOctree() {
    if (hst_scene->geoms.size() == 0) {
        // No geometry - set empty tree
        octree.rootId = -1;
    } else {
        // Setup root
        OctNode root;
        root.id = 0;
        glm::vec3 maxPos(std::numeric_limits<float>::min());
        glm::vec3 minPos(std::numeric_limits<float>::max());
        for (Geom geom : hst_scene->geoms) {
            updateBounds(maxPos, minPos, geom);
            root.geoms.push_back(geom);
        }
        root.maxCorner = maxPos;
        root.minCorner = minPos;
        hst_octnodes[0] = root;
    }
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    hst_octnodes = new OctNode[scene->geoms.size()];

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_current_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_terminated_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_first_bounce, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));
    cudaMemset(dev_first_bounce_paths, 0, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_octnodes, scene->geoms.size() * sizeof(OctNode));

    checkCUDAError("pathtraceInit");

    // Setup the Octree
    setupOctree();
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
    cudaFree(dev_current_paths);
    cudaFree(dev_terminated_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_first_bounce);
    cudaFree(dev_first_bounce_paths);
    cudaFree(dev_octnodes);
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
            thrust::uniform_real_distribution<float> u01(0, 1);
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
// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
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
            else if (geom.type == TRIANGLE) 
            {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
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
          switch (material.type) {
            case DIFFUSE:
                diffuseScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
                break;
            case MIRROR:
                mirrorScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
                break;
            case GLOSSY:
                glossyScatter(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
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
void pathtrace(uchar4 *pbo, int frame, int iter, bool cacheFirstBounce, bool sortByMaterial) {
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
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }
    }
    else {
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
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
    dev_materials
  );

  // Perform stream compaction
  cudaMemcpy(dev_current_paths, dev_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
  thrust::pair<PathSegment*, PathSegment*> res = thrust::stable_partition_copy(thrust::device, dev_paths, dev_paths + num_paths, dev_current_paths, dev_terminated_paths, keep_path());
  num_paths = res.first - dev_current_paths;
  dev_terminated_paths = res.second;
  cudaMemcpy(dev_paths, dev_current_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

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
