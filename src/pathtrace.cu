#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
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
#define SORTMATERIAL 0
#define CACHEFIRSTBOUNCE 0
#define DEPTH_OF_FIELD 0
#define OCTREE 0
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
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_cache_intersections = NULL;
//static PerformanceTimer timer;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static Triangle *dev_triangles = NULL;
static int *dev_sortTriangles = NULL;
static OctreeNode_cuda *dev_octreeVector = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// Construct a Octree
	OctreeNode *root = NULL;
	int octreeDepth = 4;
	constructOctree(root, octreeDepth, -6, 6, -1, 11, -6, 1); // TODO£ºset according to scene file
	// traverse the Octree
	for (int i = 0; i < scene->triangles.size(); i++) {
		traverseOctree(root, scene->triangles[i], i);
	}
	std::vector<int> sortTriangles; // sort Triangles accoring to the OctreeNodes
	std::vector<OctreeNode_cuda> octreeVector; // store octree in an array
	traverseOctreeToArray(root, sortTriangles, octreeVector);

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cout << "geoms.size(): " << scene->geoms.size() << endl;
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	
  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	cout << "triangles.size(): " << scene->triangles.size() << endl;
	cudaMalloc(&dev_sortTriangles, sortTriangles.size() * sizeof(int));
	cudaMemcpy(dev_sortTriangles, sortTriangles.data(), sortTriangles.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_octreeVector, octreeVector.size() * sizeof(OctreeNode_cuda));
	cudaMemcpy(dev_octreeVector, octreeVector.data(), octreeVector.size() * sizeof(OctreeNode_cuda), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	cudaFree(dev_cache_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_triangles);
	cudaFree(dev_sortTriangles);
	cudaFree(dev_octreeVector);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];
		segment.ray.origin = cam.position;

 segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		// depth-of-field
#if DEPTH_OF_FIELD
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		float aperture = 0.2;
	    float angle = u01(rng) * 2 * PI;
		float radius = u01(rng) * aperture;
		float offsetX = radius * cos(angle);
		float offsetY = radius * sin(angle);
		segment.ray.origin += glm::vec3(offsetX, offsetY, 0.0f);
#endif

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
	, Triangle * triangles
	, int * sortTriangles
	, OctreeNode_cuda * octreeVector
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
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH)
			{
#if OCTREE
				t = meshIntersectionTest(geom, triangles, sortTringles, octreeVector, pathSegment.ray, tmp_intersect, tmp_normal, outside);
#else
				t = meshIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);
#endif
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
		pathSegments[idx].remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
	  else {  
		  // pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
		  // pathSegments[idx].color *= u01(rng); // apply some noise because why not
		  PathSegment pathSegment = pathSegments[idx];
		  glm::vec3 intersect = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
		  scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
	  }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
	  pathSegments[idx].remainingBounces = 0;
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

struct noBounce
{
	__host__ __device__
		bool operator()(const PathSegment& pathSegment)
	{
		return pathSegment.remainingBounces > 0;
	}
};

struct materialCmp {
	__host__ __device__
		bool operator()(const ShadeableIntersection& m1, const ShadeableIntersection& m2) {
		return m1.materialId < m2.materialId;
	}
};

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

    // TODO: perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	int num_paths_ini = num_paths;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHEFIRSTBOUNCE
	if (iter == 1 && depth == 0) {
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_cache_intersections
			, dev_triangles
			, dev_sortTriangles
			, dev_octreeVector
			);
		checkCUDAError("cache first bounce");
	}
	else if (depth == 0){
		cudaMemcpy(dev_intersections, dev_cache_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
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
			, dev_sortTriangles
			, dev_octreeVector
			);
		checkCUDAError("trace one bounce");
	}
	cudaDeviceSynchronize();
	depth++;
#else
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		, dev_triangles
		, dev_sortTriangles
		, dev_octreeVector
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();
	depth++;
#endif

	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
	 
	//timer.startGpuTimer(); 
	  // sort the material
#if SORTMATERIAL
	  thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);
	  thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
	  thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections + num_paths, dev_thrust_paths, materialCmp());
#endif

	  shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
		iter,
		num_paths,
		dev_intersections,
		dev_paths,
		dev_materials
	  );
	  dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, noBounce());
	  num_paths = dev_path_end - dev_paths;
	  if (num_paths <= 0 || depth >= traceDepth) {
		  iterationComplete = true; // TODO: should be based off stream compaction results.
	  }
	}
	//timer.endGpuTimer();

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths_ini, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

// Octree
void constructOctree(OctreeNode *&root,
	int maxdepth,
	float xmin, float xmax,
	float ymin, float ymax,
	float zmin, float zmax)
{

	maxdepth--;
	if (maxdepth > 0)
	{
		root = new OctreeNode(xmin, xmax, ymin, ymax, zmin, zmax);
		root->xmin = xmin;
		root->xmax = xmax;
		root->ymin = ymin;
		root->ymax = ymax;
		root->zmin = zmin;
		root->zmax = zmax;
		float xm = (xmax - xmin) / 2;
		float ym = (ymax - ymin) / 2;
		float zm = (zmax - zmin) / 2;
		constructOctree(root->tlf, maxdepth, xmin, xmax - xm, ymax - ym, ymax, zmax - zm, zmax);
		constructOctree(root->tlb, maxdepth, xmin, xmax - xm, ymin, ymax - ym, zmax - zm, zmax);
		constructOctree(root->trf, maxdepth, xmax - xm, xmax, ymax - ym, ymax, zmax - zm, zmax);
		constructOctree(root->trb, maxdepth, xmax - xm, xmax, ymin, ymax - ym, zmax - zm, zmax);
		constructOctree(root->blf, maxdepth, xmin, xmax - xm, ymax - ym, ymax, zmin, zmax - zm);
		constructOctree(root->blb, maxdepth, xmin, xmax - xm, ymin, ymax - ym, zmin, zmax - zm);
		constructOctree(root->brf, maxdepth, xmax - xm, xmax, ymax - ym, ymax, zmin, zmax - zm);
		constructOctree(root->brb, maxdepth, xmax - xm, xmax, ymin, ymax - ym, zmin, zmax - zm);
	}
}

void traverseOctree(OctreeNode *&root, Triangle &t, int Idx) {
	if (root == NULL) {
		return;
	}
	for (int i = 0; i < 3; i++) {
		glm::vec3 v = t.vertices[i];
		if (v[0] >= root->xmin && v[0] <= root->xmax
			&& v[1] >= root->ymin && v[2] <= root->ymax
			&& v[2] >= root->zmin && v[2] <= root->ymax) {
			root->hasTriangle = true;
			if (root->tlf == NULL
				&& root->tlb == NULL
				&& root->trf == NULL
				&& root->trb == NULL
				&& root->blf == NULL
				&& root->blb == NULL
				&& root->brf == NULL
				&& root->brb == NULL) {
				root->triangleIdx.push_back(Idx);
				return;
			}
			traverseOctree(root->tlf, t, Idx);
			traverseOctree(root->tlb, t, Idx);
			traverseOctree(root->trf, t, Idx);
			traverseOctree(root->trb, t, Idx);
			traverseOctree(root->blf, t, Idx);
			traverseOctree(root->blb, t, Idx);
			traverseOctree(root->brf, t, Idx);
			traverseOctree(root->brb, t, Idx);
			break;
		}
	}
}

int traverseOctreeToArray(OctreeNode *root
	, std::vector<int> &sortTriangles
	, std::vector<OctreeNode_cuda> &octreeVector)
{
	OctreeNode_cuda node(
		root->xmin, root->xmax,
		root->ymin, root->ymax,
		root->zmin, root->zmax);

	if (root->tlf == NULL || !root->tlf->hasTriangle) {
		node.tlf = -1;
	}
	else {
		node.tlf = traverseOctreeToArray(root->tlf, sortTriangles, octreeVector);
	}
	if (root->tlb == NULL || !root->tlb->hasTriangle) {
		node.tlb = -1;
	}
	else {
		node.tlb = traverseOctreeToArray(root->tlb, sortTriangles, octreeVector);
	}
	if (root->trf == NULL || !root->trf->hasTriangle) {
		node.trf = -1;
	}
	else {
		node.trf = traverseOctreeToArray(root->trf, sortTriangles, octreeVector);
	}
	if (root->trb == NULL || !root->trb->hasTriangle) {
		node.trb = -1;
	}
	else {
		node.trb = traverseOctreeToArray(root->trb, sortTriangles, octreeVector);
	}
	if (root->blf == NULL || !root->blf->hasTriangle) {
		node.blf = -1;
	}
	else {
		node.blf = traverseOctreeToArray(root->blf, sortTriangles, octreeVector);
	}
	if (root->blb == NULL || !root->blb->hasTriangle) {
		node.blb = -1;
	}
	else {
		node.blb = traverseOctreeToArray(root->blb, sortTriangles, octreeVector);
	}
	if (root->brf == NULL || !root->brf->hasTriangle) {
		node.brf = -1;
	}
	else {
		node.brf = traverseOctreeToArray(root->brf, sortTriangles, octreeVector);
	}
	if (root->brb == NULL || !root->brb->hasTriangle) {
		node.brb = -1;
	}
	else {
		node.brb = traverseOctreeToArray(root->brb, sortTriangles, octreeVector);
	}

	if (root->triangleIdx.size() > 0) {
		node.triangleStart = sortTriangles.size();
		for (int i : root->triangleIdx) {
			sortTriangles.push_back(i);
		}
		node.triangleEnd = sortTriangles.size() - 1;
	}
	octreeVector.push_back(node);
	return octreeVector.size() - 1;
}
