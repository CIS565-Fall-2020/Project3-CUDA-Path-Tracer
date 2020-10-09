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

#define ERRORCHECK 1

#define STREAM_COMPACTION
#define SORT_MATERIAL
// #define DEPTH_OF_FIELD

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

// Copy from my CIS561 HW
__host__ __device__
glm::vec3 squareToDiskConcentric(const glm::vec2 &sample)
{
	// Reference PBRT 13.6.2
	// Map [0, 1] to [-1, -1]
	glm::vec2 sampleOffset = sample * 2.f - glm::vec2(1.f, 1.f);

	// Handle degeneracy at the origin
	if (sampleOffset.x == 0.f && sampleOffset.y == 0.f)
	{
		return glm::vec3(0.f, 0.f, 0.f);
	}

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(sampleOffset.x) > std::abs(sampleOffset.y))
	{
		r = sampleOffset.x;
		theta = PI / 4.f * (sampleOffset.y / sampleOffset.x);
	}
	else {
		r = sampleOffset.y;
		theta = PI / 2.f - PI / 4.f * (sampleOffset.x / sampleOffset.y);
	}

	return glm::vec3(r * std::cos(theta), r * std::sin(theta), 0.f);
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
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

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
// TODO: static variables for device memory, any extra info you need, etc

// Cache the first bounce intersections for re-use
static PathSegment* dev_cachedFirstPaths = NULL;
static ShadeableIntersection* dev_cachedFirstIntersections = NULL;

// Store all triangles used in obj models
static int modelNum = 0;
static glm::vec3** host_modelTriangles = NULL;
static glm::vec3** dev_modelTriangles = NULL;

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

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_cachedFirstPaths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_cachedFirstIntersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cachedFirstIntersections, 0, pixelcount * sizeof(ShadeableIntersection));

	modelNum = scene->objModels.size();
	host_modelTriangles = new glm::vec3*[modelNum];
	for (int i = 0; i < modelNum; ++i)
	{
		cudaMalloc(&(host_modelTriangles[i]), scene->objModels[i].triangles.size() * sizeof(glm::vec3));
		cudaMemcpy(host_modelTriangles[i], scene->objModels[i].triangles.data(),
			scene->objModels[i].triangles.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	}
	cudaMalloc(&dev_modelTriangles, modelNum * sizeof(glm::vec3*));
	cudaMemcpy(dev_modelTriangles, host_modelTriangles, modelNum * sizeof(glm::vec3*), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_cachedFirstIntersections);
	cudaFree(dev_cachedFirstPaths);

	for (int i = 0; i < modelNum; ++i)
	{
		cudaFree(host_modelTriangles[i]);
	}
	delete[] host_modelTriangles;
	cudaFree(dev_modelTriangles);

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

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
#ifdef DEPTH_OF_FIELD
		float lensRadius = 0.5f;
		float focalDistance = 8.5f;
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		glm::vec3 pLens = lensRadius * squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
		float ft = focalDistance / glm::dot(segment.ray.direction, cam.view);
		///float ft = focalDistance / segment.ray.direction.z;
		glm::vec3 pFocus = cam.position + ft * segment.ray.direction;
		segment.ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
		segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);

#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths, 
	PathSegment * pathSegments, Geom * geoms, int geoms_size, 
	ShadeableIntersection * intersections, glm::vec3 ** modelTriangles)
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
			else if (geom.type == OBJMODEL)
			{
				glm::vec3* triangle = modelTriangles[geom.modelid];
				t = objIntersectionTest(geom, pathSegment.ray, triangle, tmp_intersect, tmp_normal, outside);
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

__global__ void shadeMaterial(
	int iter, int num_paths, int depth,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	Material * materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0)
		{
			return;
		}
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			// BSDF
			else {
				scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t),
					intersection.surfaceNormal, material, rng);
				pathSegments[idx].remainingBounces--;
				if (pathSegments[idx].remainingBounces == 0)
				{
					pathSegments[idx].color = glm::vec3(0, 0, 0);
				}
			}
		}
		// If there was no intersection, color the ray black.
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}


struct isRemainingBounceNotZero
{
	__host__ __device__ bool operator()(const PathSegment& segment)
	{
		return segment.remainingBounces > 0;
	}
};

struct CompMaterial
{
	__host__ __device__ bool operator()(const ShadeableIntersection& i0, const ShadeableIntersection& i1)
	{
		return i0.materialId < i1.materialId;
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

	// DONE: perform one iteration of path tracing



	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// Cache the first intersection
	if (iter == 1)
	{
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_cachedFirstPaths);
		checkCUDAError("generate camera ray");
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		cudaMemset(dev_cachedFirstIntersections, 0, pixelcount * sizeof(ShadeableIntersection));
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (depth, num_paths,
			dev_cachedFirstPaths, dev_geoms, hst_scene->geoms.size(), dev_cachedFirstIntersections,
			dev_modelTriangles);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
	}

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// Use cached first intersection
		if (depth == 0)
		{
			cudaMemcpy(dev_paths, dev_cachedFirstPaths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_intersections, dev_cachedFirstIntersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else
		{
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

			// tracing
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (depth, num_paths,
				dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections,
				dev_modelTriangles);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
		
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (iter, num_paths, depth,
			dev_intersections, dev_paths, dev_materials);

#ifdef STREAM_COMPACTION
		// Thrust stream compaction
		// Reomve_if does not work
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths,
			isRemainingBounceNotZero());
		num_paths = dev_path_end - dev_paths;
#endif

#ifdef SORT_MATERIAL
		// Thrust sort materials
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths,
			dev_paths, CompMaterial());
#endif

		depth++;
		//std::cout <<"Iter:" <<iter << " Depth:" << depth << " num paths:" << num_paths << std::endl;
		iterationComplete = (num_paths == 0 || depth == traceDepth);
	}
	
	// Assemble this iteration and apply it to the image
	num_paths = pixelcount;	// reset num_paths
	dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
