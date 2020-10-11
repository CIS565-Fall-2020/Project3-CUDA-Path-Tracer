#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define USE_SHADE_MATERIAL
#define SORT_RAYS_BY_MATERIALS 
//#define CACHE_FIRST_BOUNCE 
#define STREAM_COMPACT_RAYS 
#define ANTI_ALIASING
//#define DEPTH_OF_FIELD
#define DIRECT_LIGHTING 
//#define MOTION_BLUR 
//#define MOTION_BLUR_2 //Ghost mode lol 
//#define BOKEH
#define BOUNDING_VOLUME

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
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

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection* dev_cache_intersections = NULL;
//Trying 2 different approaches - change after finalizing 
static Triangle* dev_mesh_triangles = NULL;
//static Mesh* dev_mesh = 0; 
static int* dev_num_triangles = 0;
static Geom* dev_lights = 0;

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	const Camera& cam = hst_scene->state.camera;
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
	cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	//Copy Mesh Data from host to device 
	//cudaMalloc(&dev_mesh, sizeof(Mesh));
	//cudaMemcpy(dev_mesh, &scene->mesh, sizeof(Mesh), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_mesh_triangles, scene->mesh.triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_mesh_triangles, scene->mesh.triangles.data(), scene->mesh.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_num_triangles, sizeof(int));
	cudaMemcpy(dev_num_triangles, &scene->mesh.num_triangles, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_cache_intersections);
	//cudaFree(dev_mesh);
	cudaFree(dev_mesh_triangles);
	cudaFree(dev_num_triangles);
	cudaFree(dev_lights);
	checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec3 squareToDiskConcentric(const glm::vec2& sample)
{
	glm::vec2 uOffset = 2.f * sample - glm::vec2(1.f, 1.f);
	if (uOffset.x == 0.f && uOffset.y == 0.f)
	{
		return glm::vec3(0.f, 0.f, 0.f);
	}
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y))
	{
		r = uOffset.x;
		theta = (PI / 4) * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = (PI / 2) - (PI / 4) * (uOffset.x / uOffset.y);
	}
	return r * glm::vec3(std::cos(theta), std::sin(theta), 0.f);
}


__host__ __device__ float getSquared(float x) { return x * x; }

__host__ __device__ glm::vec3 squareToBokeh(const glm::vec2& sample)
{
	//Rejection Sampling
	//Get point on Disc
	glm::vec3 p = glm::vec3(sample, 0.f);
	p = p * 2.f - glm::vec3(1.f);
	p *= 1.5f;
	float x = p.x, y = p.y;
	//Reject point if it doesn't lie on heart
	if ((getSquared(x) + getSquared((5 * y / 4.f) - sqrt(abs(x)))) - 1 < 0) {
		return p;
	}
	return glm::vec3(0.f);
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
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

#ifdef MOTION_BLUR_2 

		//Jitter the ray randomly about any axes 
		glm::vec3 jitteredRayOrigin = u01(rng) * glm::vec3(0.5f, 1.25f, 0.f);
		segment.ray.origin += jitteredRayOrigin;
#endif // MOTION_BLUR_2 

		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
#ifdef ANTI_ALIASING && !CACHE_FIRST_BOUNCE
		float x_offset = u01(rng), y_offset = u01(rng);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + x_offset)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + y_offset)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif // ANTI_ALIASING

#ifdef DEPTH_OF_FIELD
		if (cam.lensRadius > 0.f)
		{
			//Refer to 561 Path Tracer 
			glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
			glm::vec2 point_on_lens(0.f);

#ifdef BOKEH
			point_on_lens = glm::vec2(cam.lensRadius * glm::vec3(glm::rotate(sample, 45.f), 0.f));
#else
			point_on_lens = glm::vec2(squareToDiskConcentric(sample)) * cam.lensRadius;
#endif // BOKEH
			glm::vec3 pof = cam.position + (cam.focalDistance * segment.ray.direction);

			segment.ray.origin = cam.position + (cam.up * point_on_lens.y) + (cam.right * point_on_lens.x);
			segment.ray.direction = glm::normalize(pof - segment.ray.origin);
		}
#endif // DEPTH_OF_FIELD
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
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, Triangle* triangles
	, int num_triangles
	, int iter
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
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
#ifdef MOTION_BLUR
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);
				thrust::uniform_real_distribution<float> u01(0, 1);
				//Jitter the ray randomly about any axes 
				Ray jittered = pathSegment.ray; 
				jittered.origin += u01(rng) * glm::vec3(0.25f, 0.75f, 0.f);
				t = sphereIntersectionTest(geom, jittered, tmp_intersect, tmp_normal, outside);
#else
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
#endif //MOTION_BLUR
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH)
			{
#ifdef BOUNDING_VOLUME
				bool success = false;
				meshBoundingVolumeTest(geom, pathSegment.ray, geom.geomMinCorner, geom.geomMinCorner, tmp_intersect, success); 
				if (success)
				{
					t = trianglesIntersectionTest(geom, triangles, num_triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				}
#else
				//t = meshIntersectionTest(mesh, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				t = trianglesIntersectionTest(geom, triangles, num_triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);

#endif //BOUNDING_VOLUME

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
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
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
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

//This is called in the shading stage 
__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection& currisect = shadeableIntersections[idx];
		PathSegment& currpath = pathSegments[idx];
		if (currpath.remainingBounces > 0 && currisect.t > 0.f) { // if the intersection still exists...

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[currisect.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			// If object is light, stop bouncing
			if (material.emittance > 0.0f) {
				currpath.remainingBounces = 0;
				currpath.color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			else {
				glm::vec3 isect_normal = currisect.surfaceNormal;
				if (currpath.remainingBounces != 0)
				{
					scatterRay(currpath,
						getPointOnRay(currpath.ray, currisect.t),
						isect_normal,
						material,
						rng);
				}
				else
				{
					currpath.color = glm::vec3(0.f);
				}

			}
			--currpath.remainingBounces;
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			currpath.remainingBounces = 0;
			currpath.color = glm::vec3(0.0f);
		}
	}
}

__device__ __host__ int getRandLightIdx(int nLights, thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float rand = u01(rng);

	if (nLights == 0) return 0;
	int lightNum = glm::min((int)glm::floor(rand * nLights), nLights - 1);
	return lightNum;
}

__device__ __host__ glm::vec3 getPointOnSquarePlane(thrust::default_random_engine& rng, Geom light)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float rand = u01(rng);
	glm::vec2 randpoint(u01(rng), u01(rng));
	glm::vec3 point_on_plane = glm::vec3((randpoint - glm::vec2(0.5f)), 0.f);
	//Transform this light to geom's local space 
	glm::vec3 point_on_plane_local = glm::vec3(light.transform * glm::vec4(point_on_plane, 1.f));
	return point_on_plane_local;
}

__global__ void shadeMaterialDirectLighting(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* lights
	, int numLights
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection& currisect = shadeableIntersections[idx];
		PathSegment& currpath = pathSegments[idx];
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		//thrust::uniform_real_distribution<float> u01(0, 1);

		if (currpath.remainingBounces !=  1 && currpath.remainingBounces > 0 && currisect.t > 0.f)
		{
			Material material = materials[currisect.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			// If object is light, stop bouncing
			if (material.emittance > 0.0f) {
				currpath.remainingBounces = 0;
				currpath.color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			else {
				if (currpath.remainingBounces != 0)
				{
					scatterRay(currpath,
						getPointOnRay(currpath.ray, currisect.t),
						currisect.surfaceNormal,
						material,
						rng);
				}
				else
				{
					currpath.color = glm::vec3(0.f);
				}

			}
			--currpath.remainingBounces;
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else if (currpath.remainingBounces == 1 && currisect.t > 0.f)
		{
			Material material = materials[currisect.materialId];
			glm::vec3 materialColor = material.color;
			// If the material indicates that the object was a light, "light" the ray
			// If object is light, stop bouncing
			if (material.emittance > 0.0f) {
				currpath.remainingBounces = 0;
				currpath.color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			else
			{
				int randLight = getRandLightIdx(numLights, rng);
				glm::vec3 point_on_light = getPointOnSquarePlane(rng, lights[randLight]);
				//To set color 
				scatterRay(currpath
					, getPointOnRay(currpath.ray, currisect.t)
					, currisect.surfaceNormal
					, material
					, rng);
				//Ray should hit to the randomly selected light 
				if (!material.hasRefractive)
				{
					currpath.ray.direction = glm::normalize(point_on_light - currpath.ray.origin);
				}
				--currpath.remainingBounces;
			}		
		}
		else
		{
			currpath.remainingBounces = 0;
			currpath.color = glm::vec3(0.0f);
		}
	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

//Helper for sorting rays by material 
struct checkMaterialID {
	__host__ __device__ bool operator()(const ShadeableIntersection& isect1, const ShadeableIntersection& isect2) {
		if (isect1.materialId < isect2.materialId) {
			return 1;
		}
		else return 0;
	}
};

//Helper for stream compacting rays 
struct pathTerminated {
	__host__ __device__ bool operator()(const PathSegment path) {
		if (path.remainingBounces > 0) {
			return 1;
		}
		else return 0;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
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


	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	//std::cout << "PATHTRACE : Num triangles in this mesh are" << scene->mesh.num_triangles << std::endl; 

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete)
	{
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#ifdef CACHE_FIRST_BOUNCE
		if (depth == 0 && iter == 1)
		{
			//For the first iteration, compute the intersections in the cache buffer and copy them into the intersections buffer 
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_cache_intersections
				, dev_mesh_triangles
				, hst_scene->mesh.num_triangles
				, iter);
			checkCUDAError("First bounce cache error");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			depth++;
		}
		if (depth == 0 && iter != 1)
		{
			//For all the other iterations, use the cached intersections instead of computing them again
			cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}

		if (depth != 0)
		{
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_mesh_triangles
				, hst_scene->mesh.num_triangles
				, iter);
			checkCUDAError("Error in trace one bounce");
			cudaDeviceSynchronize();
			depth++;
	}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_mesh_triangles
			, hst_scene->mesh.num_triangles
			, iter);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

#endif // CACHE_FIRST_BOUNCE

		//Ray sorting by material 
#ifdef SORT_RAYS_BY_MATERIALS 
		thrust::device_ptr<ShadeableIntersection> thrust_dev_isects(dev_intersections);
		thrust::device_ptr<PathSegment> thrust_dev_pathsegs(dev_paths);
		thrust::sort_by_key(thrust_dev_isects, thrust_dev_isects + num_paths, thrust_dev_pathsegs, checkMaterialID());
#endif // SORT_RAYS_BY_MATERIALS 


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
//#ifdef USE_SHADE_MATERIAL
//		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
//			iter,
//			num_paths,
//			dev_intersections,
//			dev_paths,
//			dev_materials
//			);
//#else
//		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
//			iter,
//			num_paths,
//			dev_intersections,
//			dev_paths,
//			dev_materials
//			);
//		iterationComplete = true;
//#endif // USE_SHADE_MATERIAL

#ifdef DIRECT_LIGHTING
		shadeMaterialDirectLighting << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_lights,
			hst_scene->lights.size()
			);
#else
		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
#endif // DIRECT_LIGHTING



#ifdef STREAM_COMPACT_RAYS
		//Update the path ending 
		PathSegment* updated_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, pathTerminated());
		num_paths = updated_path_end - dev_paths;

		//Set iteration complete based off stream compaction results.
		//Check if iteration is completed 
		if (num_paths <= 0)
		{
			iterationComplete = true;
			//printf("Iteration complete! \n"); 
		}
		depth++;
#endif // STREAM_COMPACT_RAYS
}

#ifdef USE_SHADE_MATERIAL
	num_paths = dev_path_end - dev_paths;
#endif // USE_SHADE_MATERIAL

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
	}