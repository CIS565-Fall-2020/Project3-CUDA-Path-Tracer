#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "warpfunctions.h"


#define BOUNDINGBOXINTERSECTIONTEST true
#define DEPTHOFFIELD false
#define ANTIALIASING true
#define CACHEFIRSTBOUNCE !ANTIALIASING
#define DIRECTLIGHTING true
#define MOTIONBLUR false

#define ERRORCHECK 1
#define RECORDEDITERATION 100
#define MOTIONBLUR_VELOCITY glm::vec3(0, 0.96f, 0)

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)


void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
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
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) 
{
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
							   int iter, glm::vec3* image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y)
	{
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

static float gpu_time_accumulator = 0.0f;

static Scene* hst_scene = nullptr;
static glm::vec3* dev_image = nullptr;
static Geom* dev_geoms = nullptr;
static Geom* dev_light_geoms = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_paths = nullptr;
static ShadeableIntersection* dev_intersections = nullptr;

// Extra static variables for device memory, declared here by me 
static PathSegment* dev_first_paths = nullptr;
static ShadeableIntersection* dev_first_intersections = nullptr;

// gltf mesh data
static float* dev_gltf_vertices = nullptr;                  
static unsigned int* dev_gltf_faces = nullptr;
static unsigned int* dev_gltf_verts_offset = nullptr;
static unsigned int* dev_gltf_faces_offset = nullptr;
static float* dev_gltf_bbox_verts = nullptr;


cudaEvent_t iter_event_start = nullptr;
cudaEvent_t iter_event_end = nullptr;

void pathtraceInit(Scene* scene) 
{
	hst_scene = scene;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// if glTF mesh exists
	if (!scene->meshes.empty())
	{
		preprocessGltfData(scene);
	}

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// Initialize the extra device memeory 
	cudaMalloc(&dev_first_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_light_geoms, scene->lightGeoms.size() * sizeof(Geom));
	cudaMemcpy(dev_light_geoms, scene->lightGeoms.data(), scene->lightGeoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaEventCreate(&iter_event_start);
	cudaEventCreate(&iter_event_end);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() 
{
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	
	// Clean up those extra device variables 
	cudaFree(dev_first_paths);
	cudaFree(dev_first_intersections);

	cudaFree(dev_gltf_faces);
	cudaFree(dev_gltf_vertices);
	cudaFree(dev_gltf_faces_offset);
	cudaFree(dev_gltf_verts_offset);
	cudaFree(dev_gltf_bbox_verts);

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

	if (x < cam.resolution.x && y < cam.resolution.y) 
	{
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		
		// Set up the RNG
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);

#if ANTIALIASING
		// Do antialiasing by jittering the ray
		segment.ray = cam.rayCast(x + u01(rng), y + u01(rng));
#else
		segment.ray = cam.rayCast(x, y);
#endif // ANTIALIASING
		
#if DEPTHOFFIELD
		if (cam.lensRadius > 0)
		{
			// Sample point on lens
			glm::vec2 pLens = cam.lensRadius * WarpFunctions::squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
			// Compute point on plane of focus
			glm::vec3 pFocus = segment.ray.origin + cam.focalDist * segment.ray.direction;
			// Update ray for effect of lens
			segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0);
			segment.ray.direction = glm::normalize(pFocus - segment.ray.origin); 
		}
#endif // DEPTHOFFIELD

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int iter,
									 int depth, 
									 int num_paths, 
									 PathSegment* pathSegments, 
									 Geom* geoms, 
									 int geoms_size, 
									 ShadeableIntersection* intersections,
								     unsigned int* faces,
									 float* vertices,
									 unsigned int* num_faces,
									 unsigned int* num_vertices,
									 float* bbox_verts
									 )
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		const PathSegment& pathSegment = pathSegments[path_index];

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

			if (geom.type == GeomType::CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == GeomType::SPHERE)
			{
				Ray tempRay = pathSegment.ray;
#if MOTIONBLUR
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, pathSegment.remainingBounces);
				thrust::uniform_real_distribution<float> u01(0, 1);
				tempRay.origin -= cos((2 * u01(rng) - 1) * PI) * MOTIONBLUR_VELOCITY;
#endif // MOTIONBLUR
				t = sphereIntersectionTest(geom, tempRay, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == GeomType::MESH)
			{
				bool bbox_hit = true;
#if BOUNDINGBOXINTERSECTIONTEST
				int i = geom.meshid;
				Geom bbox_geom;
				bbox_geom.type = GeomType::CUBE;
				glm::vec3 bbox_scale(bbox_verts[i * 6 + 3] - bbox_verts[i * 6 + 0],
									 bbox_verts[i * 6 + 4] - bbox_verts[i * 6 + 1],
									 bbox_verts[i * 6 + 5] - bbox_verts[i * 6 + 2]);

				setGeomTransform(&bbox_geom, geom.transform * getTansformation(glm::vec3(0), glm::vec3(0), bbox_scale));
				t = boxIntersectionTest(bbox_geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				// Do bounding box intersection culling if not
				bbox_hit = t > 0.0f && t < t_min;
#endif // BOUNDINGBOXINTERSECTIONTEST
				if (bbox_hit)
				{
					t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside,
											 faces, vertices, num_faces, num_vertices, bbox_verts);
				}
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t < t_min)
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
			// The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].point = getPointOnRay(pathSegment.ray, t_min);
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].hitGeom = &geoms[hit_geom_index];
		}
	}
}

void preprocessGltfData(Scene* scene)
{
	int num_meshes = scene->getMeshesSize();

	cudaMalloc(&dev_gltf_faces, scene->total_faces * sizeof(unsigned int));
	cudaMalloc(&dev_gltf_vertices, scene->total_vertices * sizeof(float));
	cudaMalloc(&dev_gltf_bbox_verts, 6 * num_meshes * sizeof(float));

	for (int i = 0, face_offset = 0, vertice_offset = 0; i < num_meshes; i++)
	{
		const gltf::Mesh<float>& mesh = scene->meshes[i];
		int cur_num_faces = mesh.faces.size();
		int cur_num_vertices = mesh.vertices.size();

		cudaMemcpy(dev_gltf_faces + face_offset, mesh.faces.data(), cur_num_faces * sizeof(unsigned int),
			cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(dev_gltf_vertices + vertice_offset, mesh.vertices.data(), cur_num_vertices * sizeof(float),
			cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaMemcpy(dev_gltf_bbox_verts + i * 6, mesh.bbox_verts.data(), 6 * sizeof(float),
			cudaMemcpyKind::cudaMemcpyHostToDevice);
		
		scene->faces_per_mesh.push_back(face_offset);
		scene->vertices_per_mesh.push_back(vertice_offset);

		face_offset += cur_num_faces;
		vertice_offset += cur_num_vertices;
	}

	cudaMalloc(&dev_gltf_verts_offset, scene->vertices_per_mesh.size() * sizeof(unsigned int));
	cudaMalloc(&dev_gltf_faces_offset, scene->faces_per_mesh.size() * sizeof(unsigned int));

	cudaMemcpy(dev_gltf_verts_offset, scene->vertices_per_mesh.data(), scene->vertices_per_mesh.size() * sizeof(unsigned int),
		cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gltf_faces_offset, scene->faces_per_mesh.data(), scene->faces_per_mesh.size() * sizeof(unsigned int),
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	checkCUDAError("preprocess gltf data");
}

__global__ void shadeMaterial(int iter,
							  int num_paths,
							  ShadeableIntersection* shadeableIntersections,
							  PathSegment* pathSegments,
							  Material* materials,
							  Geom* lightGeoms,
							  int num_lights)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) // if the intersection exists...
		{	// Set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f)
			{
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else
			{
#if DIRECTLIGHTING
				scatterDirectRay(pathSegments[idx], intersection, material, rng, lightGeoms, num_lights);
#else
				scatterIndirectRay(pathSegments[idx], intersection, material, rng);
#endif // DIRECTLIGHTING
			}
		}
		else
		{// If there was no intersection, color the ray black.
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) 
{
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d((cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
							   (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	float iter_time = 0.f;
	cudaEventRecord(iter_event_start);
#if CACHEFIRSTBOUNCE
	if (iter == 1)
	{
		generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(dev_first_paths, dev_paths, 
				   pixelcount * sizeof(PathSegment), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		checkCUDAError("copy first paths to dev_first_paths");
	}
	else
	{
		cudaMemcpy(dev_paths, dev_first_paths,
				   pixelcount * sizeof(PathSegment), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		checkCUDAError("get first paths from cache");
	}
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif // CACHEFIRSTBOUNCE

	int depth = 0;
	PathSegment* dev_paths_end = dev_paths + pixelcount;
	int num_paths = dev_paths_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	for (int cur_num_paths = num_paths; cur_num_paths > 0; cur_num_paths = dev_paths_end - dev_paths)
	{
		// Clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		dim3 numblocksPathSegmentTracing = (cur_num_paths + blockSize1d - 1) / blockSize1d;

		// Tracing
#if CACHEFIRSTBOUNCE
		if (depth == 0 && iter > 1)
		{
			cudaMemcpy(dev_intersections, dev_first_intersections,
					   pixelcount * sizeof(ShadeableIntersection), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		}
		else
		{
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				depth,
				cur_num_paths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_intersections,
				dev_gltf_faces,
				dev_gltf_vertices，
				dev_gltf_faces_offset,
				dev_gltf_verts_offset,
				dev_gltf_bbox_verts
			);

			// In the first bounce, store first intersections in the cache _dev_first_intersections_ 
			if (depth == 0 && iter == 1)
			{
				cudaMemcpy(dev_first_intersections, dev_intersections,
						   pixelcount * sizeof(ShadeableIntersection), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
			}
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth,
			cur_num_paths,
			dev_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_intersections,
			dev_gltf_faces,
			dev_gltf_vertices，
			dev_gltf_faces_offset,
			dev_gltf_verts_offset,
			dev_gltf_bbox_verts
		);
#endif // CACHEFIRSTBOUNCE

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		// --- Shading Stage ---
		// Before shading, sort the  pathSegments so that pathSegments with the same material are contiguous in memory 
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + cur_num_paths, dev_paths, material_comp());

		// Shade path segments based on intersections and generate new rays by evaluating the BSDF.
		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			cur_num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_light_geoms,
			hst_scene->lightGeoms.size()
		);

		// Stream compact away all of the terminated paths.
		dev_paths_end = thrust::partition(thrust::device, dev_paths, dev_paths_end, raytracing_continuing());
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Calculate how long to finish this iteration
	cudaEventRecord(iter_event_end);
	cudaEventSynchronize(iter_event_end);
	cudaEventElapsedTime(&iter_time, iter_event_start, iter_event_end);
	gpu_time_accumulator += iter_time;

	if (iter == RECORDEDITERATION)
	{
		std::cout << "Elapsed time to finish " << RECORDEDITERATION << " iterations: " << gpu_time_accumulator << "ms" << endl;
		std::cout << "Average time to run a single iteration: " << gpu_time_accumulator / RECORDEDITERATION << "ms" << endl;
	}

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}