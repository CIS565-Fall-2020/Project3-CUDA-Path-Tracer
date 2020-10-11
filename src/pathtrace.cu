#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <math.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_INTER 1
#define ANTIALIASING 1
#define PI 3.1415926535897932384626f
#define POST 0

#define DEPTH_OF_FIELD 0
#define LENS_RADIUS 0.3
#define FOCUS_DISTANCE 7

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
static glm::vec3* dev_texture = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
#if CACHE_FIRST_INTER
    static ShadeableIntersection* dev_first_intersections = NULL;
#endif

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
#if CACHE_FIRST_INTER 
    cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
#endif
    cudaMalloc(&dev_texture, scene->texture.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_texture, scene->texture.data(), scene->texture.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
#if CACHE_FIRST_INTER
    cudaFree(dev_first_intersections);
#endif
    cudaFree(dev_texture);
    //cudaFree(dev_bb);
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
        float x_value = x;
        float y_value = y;
#if ANTIALIASING
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        float aa_factor = 1.0f;
        // get it in range from -aa_factor to aa_factor
        float x_rand = (u01(rng) * aa_factor * 2) - aa_factor;
        x_value += x_rand;

        float y_rand = (u01(rng) * aa_factor * 2) - aa_factor;
        y_value += y_rand;
#endif  
        segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * (x_value - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (y_value - (float)cam.resolution.y * 0.5f)
			);

#if DEPTH_OF_FIELD
        // sample point on circle lens
        thrust::uniform_real_distribution<float> u01_dof(0, 1);
        glm::vec3 lens_sample = squareToDiskConcentric(glm::vec2(u01_dof(rng), u01_dof(rng)));
        lens_sample *= LENS_RADIUS;

        // calculate focus point
        float t_val = glm::abs(FOCUS_DISTANCE / segment.ray.direction.z);
        glm::vec3 focus_point = t_val * segment.ray.direction;

        segment.ray.origin += lens_sample;
        segment.ray.direction = glm::normalize(focus_point - lens_sample);
#endif
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
    , Material* materials
    , glm::vec3* textures
    , BoundingBox bounding_box
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
        glm::vec2 uv;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

        bool intersects_bb = false;
        intersects_bb = boundingBoxIntersection(pathSegment.ray, bounding_box);

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersection(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersection(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE)
            {
                t = triangleIntersection(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
            else {
                t = FLT_MAX;
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
                uv = tmp_uv;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
            Material material = materials[geoms[hit_geom_index].materialid];
            if (material.has_bump_map) {
                float w = material.tex_bump_width - 1;
                float h = material.tex_bump_height - 1;
                if (uv[1] < 0.5f) {
                    uv[1] = uv[1] + 2.0f * (0.5f - uv[1]);
                }
                else {
                    uv[1] - 2 * (uv[1] - 0.5f);
                }

                int x = uv[0] * (material.tex_bump_width - 1);
                int y = (1.f - uv[1]) * (material.tex_bump_height - 1);
                int idx = y * material.tex_bump_width + x + material.tex_bump_index;
                
                // get the color from the bum map
                glm::vec3 texColor = textures[idx];

                // convert bump map to normal
                glm::vec3 mapNor = texColor * glm::vec3(2.f) - glm::vec3(1.f);
                 
                // calculate TBN
                glm::vec3 nor, tan, bit;
                Geom geom = geoms[hit_geom_index];
                computeTBN(geom, intersect_point, &nor, &tan, &bit);

                nor = glm::normalize(multiplyMV(geom.transform, glm::vec4(nor, 0.f)));
                tan = glm::normalize(multiplyMV(geom.transform, glm::vec4(tan, 0.f)));
                bit = glm::normalize(multiplyMV(geom.transform, glm::vec4(bit, 0.f)));
                glm::mat3 TBN = glm::mat3(tan, bit, nor);
                normal = TBN * mapNor;
            }
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
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
__global__ void shader(
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
    , glm::vec3* textures
    , int depth
    , int cam_x
    , int cam_y
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    if (pathSegments[idx].remainingBounces < 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 intersection_point = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
      if (material.tex_index != -1 && !material.is_procedural) {
          int pixel_x = intersection.uv[0] * (material.tex_width - 1);
          int pixel_y = (1.f - intersection.uv[1]) * (material.tex_height - 1);
          int idx = pixel_y * material.tex_width + pixel_x + material.tex_index;
          glm::vec3 texColor = textures[idx];
          material.color = texColor;
      }
      if (material.is_procedural) {
          // based off of: https://iquilezles.org/www/articles/palettes/palettes.htm
          material.color = glm::vec3(0.5, 0.5, 0.5) + glm::vec3(0.5, 0.5, 0.5) * cos(0.5f * PI * (glm::vec3(2.f, 1.f, 1.f) * intersection_point.x + glm::vec3(0.50, 0.20, 0.25)));
      }
      glm::vec3 materialColor = material.color;
      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
#if POST
        glm::vec3 textureColor = pathSegments[idx].color;
        float grey = 0.21 * textureColor[0] +
            0.72 * textureColor[1] +
            0.07 * textureColor[2];

        int index = pathSegments[idx].pixelIndex;
        int x_point = (index % cam_y);
        int y_point = ((index - x_point) / cam_y);

        glm::vec2 centerCoords = glm::vec2(cam_x / 2,
            cam_y / 2);
        
        float maxDistance = sqrt((centerCoords.x * centerCoords.x) +
            (centerCoords.y * centerCoords.y));

        float shortX = x_point - centerCoords.x;
        float shortY = y_point - centerCoords.y;
        float currentDistance =  (shortX * shortX) / (cam_x * cam_x) +
                                 (shortY * shortY) / (cam_y * cam_y);

        float intensity = 2.3f; // how intense the vignette is
        float vignette = currentDistance * intensity * grey;
        grey = grey - vignette;

        pathSegments[idx].color = glm::vec3(grey, grey, grey);
#endif
        pathSegments[idx].remainingBounces = 0; // stop iterating!
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      else {
          // decrement the number of bounces we have left
          pathSegments[idx].remainingBounces -= 1;
          if (pathSegments[idx].remainingBounces == 0) {
              // last intersection did not hit a light ):
              pathSegments[idx].color = glm::vec3(0.f); // suggested by piazza post @146
              return;
          }
          // keep going! we have more bounces left to generate new ray
          // we want to call scatter ray... need to calculate intersection location (in world space)
          glm::vec3 inter = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
          scatterRay(pathSegments[idx], inter, intersection.surfaceNormal, material, rng, iter, depth);
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0; // stop iterating!
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

// based off of: https://thrust.github.io/doc/group__stream__compaction_ga5fa8f86717696de88ab484410b43829b.html
struct is_path_terminated
{
    __host__ __device__ bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
     }
};

#if SORT_BY_MATERIAL
struct compare_materials {
    __host__ __device__ bool operator()(const ShadeableIntersection& intersect1, const ShadeableIntersection& intersect2) {
        return intersect1.materialId > intersect2.materialId;
    }
};
#endif

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const BoundingBox& bounding_box = hst_scene->bounding_box;

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

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
    while (num_paths > 0) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // here we want to cache the first intersection
        bool intersections_computed = false;

#if CACHE_FIRST_INTER && !ANTIALIASING
        if (depth == 0) {
            intersections_computed = true;
            if (iter <= 1) {
                computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    depth
                    , num_paths
                    , dev_paths
                    , dev_geoms
                    , hst_scene->geoms.size()
                    , dev_first_intersections
                    , dev_materials
                    , dev_texture
                    , bounding_box
                    );
                checkCUDAError("trace first bounce, first iter");
            }
            thrust::copy(thrust::device, dev_first_intersections, dev_first_intersections + pixelcount, dev_intersections);
        }
#endif
        if (!intersections_computed) {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                , dev_materials
                , dev_texture
                , bounding_box
                );
            checkCUDAError("trace one bounce");
        }
        cudaDeviceSynchronize();
	    depth++;

	    // --- Shading Stage ---
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if SORT_BY_MATERIAL // we want to reorder the segments so that ones with same materials are continguous in memory
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_materials());
#endif

        shader<<<numblocksPathSegmentTracing, blockSize1d>>> (
          iter,
          num_paths,
          dev_intersections,
          dev_paths,
          dev_materials,
          dev_texture,
          depth,
          cam.resolution.x, 
          cam.resolution.y
        );
        // stream compaction here: We want to remove all the paths with remaining bounces = 0;
        // changed to stable partition from remove_if after piazze @149
        dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, is_path_terminated());
        num_paths = dev_path_end - dev_paths;
	}

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    // wow, was working on this bug for so long, turns out it was because num_paths was changed in loop above to 0
	finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths); 

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
