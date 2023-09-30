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

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <device_launch_parameters.h>
#include "cfg.h"

#pragma region feature_parameter
//# define camera_jittering 1 // camera antialiasing

#pragma endregion

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
static int* dev_lightIDs = NULL;
// add triangles
static Triangle* dev_triangles = NULL;
static GLTF_Model* dev_gltf_models = NULL;

static Primitive* dev_primitives = NULL;
static LinearBVHNode* dev_BVH_nodes = NULL;

static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;

static glm::vec3* dev_textures = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

#if cache_first_bounce
static ShadeableIntersection* dev_first_intersections_cache = NULL;
#endif

#if material_sort_ID
static int * dev_materialIDs = NULL;
thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections;
thrust::device_ptr<PathSegment> dev_thrust_paths;
thrust::device_ptr<int> dev_thrust_ID;
thrust::zip_iterator<
    thrust::tuple<
    thrust::device_ptr<ShadeableIntersection>,
    thrust::device_ptr<PathSegment>
    >
> zip_it;
#endif

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("malloc device images\n");

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    // light IDs
    cudaMalloc(&dev_lightIDs, scene->lightIDs.size() * sizeof(int));
    cudaMemcpy(dev_lightIDs, scene->lightIDs.data(), scene->lightIDs.size() * sizeof(int), cudaMemcpyHostToDevice);
    // add triangles
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    // and gltf models
    cudaMalloc(&dev_gltf_models, scene->gltf_models.size() * sizeof(GLTF_Model));
    cudaMemcpy(dev_gltf_models, scene->gltf_models.data(), scene->gltf_models.size() * sizeof(GLTF_Model), cudaMemcpyHostToDevice);
    // primitives for BVH intersection
    cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
    cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    checkCUDAError("malloc device primitives \n");
    // linear/compact bvh nodes on gpu
    cudaMalloc(&dev_BVH_nodes, scene->LBVHnodes.size() * sizeof(LinearBVHNode));
    cudaMemcpy(dev_BVH_nodes, scene->LBVHnodes.data(), scene->LBVHnodes.size() * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
    checkCUDAError("malloc device bvh nodes\n");
  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    checkCUDAError("malloc device material\n");
  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    checkCUDAError("malloc device intersections\n");
    // TODO: initialize any extra device memeory you need
#if cache_first_bounce
    cudaMalloc(&dev_first_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if material_sort_ID
    cudaMalloc(&dev_materialIDs, pixelcount * sizeof(int));
    cudaMemset(dev_materialIDs, 0,  pixelcount * sizeof(int));
#endif
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
    cudaFree(dev_lightIDs);
    checkCUDAError("Free device light ids");
  	cudaFree(dev_materials);
    checkCUDAError("Free device materials");
  	cudaFree(dev_intersections);

    cudaFree(dev_triangles);
    cudaFree(dev_gltf_models);
    // TODO: clean up any extra device memory you created
#if cache_first_bounce
    cudaFree(dev_first_intersections_cache);
#endif
#if material_sort
#if material_sort_ID
    cudaFree(dev_materialIDs);
#endif
#endif
    checkCUDAError("pathtraceFree");
}


__host__ __device__
glm::vec2 ConcentricSampleDisk(
    thrust::default_random_engine& rng
    ) {
    // pbrt 
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 t(u01(rng), u01(rng));
    // map to [-1, -1]
    t = 2.0f * t - glm::vec2(1.0f, 1.0f);
    if (t.x == 0.0f && t.y == 0.0f) {
        return t;
    }
    float theta, r;
    if (abs(t.x) > abs(t.y)) {
        r = t.x;
        theta = (PI / 4.0f) * t.y / t.x;
    }
    else {
        r = t.y;
        theta = (PI / 2.0f) - PI / 4.0f * t.x / t.y;
    }

    return r * glm::vec2(cos(theta), sin(theta));
    
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
        segment.colorSum = glm::vec3(0.);
        segment.colorThroughput = glm::vec3(1.);
        
		// TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> du(0.0, 0.5);
        segment.ray.time = du(rng);
#if camera_jittering
        
        //thrust::uniform_real_distribution<float> u01(0, 1);
        thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
			);

#else
        segment.ray.direction = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

#if dof
        // pbrt 6.2.3
        //thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        glm::vec2 offset = ConcentricSampleDisk(rng) * cam.apertureRadius;

        float ft = abs(cam.focusDist / segment.ray.direction.z);
        //glm::vec3 pFocus = getPointOnRay(segment.ray, ft);
        glm::vec3 pFocus = ft * segment.ray.direction;

        segment.ray.origin += glm::vec3(offset, 0.0f);
        segment.ray.direction = glm::normalize(pFocus - glm::vec3(offset, 0.0f));
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__host__ __device__
glm::mat4 dev_buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
    glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}
// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
    int num_paths,
    PathSegment * pathSegments,
    ShadeableIntersection * intersections,
    Geom* geoms, int geoms_size, GLTF_Model* models = dev_gltf_models, Triangle* triangles = dev_triangles,
    Primitive* primitives = dev_primitives, LinearBVHNode* LBVHnodes = dev_BVH_nodes
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

#if RAY_SCENE_INTERSECTION == BRUTE_FORCE
        int prim_idx = SceneIntersection(pathSegment.ray, geoms, geoms_size, models, triangles, intersections[path_index]);
        /*if (prim_idx != - 1 && geoms[prim_idx].type == PLANE) {
            printf("intersect PLANE with t: %f, at pos %f, %f, %f, with mat id: %d, remain bounce: %d\n",
                intersections[path_index].t,
                intersections[path_index].vtx.pos.x, intersections[path_index].vtx.pos.y, intersections[path_index].vtx.pos.z,
                intersections[path_index].materialId,
                pathSegment.remainingBounces
            );
        }*/
#else if RAY_SCENE_INTERSECTION == HBVH
        int prim_idx = SceneIntersection(pathSegment.ray, primitives, LBVHnodes, intersections[path_index]);
       /* if (prim_idx >= 0) {
            if (primitives[prim_idx].type == TRIANGLE) {
                vc3 p = getPointOnRay(pathSegment.ray, intersections[path_index].t);
                printf("intersect triangle with t: %f, at pos %f, %f, %f, with mat id: %d, remain bounce: %d\n", 
                    intersections[path_index].t, 
                    intersections[path_index].vtx.pos.x, intersections[path_index].vtx.pos.y, intersections[path_index].vtx.pos.z,
                    intersections[path_index].materialId,
                    pathSegment.remainingBounces
                );
            }
        }*/
        
#endif
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
        pathSegments[idx].colorSum *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.vtx.normal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].colorSum *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].colorSum *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].colorSum = glm::vec3(0.0f);
    }
  }
}

#pragma region myMaterial
__global__ void shadeTrueMaterial(
    int iter,
    int max_depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    int* lightIDs,
    int light_size,
    int env_light_id_idx,
    Geom* geoms,
    int geom_size,
    Material* materials,
    Triangle* triangles,
    GLTF_Model* gltf_models,
    Primitive* primitives, LinearBVHNode* LBVHnodes,
    glm::vec3* textures,
    glm::vec3* dev_image
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& cur_pathSegment = pathSegments[idx];
        
        bool specularBounce = (cur_pathSegment.prevSample.Bxdf & BSDF_SPECULAR) != 0;
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material itsct_m = materials[intersection.materialId];
            sampleMaterialFromTex(itsct_m, intersection.vtx.uv, textures);
            glm::vec3 materialColor = itsct_m.color;

            // If the material indicates that the object was a light, "light" the ray
            if (itsct_m.emittance > 0.0f) {
#if DirectLightPass == 1:
                if (cur_pathSegment.remainingBounces == max_depth || specularBounce) {
                    cur_pathSegment.colorSum += cur_pathSegment.colorThroughput * (materialColor * itsct_m.emittance);
                    // stop if hit a light
                    cur_pathSegment.remainingBounces = 0;
                }
#else
                cur_pathSegment.colorSum += cur_pathSegment.colorThroughput * (materialColor * itsct_m.emittance);
                // stop if hit a light
                cur_pathSegment.remainingBounces = 0;
#endif
                cur_pathSegment.prevSample = {0.f, 0};
            }
            else {
                vc3& n = intersection.vtx.normal;

                if (itsct_m.normalTexture.valid == 1) {
                    vc3 tangent_normal = glm::normalize((Float)2 * sampleTexture(textures, intersection.vtx.uv, itsct_m.normalTexture) - (Float)1);
                    n = geometry::normalMapping(tangent_normal, n);
                }
#if DirectLightPass == 1
                UniformSampleOneLight(
                    cur_pathSegment,
                    intersection,
                    itsct_m,
                    materials,
                    -cur_pathSegment.ray.direction,
                    light_size,
                    lightIDs,
                    env_light_id_idx,
                    geoms,
                    geom_size,
                    gltf_models,
                    triangles,
                    primitives, LBVHnodes,
                    textures,
                    rng
                );

#endif
#if InDirectLightPass == 1
                scatterRay(
                    cur_pathSegment,
                    intersection,
                    itsct_m,
                    textures,
                    rng
                );
#else:
                cur_pathSegment.remainingBounces = 0;
#endif
            }
            
        }
        else {
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
            //printf("env_light_id: %d \n", env_light_id_idx);
            if (env_light_id_idx == NULL_PRIMITIVE) {
                cur_pathSegment.colorThroughput = glm::vec3(0.0f);
            }
            else {
                //if (cur_pathSegment.remainingBounces == max_depth || specularBounce) {
                //    
                //}
                int env_light_id = lightIDs[env_light_id_idx];
#if RAY_SCENE_INTERSECTION == BRUTE_FORCE
                GeomTransform t = geoms[env_light_id].geomT;
                int mat_id = geoms[env_light_id].materialid;
#elif RAY_SCENE_INTERSECTION == HBVH
                GeomTransform t = geoms[env_light_id].geomT;
                int mat_id = geoms[env_light_id].materialid;
#endif // RAY_SCENE_INTERSECTION == BRUTE_FORCE
                Float weight = 1.0;
                
                if (cur_pathSegment.remainingBounces != max_depth && !specularBounce) {
                    weight = powerHeuristic(
                        1, cur_pathSegment.prevSample.BSDFPdf,
                        1, env_light_pdf(geoms[env_light_id], materials[mat_id], cur_pathSegment.ray.direction)
                        // TODO I want the env light pdf be the first term, but turns out too much fireflies
                    );
                    //printf("env weight %f\n", weight);
                }
                vc3 env = cur_pathSegment.colorThroughput * env_Light_Le(t, cur_pathSegment.ray.direction, materials[mat_id], textures) * weight;
                
                cur_pathSegment.colorSum += env;
            }
            cur_pathSegment.remainingBounces = 0;
        }
    }
        
    
}
#pragma endregion

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
#if _DEBUG
        vc3 c = iterationPath.colorSum;
        if (isfinite(c.x) && isfinite(c.y) && isfinite(c.z)) {
            image[iterationPath.pixelIndex] += c;
        }
        else {
            printf("pixel index : %d meet nan or infinite value: (%f, %f, %f)\n", iterationPath.pixelIndex, c.x, c.y, c.z);
        }
#else
        image[iterationPath.pixelIndex] += iterationPath.colorSum;
#endif
	}
}


// ref:https://thrust.github.io/doc/group__stream__compaction_ga5fa8f86717696de88ab484410b43829b.html
struct parition_not_end
{
    __host__ __device__
        bool operator()(const PathSegment& ps)
    {
        return ps.remainingBounces > 0;
    }
};

struct material_operator_bigger {
    __host__ __device__
        bool operator()(const ShadeableIntersection& intsct1, const ShadeableIntersection& intsct2)
    {

        return intsct1.materialId > intsct2.materialId;
    }
};


struct materialID_operator_bigger {
    __host__ __device__
        bool operator()(const int& id1, const int& id2)
    {

        return id1 > id2;
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

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

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

    bool iterationComplete = false;
	while (!iterationComplete) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if cache_first_bounce
        if (iter == 1 && depth == 0) {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_first_intersections_cache,
                dev_geoms, hst_scene->geoms.size()
                );
            checkCUDAError("no caching first bounce trace one bounce");
            //cudaDeviceSynchronize();
            cudaMemcpy(dev_intersections, dev_first_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else {
            if (depth == 0) {
                cudaMemcpy( dev_intersections, dev_first_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
            else {
                // can not cache the rest depth bounce
                computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    depth,
                    num_paths,
                    dev_paths,
                    dev_intersections,
                    dev_geoms, hst_scene->geoms.size()
                    );
                checkCUDAError("no caching first bounce trace one bounce");
                //cudaDeviceSynchronize();
            }
        }
#else
        computeIntersections <<<numblocksPathSegmentTracing, blockSize1d >>>(
            depth,
            num_paths,
            dev_paths,
            dev_intersections,
            dev_geoms, hst_scene->geoms.size()
            );
        checkCUDAError("no caching first bounce trace one bounce");
         //cudaDeviceSynchronize();
#endif // cache_first_bounce
        depth++;
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
#if material_sort
#if material_sort_ID
         // map intersections -> ID(int) to trigger radix sort
        construct_materialIDs << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_materialIDs);
        // have to sort both intersections and paths by key
        // ref https://stackoverflow.com/questions/6617066/sorting-3-arrays-by-key-in-cuda-using-thrust-perhaps/42484689#42484689
        /*
        thrust::device_vector<ShadeableIntersection> dev_thrust_intersections_vec(dev_intersections, dev_intersections + num_paths);
        thrust::device_vector<PathSegment> dev_thrust_paths_vec(dev_paths, dev_paths + num_paths);
        thrust::device_vector<int> dev_thrust_ID_vec(dev_materialIDs, dev_materialIDs + num_paths);
        thrust::zip_iterator<
            thrust::tuple<
                thrust::device_vector<ShadeableIntersection>::iterator, 
                thrust::device_vector<PathSegment>::iterator>
            > 
            zip_it 
            = thrust::make_zip_iterator(thrust::make_tuple(dev_thrust_intersections_vec.begin(), dev_thrust_paths_vec.begin()));
        thrust::sort_by_key(dev_thrust_ID_vec.begin(), dev_thrust_ID_vec.end(), zip_it);
        */
        dev_thrust_intersections = thrust::device_ptr<ShadeableIntersection>(dev_intersections);
        dev_thrust_paths = thrust::device_ptr<PathSegment>(dev_paths);
        dev_thrust_ID = thrust::device_ptr<int>(dev_materialIDs);
        zip_it = thrust::make_zip_iterator(thrust::make_tuple(dev_thrust_intersections, dev_thrust_paths));
        thrust::sort_by_key(dev_thrust_ID, dev_thrust_ID + num_paths, zip_it);

#else
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_operator_bigger());

#endif
       
        #endif
        shadeTrueMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
        iter,
        traceDepth,
        num_paths,
        dev_intersections,
        dev_paths,
        dev_lightIDs,
        hst_scene->lightIDs.size(),
        hst_scene->environmentLightID_idx,
        dev_geoms,
        hst_scene->geoms.size(),
        dev_materials,
        dev_triangles,
        dev_gltf_models,
        dev_primitives, dev_BVH_nodes,
        dev_textures,
        dev_image
        );
        if (depth > traceDepth) {
            iterationComplete = true;
        }

         // Done: end based off stream compaction results.
        dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, parition_not_end());
        if (dev_path_end == dev_paths) {
            iterationComplete = true;
        }
        else {
            num_paths = dev_path_end - dev_paths;
        }


	}

  // Assemble this iteration and apply it to the image
    
	//finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

__global__ void correctTexturesKernel(glm::vec3* texture, glm::vec3 gamma, int size)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < size)
        texture[index] = glm::pow(texture[index], gamma);
}

void initDeviceTexture(Scene* scene)
{
    int totalMemory = 0;

    for (int i = 0; i < scene->textures.size(); i++)
        totalMemory += scene->textures[i]->xSize * scene->textures[i]->ySize;

    std::cout << "Total texture memory: " << totalMemory / 1024 / 1024 * sizeof(glm::vec3) << " MB." << std::endl;

    std::vector<int> offsetList;

    if (totalMemory > 0)
    {
        cudaMalloc(&dev_textures, totalMemory * sizeof(glm::vec3));

        const int blockSize1d = 128;

        int offset = 0;
        for (int i = 0; i < scene->textures.size(); i++)
        {
            offsetList.push_back(offset);

            Texture* tex = scene->textures[i];
            int size = tex->xSize * tex->ySize;
            cudaMemcpy(dev_textures + offset, tex->pixels, size * sizeof(glm::vec3), cudaMemcpyHostToDevice);

            glm::vec3 gamma = glm::vec3(tex->gamma);
            dim3 numBlocksPixels = (size + blockSize1d - 1) / blockSize1d;
            correctTexturesKernel << <numBlocksPixels, blockSize1d >> > (dev_textures + offset, gamma, size);

            offset += size;
        }
    }

    // Now we need to set all texture descriptor indices
    /*if (scene->state.camera.bokehTexture.index >= 0)
        scene->state.camera.bokehTexture.index = offsetList[scene->state.camera.bokehTexture.index];*/

    for (Material& m : scene->materials)
    {
        if (m.baseColorTexture.index >= 0)
            m.baseColorTexture.index = offsetList[m.baseColorTexture.index];

        if (m.specularTexture.index >= 0)
            m.specularTexture.index = offsetList[m.specularTexture.index];

        if (m.normalTexture.index >= 0)
            m.normalTexture.index = offsetList[m.normalTexture.index];

        if (m.disneyPara.RoughMetalTexture.index >= 0)
            m.disneyPara.RoughMetalTexture.index = offsetList[m.disneyPara.RoughMetalTexture.index];

        // TODO emissive
    }

    checkCUDAError("initializeDeviceTextures end");
}
