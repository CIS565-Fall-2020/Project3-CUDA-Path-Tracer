#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <chrono>

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

struct rayContinueJudge {
    __host__ __device__
        bool operator()(const PathSegment& seg) {
        return seg.remainingBounces > 0;// If there is remaining Bounces in the current ray, the ray will continue
    }
};

struct materialIdCompare {
    __host__ __device__
        bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2) {
        return s1.materialId < s2.materialId;
    }
};

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
static Geom* dev_light = NULL;
static ShadeableIntersection* dev_intersections = NULL;
float avgerageTime = 0;
static int num_lights = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection* dev_intersections_cache = NULL;    // cache first iteration.
static Triangle* dev_triangles = NULL;                           // triangles

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
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_light, scene->lights.size() * sizeof(Geom));
    cudaMemcpy(dev_light, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    num_lights = scene->lights.size();
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_intersections_cache);
    cudaFree(dev_triangles);
    cudaFree(dev_light);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int DOF_on, int AA_on)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        constexpr float lens_radius = 0.5f;
        constexpr float focal_distance = 5.5f;
        // TODO: implement antialiasing by jittering the ray

        // Motion blur by the linear combindation of motion vector
        thrust::normal_distribution<float> n01(0, 1);
        float t = abs(n01(rng));
        glm::vec3 view = t * (cam.view + cam.motion) + (1 - t) * (cam.view);
        cam.view = view;

        if (AA_on)
        {
            //Anti Aliasing by jittering the ray
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
            );
        }
        else
        {
            // Normal way
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );
        }
        if (DOF_on)
        {
            // sample point on lens
            float r = u01(rng) * lens_radius;
            float theta = u01(rng) * 2 * PI;
            glm::vec3 p_lens(r * cos(theta), r * sin(theta), 0.0f);

            // compute point on plane of focus
            float ft = focal_distance / glm::abs(segment.ray.direction.z);
            glm::vec3 p_focus = segment.ray.origin + ft * segment.ray.direction;

            // update ray for effect of lens
            segment.ray.origin += p_lens;
            segment.ray.direction = glm::normalize(p_focus - segment.ray.origin);
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
    , PathSegment* pathSegments
    , Geom* geoms
    , Triangle* triangles
    , int geoms_size
    , ShadeableIntersection* intersections
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
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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

// The shade method that is actually used.
__global__ void shadeMaterial(
    int iter
    , int depth
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
        if (pathSegments[idx].remainingBounces <= 0) {
            return;
        }
        if (intersection.t > 0.0f) {// if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (material.emittance > 0.0f) {
                // Intersect with the light source
                pathSegments[idx].color *= (materialColor * material.emittance);// Terminate the path
                pathSegments[idx].remainingBounces = -1;
            }
            else {  
                // Normal intersections.
                glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction; // Caculate the intersection
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng); // Ca;; the scatter ray
            }
        }
        else {    
            // No Intersection detected
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = -1;// Terminate the path
        }
    }
}

__global__ void shadeDirectLightMaterial(
    int iter
    , int depth
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , Geom* lights
    , int num_lights
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathSegments[idx].remainingBounces < 0) 
    {
        return;
    }
    if (idx < num_paths && pathSegments[idx].remainingBounces > 2
        || pathSegments[idx].remainingBounces == 1)
    {
        //Normal shading component
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (material.emittance > 0.0f) {
                // Intersect with the light source
                pathSegments[idx].color *= (materialColor * material.emittance);// Terminate the path
                pathSegments[idx].remainingBounces = -1;
            }
            else {
                // Normal intersections.
                glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction; // Caculate the intersection
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng); // Ca;; the scatter ray
            }
        }
        else {
            // No Intersection detected
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = -1;// Terminate the path
        }

    }
    else if (idx < num_paths && pathSegments[idx].remainingBounces == 2) {

        // Direct Lighting component
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;


            if (material.emittance > 0.0f) {
                // Intersect with the light source
                pathSegments[idx].color *= (materialColor * material.emittance);// Terminate the path
                pathSegments[idx].remainingBounces = -1;
            }
            else {

                int randIdx = glm::min((int)std::floor(u01(rng) * num_lights), num_lights - 1);

                // Sample a point on the light (squareplane form, will work for cube)
                glm::vec4 local_pos = glm::vec4(glm::vec2(u01(rng) - 0.5f, u01(rng)) - 0.5f, 0.f, 1.f);
                glm::vec3 light_pos = glm::vec3(lights[randIdx].transform * local_pos);

                // Scatter ray normally
                glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction; // Caculate the intersection
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng); // Ca;; the scatter ray

                // But then change the ray to point directly at the light, 
                pathSegments[idx].ray.direction = glm::normalize(light_pos - pathSegments[idx].ray.origin);
            }
        }
        else {
            // No Intersection detected
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = -1;// Terminate the path
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
void pathtrace(uchar4* pbo, int frame, int iter, bool sort_by_material, bool cache_first_iteration, bool DOF_on, bool AA_on, bool direct_light) {
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
    int DOF = (DOF_on) ? 1 : 0;
    int AA = (AA_on) ? 1 : 0;
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

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, DOF, AA);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
    bool iterationComplete = false;
    while (!iterationComplete) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // path trace
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        if (cache_first_iteration)
        {
            // Cache first iteration mode on
            if (iter != 0 || (iter == 0 && depth == 0)) {
                computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    depth
                    , num_paths
                    , dev_paths
                    , dev_geoms
                    , dev_triangles
                    , hst_scene->geoms.size()
                    , dev_intersections
                    );
                checkCUDAError("trace one bounce");
                if (iter == 0) {      // bulid the cache
                    cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
                }
            }
            else {                // use the cache
                cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
        }
        else
        {
            // Cache first iteration mode off
            // Normally caculate the intersections
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_triangles
                , hst_scene->geoms.size()
                , dev_intersections
                );
            checkCUDAError("trace one bounce");
        }
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
      // evaluating the BSDF.
      // Start off with just a big kernel that handles all the different
      // materials you have in the scenefile.
      // TODO: compare between directly shading the path segments and shading
      // path segments that have been reshuffled to be contiguous in memory.

        if (sort_by_material)
        {
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialIdCompare());
        }
        // shade
        if (direct_light)
        {
            shadeDirectLightMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter,
                depth,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials,
                dev_light,
                num_lights
                );
            
        }
        else
        {
            shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter,
                depth,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials
                );
        }
        cudaDeviceSynchronize();
        // stream compactions
        //dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, rayContinueJudge());
        //num_paths = dev_path_end - dev_paths; // Update the path numbers
        //iterationComplete = (num_paths <= 0)? true : false;

        // Without stream compactions
        iterationComplete = (depth >= traceDepth)? true : false;

        std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> period = timer_end - timer_start;
        float prev_cpu_time = static_cast<decltype(prev_cpu_time)>(period.count());
        avgerageTime = (avgerageTime * (iter - 1) + prev_cpu_time) / (iter);
        
        // Debug messages
        //cout << "Iterations:" << iter << ", Depth: " << depth << ", Remaining Rays:" << num_paths << endl;
        //cout << "Iterations:" << iter << ", Time: " << prev_cpu_time << ", Average Time" << avgerageTime << endl;
    }
    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}