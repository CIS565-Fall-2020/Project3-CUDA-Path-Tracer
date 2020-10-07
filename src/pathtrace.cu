#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "timer.h"

#define DEPTH_OF_FIELD 1
#define BOUNDING_BOX 1
#define ANTIALIASING 1
#define SORT_MATERIAL 1
#define CACHE_FIRST_ISECT 0

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

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

static ShadeableIntersection* dev_first_intersections = NULL;


// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene* scene) {
    hst_scene = scene;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    for (int i = 0; i < scene->geoms.size(); i++)
    {

        Geom& geom = scene->geoms[i];
        cudaMalloc(&geom.dev_triangles, geom.triangles_size * sizeof(Triangle));
        cudaMemcpy(geom.dev_triangles, (scene->triangles[i]).data(), geom.triangles_size * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
#if SORT_MATERIAL
    cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    if (hst_scene != NULL) {
        for (int i = 0; i < hst_scene->geoms.size(); i++)
        {
            Geom& geom = hst_scene->geoms[i];
            cudaFree(geom.dev_triangles);
        }
    }
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    // TODO: clean up any extra device memory you created
#if CACHE_FIRST_ISECT
    cudaFree(dev_first_intersections);
#endif

    checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u) {
    glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

    if (uOffset.x == 0 && uOffset.y == 0) {
        return glm::vec2(0, 0);
    }

    float theta, r;

    if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
        r = uOffset.x;
        theta = 0.785398f * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = 1.570796f - 0.785398f * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(glm::cos(theta), glm::sin(theta));

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
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        float x_antialias = x;
        float y_antialias = y;

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);

#if ANTIALIASING
        float offset = 0.5f;
        x_antialias += u01(rng);
        y_antialias += u01(rng);
#endif

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (x_antialias - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (y_antialias - (float)cam.resolution.y * 0.5f)
        );

#if DEPTH_OF_FIELD && !CACHE_FIRST_ISECT
        float lensRadius = 0.5f;
        float focalDistance = 11.f;

        thrust::default_random_engine rngDOF = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> uDOF(0, 1);

        glm::vec2 pLens = lensRadius * ConcentricSampleDisk(glm::vec2(uDOF(rngDOF), uDOF(rngDOF)));

        float ft = glm::abs(focalDistance / segment.ray.direction.z);
        glm::vec3 pFocus = segment.ray.origin + segment.ray.direction * ft;

        segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0);
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
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment* pathSegments
    , Geom* geoms
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
            else if (geom.type == OBJ) {

#if BOUNDING_BOX
                if (rectangleIntersectionTest(pathSegment.ray, geom)) {
                    t = meshTriangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
                }
                else {
                    t = -1;
                }
#else
                t = meshTriangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
#endif
            }
            else if (geom.type == SDF1) {
                t = sdfIntersection(geom, pathSegment.ray, tmp_intersect, tmp_normal);
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
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else if (pathSegments[idx].remainingBounces == 1) {
                pathSegments[idx].color = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces -= 1;
            }
            else {
                scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t,
                    intersection.surfaceNormal, material, rng);
                pathSegments[idx].remainingBounces -= 1;
            }
        }
        else {
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

struct should_terminate
{
    __host__ __device__
        bool operator()(const PathSegment& pathSegment)
    {
        return (pathSegment.remainingBounces > 0);
    }
};

struct compareIntersections
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
    {
        return a.materialId > b.materialId;
    }
};


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    //const int traceDepth = 200;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int num_paths_start = num_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    bool iterationComplete = false;

    timer().startGpuTimer();
    while (!iterationComplete) {
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        if (CACHE_FIRST_ISECT && !ANTIALIASING && depth == 0 && iter != 1) {
            // if it is the firts bounce of the noot first intersection, get the saved intersections
            thrust::copy(thrust::device, dev_first_intersections, dev_first_intersections + num_paths_start, dev_intersections);

            // sort intersections with similar materials together
            if (SORT_MATERIAL) {
                thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareIntersections());
            }
        }
        else {
            // clean shading chunks
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

            // tracing
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (depth, num_paths, dev_paths, dev_geoms
                , hst_scene->geoms.size(), dev_intersections);
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();

            // if it is the first bounce of the first iteration, store the intersections
            if (CACHE_FIRST_ISECT && !ANTIALIASING && depth == 0 && iter == 1) {
                thrust::copy(thrust::device, dev_intersections, dev_intersections + num_paths_start, dev_first_intersections);
            }
            else if (SORT_MATERIAL) {
                // sort intersections with similar materials together
                thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareIntersections());
            }
        }
        depth++;

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (iter, num_paths, dev_intersections, dev_paths, dev_materials);

        PathSegment* dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, should_terminate());
        num_paths = dev_path_end - dev_paths;


        if (num_paths == 0) {
            iterationComplete = true;
        }
    }
    timer().endGpuTimer();

    // Assemble this iteration and apply it to the image
    //dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths_start, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
