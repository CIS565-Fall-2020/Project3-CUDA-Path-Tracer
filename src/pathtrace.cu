#include <cstdio>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <chrono>
#include <iostream>
#include <iomanip>

#define ERRORCHECK 1
#define matSort false
#define CACHE_FIRST_BOUNCE false
#define stochasticAlias true
#define DIRECT_LIGHTING true
#define DOF false


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

static int* dev_materialIds;
static PathSegment* dev_cachedPaths = NULL;
static ShadeableIntersection* dev_cachedIntersections = NULL;

//static std::chrono::steady_clock::time_point timer;
//static ShadeableIntersection* dev_firstHit = NULL;

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

    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));
    cudaMalloc(&dev_cachedPaths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_cachedIntersections, pixelcount * sizeof(ShadeableIntersection));

}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_materialIds);
    cudaFree(dev_cachedPaths);
    cudaFree(dev_cachedIntersections);
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

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void rayGenerator_withAntiAliasing(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    thrust::default_random_engine rng = makeSeededRandomEngine(x, y, 0);
    thrust::uniform_real_distribution<float> u(-1, 1);

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u(rng) - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;


        
        if (DOF) {

            glm::vec3 focalPoint = segment.ray.origin + (8.0f * segment.ray.direction);
            glm::vec3 rand{ u(rng) / 2 , u(rng) / 2 , 0 };
            rand = rand * 0.5f;
            segment.ray.origin = segment.ray.origin + rand;
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        }

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
            else if (geom.type == TRIANGLE) {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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

__global__ void BSDFShader(int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , glm::vec3 globalLight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) { return; }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment segment = pathSegments[idx];

    // If there was no intersection, color the ray black.
    if (intersection.t <= 0.0f) {
        segment.remainingBounces = 0;
        segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
        pathSegments[idx] = segment;
        return;
    }

    Material mat = materials[intersection.materialId];
    glm::vec3 matColor = mat.color;

    // If the material indicates that the object was a light, "light" the ray
    if (mat.emittance > 0.0f) {
        segment.color = segment.color * (matColor * mat.emittance);
        segment.remainingBounces = 0;
        pathSegments[idx] = segment;
        return;
    }

    // Else we handle the case that we hit a regular object
    // First we update the color of the path segment, then we can compute the new ray direction
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
    scatterRay(segment, getPointOnRay(segment.ray, intersection.t), intersection.surfaceNormal, mat, rng, iter, segment.remainingBounces);

    if (segment.remainingBounces == 0) {
        segment.color = segment.color * globalLight;
    }
    pathSegments[idx] = segment;
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

struct isPathAlive {
    __host__ __device__
        bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
    }
};

struct compareMat {
    __host__ __device__
        bool operator()(const ShadeableIntersection& MatA, const ShadeableIntersection& MatB) {
        return MatA.materialId > MatB.materialId;
    }
};

__global__ void getMatID(int paths, ShadeableIntersection* shadeableIntersections, int* result) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= paths) { return; }
    result[idx] = shadeableIntersections[idx].materialId;
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    glm::vec3 global_Light = hst_scene->globalLight;

    const int numPx = cam.resolution.x * cam.resolution.y;

    if (!DIRECT_LIGHTING) {
        global_Light = glm::vec3(0, 0, 0);
    }

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

        if (iter == 1 || !CACHE_FIRST_BOUNCE) {
        if (stochasticAlias) {
            rayGenerator_withAntiAliasing << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
        }
        else {
            generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
        }
    }
    else {
        cudaMemcpy(dev_paths, dev_cachedPaths, numPx * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_intersections, dev_cachedIntersections, numPx * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
    }

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + numPx;
    int num_paths = dev_path_end - dev_paths;
    int total_paths = num_paths;
    isBouncing data_compactor;

    bool iterating = false;
    while (!iterating) {

        dim3 blockCount = (num_paths + blockSize1d - 1) / blockSize1d;
        if (iter == 1 || !CACHE_FIRST_BOUNCE || depth > 0) {
            cudaMemset(dev_intersections, 0, numPx * sizeof(ShadeableIntersection));

            computeIntersections << <blockCount, blockSize1d >> > (
                depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
            cudaDeviceSynchronize();

            if (matSort) {
                getMatID << <blockCount, blockSize1d >> > (num_paths, dev_intersections, dev_materialIds);
                thrust::device_ptr<ShadeableIntersection> dev_intersections_start(dev_intersections);
                thrust::device_ptr<PathSegment> dev_paths_start(dev_paths);
                thrust::device_ptr<int> dev_materialIds_start(dev_materialIds);
                thrust::zip_iterator<thrust::tuple<thrust::device_ptr<ShadeableIntersection>, thrust::device_ptr<PathSegment>>> zipped = thrust::make_zip_iterator(thrust::make_tuple(dev_intersections_start, dev_paths_start));
                thrust::sort_by_key(dev_materialIds_start, dev_materialIds_start + num_paths, zipped);
            }
        }

        if (iter == 1 && CACHE_FIRST_BOUNCE && depth == 0) {
            cudaMemcpy(dev_cachedPaths, dev_paths, numPx * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_cachedIntersections, dev_intersections, numPx * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }

        BSDFShader << <blockCount, blockSize1d >> > (iter, num_paths, dev_intersections, dev_paths, dev_materials, global_Light);
        cudaDeviceSynchronize();

        thrust::device_ptr<PathSegment> start(dev_paths);
        thrust::device_ptr<PathSegment> end(dev_path_end);
        end = thrust::partition(start, end, data_compactor);

        dev_path_end = thrust::raw_pointer_cast(end);
        num_paths = dev_path_end - dev_paths;
        depth++;

        if (num_paths == 0 || depth > traceDepth) {
            iterating = true;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 pxPerBlock = (numPx + blockSize1d - 1) / blockSize1d;
    finalGather << <pxPerBlock, blockSize1d >> > (total_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        numPx * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}
