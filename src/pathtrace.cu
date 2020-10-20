#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <gltf-loader.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define MATERIAL_SORT
#define CACHE_BOUNCE    // Determine whether cache the first bounce or do the stochastic sampling
// #define THIN_LENS_CAMERA

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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        /*
        glm::vec3 pos_color = 255.f * glm::abs(gBuffer[index].pos) / glm::l2Norm(gBuffer[index].pos);
        pbo[index].w = 0;
        pbo[index].x = pos_color.x;
        pbo[index].y = pos_color.y;
        pbo[index].z = pos_color.z;
        */
        
        glm::vec3 normal_color = glm::abs(gBuffer[index].normal * 255.0f);
        pbo[index].w = 0;
        pbo[index].x = normal_color.x;
        pbo[index].y = normal_color.y;
        pbo[index].z = normal_color.z;
        
        /*
        float timeToIntersect = gBuffer[index].t * 256.0;

        pbo[index].w = 0;
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;
        */
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection* dev_cache_intersections = NULL;
static PathSegment* dev_cache_paths = NULL;


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

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_cache_paths, pixelcount * sizeof(PathSegment));
    checkCUDAError("pathtraceInit");
}

__global__ void cpyCoordVals(float* tar, float* src, int offset, int src_length) {
    int src_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (src_index >= src_length) {
        return;
    }
    int tar_index = src_index / 3;
    if (src_index % 3 == offset) {
        tar[tar_index] = src[src_index];
    }
}


__global__ void warpUV(int uv_size, float* dev_uv) {
    int uv_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (uv_index >= uv_size) {
        return;
    }
    float uv_val = dev_uv[uv_index];
    if (uv_val > 1 || uv_val < 0) {
        dev_uv[uv_index] = uv_val - glm::floor(uv_val);
    }
}


void meshInit(Scene* scene) {
    
    for (int i = 0; i < scene->geoms.size(); ++i) {
        Geom& temp_geo_ref = scene->geoms[i];
        if (temp_geo_ref.type == GeomType::MESH) {
            cout << "Init mesh geo:" << endl;
            // string filename = "C:\\JiaruiYan\\MasterDegreeProjects\\CIS565\\Proj3\\tinygltf_test\\gltf_test\\scene\\Box.gltf";
            // string filename = "C:\\JiaruiYan\\MasterDegreeProjects\\CIS565\\Proj3\\Project3-CUDA-Path-Tracer\\read_models\\duck\\Duck.gltf";
            string filename = scene->mesh_filename;
            std::vector<example::Mesh<float> > meshes;
            std::vector<example::Material> materials;
            std::vector<example::Texture> textures;
            bool ret = LoadGLTF(filename, 1.0, &meshes, &materials, &textures);
            if (!ret) {
                std::cerr << "Failed to load [ " << filename << " ]" << std::endl;
            }

            example::Mesh<float>& model_ref = meshes[0];
            temp_geo_ref.indices_num = model_ref.faces.size();

            float *dev_x_coord, *dev_y_coord, *dev_z_coord;

            cudaMalloc(&temp_geo_ref.dev_mesh_positions, model_ref.vertices.size() * sizeof(float));
            cudaMemcpy(temp_geo_ref.dev_mesh_positions, model_ref.vertices.data(), model_ref.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);

            cudaMalloc(&dev_x_coord, model_ref.vertices.size() * sizeof(float) / 3);
            cudaMalloc(&dev_y_coord, model_ref.vertices.size() * sizeof(float) / 3);
            cudaMalloc(&dev_z_coord, model_ref.vertices.size() * sizeof(float) / 3);

            const int blockSize1d = 128;
            dim3 numblocksCpyCoord = (model_ref.vertices.size() + blockSize1d - 1) / blockSize1d;
            cpyCoordVals <<<numblocksCpyCoord, blockSize1d>>> (dev_x_coord, temp_geo_ref.dev_mesh_positions, 0, model_ref.vertices.size());
            cpyCoordVals <<<numblocksCpyCoord, blockSize1d>>> (dev_y_coord, temp_geo_ref.dev_mesh_positions, 1, model_ref.vertices.size());
            cpyCoordVals <<<numblocksCpyCoord, blockSize1d>>> (dev_z_coord, temp_geo_ref.dev_mesh_positions, 2, model_ref.vertices.size());

            float* dev_max_x = thrust::max_element(thrust::device, dev_x_coord, dev_x_coord + model_ref.vertices.size() / 3);
            float* dev_min_x = thrust::min_element(thrust::device, dev_x_coord, dev_x_coord + model_ref.vertices.size() / 3);
            float host_max_x, host_min_x;
            cudaMemcpy(&host_max_x, dev_max_x, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_min_x, dev_min_x, sizeof(float), cudaMemcpyDeviceToHost);

            float* dev_max_y = thrust::max_element(thrust::device, dev_y_coord, dev_y_coord + model_ref.vertices.size() / 3);
            float* dev_min_y = thrust::min_element(thrust::device, dev_y_coord, dev_y_coord + model_ref.vertices.size() / 3);
            float host_max_y, host_min_y;
            cudaMemcpy(&host_max_y, dev_max_y, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_min_y, dev_min_y, sizeof(float), cudaMemcpyDeviceToHost);

            float* dev_max_z = thrust::max_element(thrust::device, dev_z_coord, dev_z_coord + model_ref.vertices.size() / 3);
            float* dev_min_z = thrust::min_element(thrust::device, dev_z_coord, dev_z_coord + model_ref.vertices.size() / 3);
            float host_max_z, host_min_z;
            cudaMemcpy(&host_max_z, dev_max_z, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_min_z, dev_min_z, sizeof(float), cudaMemcpyDeviceToHost);

            cout << "upper corner:(" << host_max_x << ", " << host_max_y << ", " << host_max_z << ")" << endl;
            cout << "downward corner:(" << host_min_x << ", " << host_min_y << ", " << host_min_z << ")" << endl;

            temp_geo_ref.bounding_box_down_corner[0] = host_min_x - 0.1f;
            temp_geo_ref.bounding_box_down_corner[1] = host_min_y - 0.1f;
            temp_geo_ref.bounding_box_down_corner[2] = host_min_z - 0.1f;
            temp_geo_ref.bounding_box_upper_corner[0] = host_max_x + 0.1f;
            temp_geo_ref.bounding_box_upper_corner[1] = host_max_y + 0.1f;
            temp_geo_ref.bounding_box_upper_corner[2] = host_max_z + 0.1f;

            cudaMalloc(&temp_geo_ref.dev_mesh_indices, model_ref.faces.size() * sizeof(unsigned int));
            cudaMemcpy(temp_geo_ref.dev_mesh_indices, model_ref.faces.data(), model_ref.faces.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

            cudaMalloc(&temp_geo_ref.dev_mesh_normals, model_ref.facevarying_normals.size() * sizeof(float));
            cudaMemcpy(temp_geo_ref.dev_mesh_normals, model_ref.facevarying_normals.data(), model_ref.facevarying_normals.size() * sizeof(float), cudaMemcpyHostToDevice);


            /*
            cout << "uvs:" << endl;
            for (int i = 0; i < meshes[0].facevarying_uvs.size() / 2; i = i + 1) {
                cout << "uv[" << i << "]:(" << meshes[0].facevarying_uvs[i * 2] << ", " << meshes[0].facevarying_uvs[i * 2 + 1] << ")" << endl;
            }
            */

            if (textures.size() != 0) {
                // dim3 numblocksUVWarp = (model_ref.facevarying_uvs.size() + blockSize1d - 1) / blockSize1d;
                cudaMalloc(&temp_geo_ref.dev_uvs, model_ref.facevarying_uvs.size() * sizeof(float));
                cudaMemcpy(temp_geo_ref.dev_uvs, model_ref.facevarying_uvs.data(), model_ref.facevarying_uvs.size() * sizeof(float), cudaMemcpyHostToDevice);
                // warpUV <<<numblocksUVWarp, blockSize1d>>> (model_ref.facevarying_uvs.size(), temp_geo_ref.dev_uvs);

                /* NOTE: UV is facevarying instead of vertices varying.
                for (int i = 0; i < model_ref.faces.size() / 3; ++i) {
                    int idx1 = model_ref.faces[i * 3];
                    int idx2 = model_ref.faces[i * 3 + 1];
                    int idx3 = model_ref.faces[i * 3 + 2];
                    cout << "Triangle(" << idx1 << ", " << idx2 << ", " << idx3 << ")" << endl;
                    cout << "UV" << idx1 << ":(" << model_ref.facevarying_uvs[idx1 * 2] << ", " << model_ref.facevarying_uvs[idx1 * 2 + 1] << ")" << endl;
                    cout << "UV" << idx2 << ":(" << model_ref.facevarying_uvs[idx2 * 2] << ", " << model_ref.facevarying_uvs[idx2 * 2 + 1] << ")" << endl;
                    cout << "UV" << idx3 << ":(" << model_ref.facevarying_uvs[idx3 * 2] << ", " << model_ref.facevarying_uvs[idx3 * 2 + 1] << ")" << endl;
                    // cout << "UV:(" << model_ref.facevarying_uvs[i * 2] << ", " << model_ref.facevarying_uvs[i * 2 + 1] << ")" << endl;
                }
                */
                int image_size = textures[0].components * textures[0].width * textures[0].height * sizeof(unsigned char);
                cudaMalloc(&temp_geo_ref.dev_texture, image_size * sizeof(unsigned char));
                cudaMemcpy(temp_geo_ref.dev_texture, textures[0].image, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

                temp_geo_ref.hasTexture = true;
                temp_geo_ref.texture_width = textures[0].width;
                temp_geo_ref.texture_height = textures[0].height;
            }
            else {
                temp_geo_ref.hasTexture = false;
            }

            cudaFree(dev_x_coord);
            cudaFree(dev_y_coord);
            cudaFree(dev_z_coord);
        }
    }
    checkCUDAError("Mesh init");
}

void pathtraceFree(Scene* scene) {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_cache_intersections);
    cudaFree(dev_cache_paths);
    checkCUDAError("pathtraceFree");
}

void meshFree(Scene* scene) {
    for (int i = 0; scene != nullptr && i < scene->geoms.size(); ++i) {
        Geom& temp_geo_ref = scene->geoms[i];
        if (temp_geo_ref.type == GeomType::MESH) {
            if (temp_geo_ref.dev_mesh_indices != nullptr) {
                cudaFree(temp_geo_ref.dev_mesh_indices);
            }
            if (temp_geo_ref.dev_mesh_normals != nullptr) {
                cudaFree(temp_geo_ref.dev_mesh_normals);
            }
            if (temp_geo_ref.dev_mesh_positions != nullptr) {
                cudaFree(temp_geo_ref.dev_mesh_positions);
            }
            if (temp_geo_ref.hasTexture) {
                if (temp_geo_ref.dev_uvs != nullptr) {
                    cudaFree(temp_geo_ref.dev_uvs);
                }
                if (temp_geo_ref.dev_texture != nullptr) {
                    cudaFree(temp_geo_ref.dev_texture);
                }
            }
        }
    }
    checkCUDAError("GeoFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/

__device__ glm::vec2 ConcentricSampleDisk(const glm::vec2 u) {
    // Map uniform random numbers to [-1, 1]
    glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);
    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) {
        return glm::vec2(0.f);
    }
    // Apply concentric mapping to point
    float theta, r;
    if (fabs(uOffset.x) > fabs(uOffset.y)) {
        r = uOffset.x;
        theta = 0.25 * 3.1415926 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = 0.5 * 3.1415926 - 0.25 * 3.1415926 * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(cosf(theta), sinf(theta));
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
#ifdef CACHE_BOUNCE
        // If we choose to cache the first bounce, then it would not be jittered.
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#else
        // We will jitter rays if there is not first bounce cache.
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng) - 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng) - 0.5f)
        );
#endif // CACHE_BOUNCE

        bool thin_len_cam = false;

#ifdef THIN_LENS_CAMERA
        thin_len_cam = true;
#endif // THIN_LENS_CAMERA

        float lensRadius = 0.1f;
        float focalDistance = cam.focal_length;
        if (thin_len_cam) {
            // Sample point on lens
            glm::vec2 pLens = lensRadius * ConcentricSampleDisk(glm::vec2(u01(rng), u01(rng)));
            // Compute point on plane of focus
            float ft = focalDistance / glm::dot(segment.ray.direction, cam.view);
            glm::vec3 pFocus = ft * segment.ray.direction + segment.ray.origin;
            // Update ray for effect of lens
            segment.ray.origin = cam.position + cam.up * pLens[1] + cam.right * pLens[0];
            // segment.ray.origin = cam.position;
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
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
        pathSegments[path_index].ori_id = path_index;

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

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
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_uv, tmp_intersect, tmp_normal, outside);
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
            intersections[path_index].surfaceNormal = glm::vec3(0.f, 0.f, 0.f);
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = outside;
            intersections[path_index].hit_type = geoms[hit_geom_index].type;
            intersections[path_index].hasTexture = geoms[hit_geom_index].hasTexture;
            intersections[path_index].geomId = hit_geom_index;
            
            if (geoms[hit_geom_index].type == MESH && geoms[hit_geom_index].hasTexture) {
                intersections[path_index].uv = tmp_uv;
            }
            

            if (abs(normal.x) > abs(normal.y)) {
                intersections[path_index].tangent = glm::vec3(-normal.z, 0.f, normal.x) / sqrt(normal.x * normal.x + normal.z * normal.z);
            }
            else {
                intersections[path_index].tangent = glm::vec3(0.f, normal.z, -normal.y) / sqrt(normal.y * normal.y + normal.z * normal.z);
            }
            intersections[path_index].bitangent = glm::cross(normal, intersections[path_index].tangent);
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
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}


__device__ int directlight_shadowtest(Ray tempRay, Geom* geoms, int geoms_size) {

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_uv;

    // naive parse through global geoms
    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, tempRay, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, tempRay, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == MESH)
        {
            t = meshIntersectionTest(geom, tempRay, tmp_uv, tmp_intersect, tmp_normal, outside);
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
    return hit_geom_index;
}

__global__ void directlight_shade_bounce(int iter, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Geom* geoms
    , int geoms_size, Material* materials, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        // int ori_idx = pathSegments[idx].ori_id;
        // ShadeableIntersection intersection = shadeableIntersections[ori_idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
            // If the intersection exists
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            // thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
                image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
            }
            else {
                // direct shading:
                const ShadeableIntersection temp_intersect = shadeableIntersections[idx];
                glm::vec3 intersect_pos = temp_intersect.t * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
                thrust::uniform_real_distribution<float> u01(0, 1);
                
                // Randomly select a light source
                // Max 10 lights in the scene for direct light;
                int light_num = 0;
                int light_idxs[10];
                for (int i = 0; i < geoms_size; ++i) {
                    Geom& temp_geo_ref = geoms[i];
                    Material& temp_mat_ref = materials[temp_geo_ref.materialid];
                    if (temp_mat_ref.emittance > 0.f) {
                        // This is a light
                        light_idxs[light_num] = i;
                        ++light_num;
                    }
                }
                int random_light_idx = glm::min((int)(u01(rng) * light_num), light_num - 1);
                Geom& light = geoms[light_idxs[random_light_idx]];
                
                // Get an intersection on the surface of its shape
                glm::vec4 pObj(u01(rng) - 0.5f, -0.5f, u01(rng) - 0.5f, 1.f);
                float area = light.scale.x * light.scale.z;
                
                glm::vec4 local_normal(0.f, -1.f, 0.f, 0.f);
                glm::vec3 light_intersect = multiplyMV(light.transform, pObj);
                glm::vec3 light_normal = glm::normalize(multiplyMV(light.invTranspose, local_normal));
                float pdf = glm::length2(light_intersect - intersect_pos) / area;

                // Check if the resultant PDF is zero and the 
                // resultant Intersection are the same point in space, and return black if this is the case.
                if (pdf == 0.f || glm::l2Norm(light_intersect - intersect_pos) < FLT_EPSILON) {
                    pathSegments[idx].color = glm::vec3(0.f);
                }
                else {
                    // Set ωi to the normalized vector from the reference Intersection's
                    // point to the Shape's intersection point.
                    glm::vec3 wi = glm::normalize(light_intersect - intersect_pos);

                    // Return the light emitted along ωi from our intersection point.
                    Material& light_material = materials[light.materialid];
                    
                    glm::vec3 light_L = glm::dot(light_normal, -wi) > 0.f ? (light_material.color * light_material.emittance) : glm::vec3(0.f);
                    // printf("light_normal:(%f, %f, %f)\nwi:(%f, %f, %f)\n\n", light_normal.x, light_normal.y, light_normal.z, wi.x, wi.y, wi.z);
                    /*
                    if (glm::l2Norm(light_L) != 0.f) {
                        printf("light_L != 0\n");
                    }
                    */

                    // Shadow test
                    Ray tempRay;
                    tempRay.direction = wi;
                    tempRay.origin = intersect_pos + 0.01f * temp_intersect.surfaceNormal;
                    int hit_geom_index = directlight_shadowtest(tempRay, geoms, geoms_size);
                    if (hit_geom_index == -1 || hit_geom_index != light_idxs[random_light_idx]) {
                        pathSegments[idx].color = glm::vec3(0.f);
                    }
                    else {
                        // Evaluate the remaining component of the LTE
                        // Texture color:
                        
                        if (temp_intersect.hit_type == MESH && temp_intersect.hasTexture) {
                            Geom& temp_geo_ref = geoms[temp_intersect.geomId];
                            float temp_u = temp_intersect.uv[0];
                            float temp_v = temp_intersect.uv[1];
                            /*
                            if (temp_intersect.uv[0] > 1 || temp_intersect.uv[0] < 0) {
                                temp_u = temp_intersect.uv[0] - glm::floor(temp_intersect.uv[0]);
                                if (temp_u == 0) {
                                    
                                }
                            }
                            if (temp_intersect.uv[1] > 1 || temp_intersect.uv[1] < 0) {
                                temp_v = temp_intersect.uv[1] - glm::floor(temp_intersect.uv[1]);
                            }
                            */

                            int coordU = (int)(temp_u * (temp_geo_ref.texture_width));
                            int coordV = (int)(temp_v * (temp_geo_ref.texture_height));
                            
                            if (coordU >= 512) {
                                printf("coordU >= 512: %d\n", coordU);
                                coordU %= 512;
                            }
                            if (coordV >= 512) {
                                printf("coordV >= 512: %d\n", coordV);
                                coordV %= 512;
                            }
                            
                            int pixel_idx = coordV * temp_geo_ref.texture_width + coordU;
                            // int pixel_idx = coordU * temp_geo_ref.texture_width + coordV;
                            unsigned int colR = (unsigned int) temp_geo_ref.dev_texture[pixel_idx * 4];
                            unsigned int colG = (unsigned int) temp_geo_ref.dev_texture[pixel_idx * 4 + 1];
                            unsigned int colB = (unsigned int) temp_geo_ref.dev_texture[pixel_idx * 4 + 2];
                            materialColor[0] = (float)colR / 255.f;
                            materialColor[1] = (float)colG / 255.f;
                            materialColor[2] = (float)colB / 255.f;
                            // printf("UV:(%f, %f)\n", temp_intersect.uv[0], temp_intersect.uv[1]);
                            // printf("UVCoord:(%d, %d)\n", coordU, coordV);
                            /*
                            if (colR != 225 || colG != 191 || colB != 0) {
                                printf("(%d, %d, %d)\n", colR, colG, colB);
                            }*/
                            // printf("(%d, %d)\n", temp_geo_ref.texture_width, temp_geo_ref.texture_height);
                        }
                        

                        glm::vec3 f = materialColor / PI;
                        pathSegments[idx].color = f * light_L * glm::abs(glm::dot(wi, temp_intersect.surfaceNormal)) / (pdf / light_num);
                        pathSegments[idx].remainingBounces = 0;
                        image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
                        // if (glm::l2Norm(pathSegments[idx].color) != 0.f) {
                            // printf("light_num:%d\n", light_num);
                            // printf("pathSegments[idx].color:(%f, %f, %f)\n", pathSegments[idx].color.x, pathSegments[idx].color.y, pathSegments[idx].color.z);
                        // }
                        // printf("temp_intersect.surfaceNormal:(%f, %f, %f)\nwi:(%f, %f, %f)\n\n", temp_intersect.surfaceNormal.x, temp_intersect.surfaceNormal.y, temp_intersect.surfaceNormal.z, wi.x, wi.y, wi.z);
                        // printf("light_L:(%f, %f, %f)\n", light_L.x, light_L.y, light_L.z);
                        // printf("pathSegments[idx].color:(%f, %f, %f)\n", pathSegments[idx].color.x, pathSegments[idx].color.y, pathSegments[idx].color.z);
                    }
                }
            }
        }
        else {
            // If there was no intersection, color the ray black.
            pathSegments[idx].color = glm::vec3(0.f);
            pathSegments[idx].remainingBounces = 0;
            image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
        }
    }
}


__global__ void shade_image(int num_paths, PathSegment* pathSegments, Geom* geoms, int geoms_size, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        int mesh_idx = 0;
        for (int i = 0; i < geoms_size; ++i) {
            if (geoms[i].hasTexture) {
                mesh_idx = i;
            }
        }

        unsigned int colR = (unsigned int) geoms[mesh_idx].dev_texture[pathSegments[idx].pixelIndex * 4];
        unsigned int colG = (unsigned int) geoms[mesh_idx].dev_texture[pathSegments[idx].pixelIndex * 4 + 1];
        unsigned int colB = (unsigned int) geoms[mesh_idx].dev_texture[pathSegments[idx].pixelIndex * 4 + 2];
        
        image[pathSegments[idx].pixelIndex] += glm::vec3((float)colR / 255.f, (float)colG / 255.f, (float)colB / 255.f);
        
    }
}

__global__ void shade_bounce(int iter, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Geom* geoms, Material* materials, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        // int ori_idx = pathSegments[idx].ori_id;
        // ShadeableIntersection intersection = shadeableIntersections[ori_idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
            // If the intersection exists
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            // thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
                image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
            }
            else {
                // BSDF accumulate:
                
                // const ShadeableIntersection temp_intersect = shadeableIntersections[ori_idx];
                const ShadeableIntersection temp_intersect = shadeableIntersections[idx];
                glm::vec3 intersect_pos = temp_intersect.t * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
                scatterRay(pathSegments[idx], intersect_pos, temp_intersect.surfaceNormal, temp_intersect.outside, temp_intersect, geoms, material, rng);
            }
        }
        else {
            // If there was no intersection, color the ray black.
            pathSegments[idx].color = glm::vec3(0.f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}


__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        gBuffer[idx].t = shadeableIntersections[idx].t;
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
        gBuffer[idx].pos = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
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

struct zero_bounce
{
    __host__ __device__
    bool operator()(const PathSegment x)
    {
        return x.remainingBounces == 0;
    }
};

struct mat_sort
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2) {
        return i1.materialId > i2.materialId;
    }
};


__global__ void print_remain_bounces(int nPaths, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths) {
        if (index == 0) {
            printf("remainbounces:%d\n", iterationPaths[index].remainingBounces);
        }
    }
}




/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
// void pathtrace(uchar4 *pbo, int frame, int iter) {
void pathtrace(int frame, int iter) {
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
    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    bool iterationComplete = false;

    // std::cout << "iter:" << iter << std::endl;

    // TODO: perform one iteration of path tracing
#ifdef CACHE_BOUNCE
    if (iter == 1) {
        cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
        generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
        checkCUDAError("generate camera ray");
    }
    else {
        // Generate ray from cached first intersection
        cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_paths, dev_cache_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        dim3 numblocksPathSegmentTracing = (pixelcount + blockSize1d - 1) / blockSize1d;
        // print_remain_bounces << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_cache_paths);
        shade_bounce << <numblocksPathSegmentTracing, blockSize1d >> > (iter, pixelcount, dev_intersections, dev_paths, dev_geoms, dev_materials, dev_image);
        // print_remain_bounces << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_paths);
        PathSegment* new_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + pixelcount, zero_bounce());
        if (new_end == dev_paths) {
            iterationComplete = true;
        }
        else {
            num_paths = new_end - dev_paths;
        }
        depth++;
    }
#else
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
#endif // CACHE_BOUNCE
	
    

    // std::cout << "traceDepth:" << traceDepth << std::endl;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	while (!iterationComplete) {
        // std::cout << "Depth:" << depth << std::endl;
        // std::cout << "Num of path:" << num_paths << std::endl << std::endl;
	    // clean shading chunks
	    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        
        
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
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


        if (depth == 0) {
            generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
            // std::cout << "Generate G Buffer" << std::endl;
        }
        
        depth++;
        
#ifdef MATERIAL_SORT
        // TODO: Sort rays by material
        thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, mat_sort());
#endif // 

        
#ifdef CACHE_BOUNCE
        // Cache the first intersection
        if (iter == 1 && depth == 1) {
            cudaMemcpy(dev_cache_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_cache_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
            // print_remain_bounces << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_paths);
        }
#endif // CACHE_BOUNCE

	    // TODO:
	    // --- Shading Stage ---
	    // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        shade_bounce <<<numblocksPathSegmentTracing, blockSize1d>>> (iter, num_paths, dev_intersections, dev_paths, dev_geoms, dev_materials, dev_image);

        // TODO: should be based off stream compaction results.
        // Stream compact away all of the terminated paths.
        PathSegment* new_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, zero_bounce());
        // std::cout << "num_path before remove:" << num_paths << std::endl;
        // print_remain_bounces << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths);
        if (new_end == dev_paths) {
            iterationComplete = true;
        }
        else {
            num_paths = new_end - dev_paths;
        }
        // std::cout << "num_path after remove:" << num_paths << std::endl;
        // iterationComplete = true; 
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    // sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}

void directlight_pathtrace(uchar4* pbo, int frame, int iter) {
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

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    bool iterationComplete = false;

    // TODO: perform one iteration of path tracing
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // std::cout << "traceDepth:" << traceDepth << std::endl;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    while (!iterationComplete) {
        // std::cout << "Num of path:" << num_paths << std::endl;
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections <<<numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#ifdef MATERIAL_SORT
        // TODO: Sort rays by material
        thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, mat_sort());
#endif // 

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        
        directlight_shade_bounce <<<numblocksPathSegmentTracing, blockSize1d>>> (
            iter, 
            num_paths, 
            dev_intersections, 
            dev_paths, dev_geoms, 
            hst_scene->geoms.size(), 
            dev_materials, 
            dev_image);
        

        // dim3 numblocksImageShade = (512 * 512 + blockSize1d - 1) / blockSize1d;
        // shade_image <<<numblocksImageShade, blockSize1d>>> (num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_image);

        // TODO: should be based off stream compaction results.
        // Stream compact away all of the terminated paths.
        PathSegment* new_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, zero_bounce());
        // std::cout << "num_path before remove:" << num_paths << std::endl;
        // print_remain_bounces << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths);
        if (new_end == dev_paths) {
            iterationComplete = true;
        }
        else {
            num_paths = new_end - dev_paths;
        }
        // std::cout << "num_path after remove:" << num_paths << std::endl;
        iterationComplete = true; 
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("direct light pathtrace");

}

__global__ void denoise_iteration(glm::ivec2 resolution, int stepwidth, float c_phi, float n_phi, float p_phi, float* i_blur_kernel, glm::vec3* i_image, glm::vec3* o_image, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 cval = i_image[index];
        glm::vec3 nval = gBuffer[index].normal;
        glm::vec3 pval = gBuffer[index].pos;
        glm::vec3 sum(0.f, 0.f, 0.f);

        // printf("c_phi:%f, n_phi:%f, p_phi:%f\n", c_phi, n_phi, p_phi);

        float cum_w = 0.f;
        for (int offset_x = -2; offset_x <= 2; ++offset_x) {
            for (int offset_y = -2; offset_y <= 2; ++offset_y) {
                int i = (offset_x + 2) + 5 * (offset_y + 2);
                int tmp_x = glm::clamp(x + offset_x * stepwidth, 0, resolution.x - 1);
                int tmp_y = glm::clamp(y + offset_y * stepwidth, 0, resolution.y - 1);
                int tmp_idx = tmp_x + (tmp_y * resolution.x);

                glm::vec3 ctmp = i_image[tmp_idx];
                glm::vec3 t = cval - ctmp;

                float dist2 = glm::dot(t, t);
                float c_w = min(expf(-(dist2) / c_phi), 1.f);

                glm::vec3 ntmp = gBuffer[tmp_idx].normal;
                t = nval - ntmp;
                dist2 = max(glm::dot(t, t) / (stepwidth * stepwidth), 0.f);
                float n_w = min(expf(-(dist2) / n_phi), 1.f);

                glm::vec3 ptmp = gBuffer[tmp_idx].pos;
                t = pval - ptmp;
                dist2 = glm::dot(t, t);
                float p_w = min(expf(-(dist2) / p_phi), 1.f);

                float weight = c_w * n_w * p_w;
                // printf("weight:%f\n", weight);
                // weight = c_phi;
                // weight = 1.f;
                sum += ctmp * weight * i_blur_kernel[i];
                cum_w += weight * i_blur_kernel[i];
            }
        }
        o_image[index] = sum / cum_w;
    }
}

__global__ void make_image(glm::vec3* i_image, glm::vec3* o_image, glm::ivec2 resolution, int iter) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = i_image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        o_image[index].x = color.x;
        o_image[index].y = color.y;
        o_image[index].z = color.z;
    }
}

__global__ void denoiser_send_image_to_pbo(uchar4* pbo, glm::ivec2 resolution, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        pbo[index].w = 0;
        pbo[index].x = image[index].x;
        pbo[index].y = image[index].y;
        pbo[index].z = image[index].z;
    }
}


__global__ void getNorm(GBufferPixel* gBuffer, glm::vec3* image, float* colNorms, float* posNorms, float* norNorms, int pixelNum) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < pixelNum) {
        colNorms[index] = glm::l2Norm(image[index]);
        posNorms[index] = glm::l2Norm(gBuffer[index].pos);
        norNorms[index] = glm::l2Norm(glm::normalize(gBuffer[index].normal));
        // printf("norNorms:%f\n", norNorms[index]);
    }
}

__global__ void getVariances(float* colNorms, float* posNorms, float* norNorms, float colAvg, float norAvg, float posAvg, int pixelNum) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < pixelNum) {
        colNorms[index] = (colNorms[index] - colAvg) * (colNorms[index] - colAvg);
        posNorms[index] = (posNorms[index] - posAvg) * (posNorms[index] - posAvg);
        norNorms[index] = (norNorms[index] - norAvg) * (norNorms[index] - norAvg);
    }
}


void denoise(uchar4* pbo, int iter, int filter_size) {
    float blur_kernel[25];
    float coefficient_mask[5] = {1.f/16.f, 1.f/4.f, 3.f/8.f, 1.f/4.f, 1.f/16.f};
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 5; ++col) {
            int index = row * 5 + col;
            blur_kernel[index] = coefficient_mask[row] * coefficient_mask[col];
        }
    }

    const Camera& cam = hst_scene->state.camera;
    

    float* dev_blur_kernel;
    glm::vec3* dev_tmp_image;
    glm::vec3* dev_tmp_output_image;
    float* dev_colNorm, * dev_posNorm, * dev_norNorm;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const int blockSize1d = 128;
    dim3 numblocksPerGrid1d = (pixelcount + blockSize1d - 1) / blockSize1d;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    cudaMalloc(&dev_blur_kernel, 25 * sizeof(float));
    cudaMemcpy(dev_blur_kernel, blur_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tmp_image, pixelcount * sizeof(glm::vec3));
    cudaMemcpy(dev_tmp_image, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

    cudaMalloc(&dev_tmp_output_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_tmp_output_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_colNorm, pixelcount * sizeof(float));
    cudaMemset(dev_colNorm, 0, pixelcount * sizeof(float));

    cudaMalloc(&dev_posNorm, pixelcount * sizeof(float));
    cudaMemset(dev_posNorm, 0, pixelcount * sizeof(float));

    cudaMalloc(&dev_norNorm, pixelcount * sizeof(float));
    cudaMemset(dev_norNorm, 0, pixelcount * sizeof(float));

    checkCUDAError("denoise init variables");
    /*
    for (int i = 0; i < 25; ++i) {
        if (i % 5 == 0) {
            std::cout << std::endl;
        }
        std::cout << blur_kernel[i] << " ";
    }
    */
    
    // make_image <<<blocksPerGrid2d, blockSize2d>>> (dev_image, dev_tmp_image, cam.resolution, iter);
    // checkCUDAError("denoise make image");
    
    // Calculate variances:
    getNorm <<<numblocksPerGrid1d, blockSize1d>>> (dev_gBuffer, dev_tmp_image, dev_colNorm, dev_posNorm, dev_norNorm, pixelcount);
    cudaDeviceSynchronize();
    float colAvg = thrust::reduce(thrust::device, dev_colNorm, dev_colNorm + pixelcount) / (float)pixelcount;
    float norAvg = thrust::reduce(thrust::device, dev_norNorm, dev_norNorm + pixelcount) / (float)pixelcount;
    float posAvg = thrust::reduce(thrust::device, dev_posNorm, dev_posNorm + pixelcount) / (float)pixelcount;
    checkCUDAError("Calculate Norm");

    getVariances <<<numblocksPerGrid1d, blockSize1d>>> (dev_colNorm, dev_posNorm, dev_norNorm, colAvg, norAvg, posAvg, pixelcount);
    float colVar = thrust::reduce(thrust::device, dev_colNorm, dev_colNorm + pixelcount) / (float)pixelcount;
    float norVar = thrust::reduce(thrust::device, dev_norNorm, dev_norNorm + pixelcount) / (float)pixelcount;
    float posVar = thrust::reduce(thrust::device, dev_posNorm, dev_posNorm + pixelcount) / (float)pixelcount;
    checkCUDAError("Calculate Variance");
    
    // std::cout << "colVar:" << colVar << " norVar:" << norVar << " posVar:" << posVar << std::endl;

    for (int i = 0; i < filter_size; ++i) {
        int decrease_coefficient = 1 << i;
        // std::cout << decrease_coefficient << std::endl;
        colVar /= (float)decrease_coefficient;
        norVar /= (float)decrease_coefficient;
        posVar /= (float)decrease_coefficient;
        int step_padding = decrease_coefficient - 1;
        denoise_iteration << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, 1 + step_padding, colVar, norVar, posVar, dev_blur_kernel, dev_tmp_image, dev_tmp_output_image, dev_gBuffer);
        cudaDeviceSynchronize();
        checkCUDAError("denoise iteration");
        cudaMemcpy(dev_tmp_image, dev_tmp_output_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }
    checkCUDAError("denoise filter");

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_tmp_output_image);
    // denoiser_send_image_to_pbo << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_tmp_image);
    checkCUDAError("denoise pbo");

    cudaFree(dev_blur_kernel);
    cudaFree(dev_tmp_image);
    cudaFree(dev_tmp_output_image);
    cudaFree(dev_colNorm);
    cudaFree(dev_norNorm);
    cudaFree(dev_posNorm);
    checkCUDAError("denoise CUDA Free");
    // std::cout << std::endl;
}

void FilterCreation(float* blur_kernel, int kernel_dim) {
    int kernel_size = kernel_dim * kernel_dim;
    // intialising standard deviation to 1.0 
    float sigma = 8.f;
    float r, s = 2.f * sigma * sigma;
    float M_PI = 3.1415926f;
    float sum = 0.f;
    int kernel_start = -kernel_dim / 2;
    int kernel_end = kernel_dim / 2;
    for (int x = kernel_start; x <= kernel_end; ++x) {
        for (int y = kernel_start; y <= kernel_end; ++y) {
            r = sqrt(x * x + y * y);
            int idx = (x + kernel_end) * kernel_dim + (y + kernel_end);
            blur_kernel[idx] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += blur_kernel[idx];
        }
    }
    // normalize the kernel
    for (int i = 0; i < kernel_size; ++i) {
        blur_kernel[i] /= sum;
    }
}

__global__ void gaussian_blur_execute(glm::ivec2 resolution, int kernel_dim, float c_phi, float n_phi, float p_phi, float* i_blur_kernel, glm::vec3* i_image, glm::vec3* o_image, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    // printf("kernel_dim:%d\n", kernel_dim);

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        int kernel_start = -kernel_dim / 2;
        int kernel_end = kernel_dim / 2;
        int kernel_size = kernel_dim * kernel_dim;

        glm::vec3 cval = i_image[index];
        glm::vec3 nval = gBuffer[index].normal;
        glm::vec3 pval = gBuffer[index].pos;
        glm::vec3 sum(0.f, 0.f, 0.f);

        // printf("c_phi:%f, n_phi:%f, p_phi:%f\n", c_phi, n_phi, p_phi);

        float cum_w = 0.f;
        for (int offset_x = kernel_start; offset_x <= kernel_end; ++offset_x) {
            for (int offset_y = kernel_start; offset_y <= kernel_end; ++offset_y) {
                int i = (offset_x + kernel_end) + kernel_dim * (offset_y + kernel_end);
                int tmp_x = glm::clamp(x + offset_x, 0, resolution.x - 1);
                int tmp_y = glm::clamp(y + offset_y, 0, resolution.y - 1);
                int tmp_idx = tmp_x + (tmp_y * resolution.x);

                glm::vec3 ctmp = i_image[tmp_idx];
                glm::vec3 t = cval - ctmp;

                float dist2 = glm::dot(t, t);
                float c_w = min(expf(-(dist2) / c_phi), 1.f);

                glm::vec3 ntmp = gBuffer[tmp_idx].normal;
                t = nval - ntmp;
                dist2 = max(glm::dot(t, t), 0.f);
                float n_w = min(expf(-(dist2) / n_phi), 1.f);

                glm::vec3 ptmp = gBuffer[tmp_idx].pos;
                t = pval - ptmp;
                dist2 = glm::dot(t, t);
                float p_w = min(expf(-(dist2) / p_phi), 1.f);

                float weight = c_w * n_w * p_w;
                // printf("weight:%f\n", weight);
                // weight = c_phi;
                // weight = 1.f;
                sum += ctmp * weight * i_blur_kernel[i];
                cum_w += weight * i_blur_kernel[i];
            }
        }
        // o_image[index] = sum;
        o_image[index] = sum / cum_w;
        // o_image[index] = i_image[index];
    }
}

void gauss_denoise(uchar4* pbo, int iter, int filter_size) {
    int max_padding_pixels = 0;
    if (filter_size > 1) {
        max_padding_pixels = 1 << (filter_size - 1);
        max_padding_pixels -= 1;
    }
    
    int filter_dim = 5 + 4 * max_padding_pixels;
    int kernel_size = filter_dim * filter_dim;
    float* blur_kernel = new float[kernel_size];
    FilterCreation(blur_kernel, filter_dim);

    /*
    for (int i = 0; i < kernel_size; ++i) {
        if (i % filter_dim == 0) {
            std::cout << std::endl;
        }
        std::cout << blur_kernel[i] << " ";
    }
    std::cout << std::endl;
    */
    const Camera& cam = hst_scene->state.camera;

    float* dev_blur_kernel;
    glm::vec3* dev_tmp_image;
    glm::vec3* dev_tmp_output_image;
    float* dev_colNorm, * dev_posNorm, * dev_norNorm;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // Init cuda relevant grid sizes and block sizes
    const int blockSize1d = 128;
    dim3 numblocksPerGrid1d = (pixelcount + blockSize1d - 1) / blockSize1d;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Allocate cuda memory
    cudaMalloc(&dev_blur_kernel, kernel_size * sizeof(float));
    cudaMemcpy(dev_blur_kernel, blur_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tmp_image, pixelcount * sizeof(glm::vec3));
    cudaMemcpy(dev_tmp_image, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

    cudaMalloc(&dev_tmp_output_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_tmp_output_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_colNorm, pixelcount * sizeof(float));
    cudaMemset(dev_colNorm, 0, pixelcount * sizeof(float));

    cudaMalloc(&dev_posNorm, pixelcount * sizeof(float));
    cudaMemset(dev_posNorm, 0, pixelcount * sizeof(float));

    cudaMalloc(&dev_norNorm, pixelcount * sizeof(float));
    cudaMemset(dev_norNorm, 0, pixelcount * sizeof(float));

    checkCUDAError("denoise init variables");

    // Calculate variances:
    getNorm << <numblocksPerGrid1d, blockSize1d >> > (dev_gBuffer, dev_tmp_image, dev_colNorm, dev_posNorm, dev_norNorm, pixelcount);
    cudaDeviceSynchronize();
    float colAvg = thrust::reduce(thrust::device, dev_colNorm, dev_colNorm + pixelcount) / (float)pixelcount;
    float norAvg = thrust::reduce(thrust::device, dev_norNorm, dev_norNorm + pixelcount) / (float)pixelcount;
    float posAvg = thrust::reduce(thrust::device, dev_posNorm, dev_posNorm + pixelcount) / (float)pixelcount;
    checkCUDAError("Calculate Norm");

    getVariances << <numblocksPerGrid1d, blockSize1d >> > (dev_colNorm, dev_posNorm, dev_norNorm, colAvg, norAvg, posAvg, pixelcount);
    cudaDeviceSynchronize();
    float colVar = thrust::reduce(thrust::device, dev_colNorm, dev_colNorm + pixelcount) / (float)pixelcount;
    float norVar = thrust::reduce(thrust::device, dev_norNorm, dev_norNorm + pixelcount) / (float)pixelcount;
    float posVar = thrust::reduce(thrust::device, dev_posNorm, dev_posNorm + pixelcount) / (float)pixelcount;
    checkCUDAError("Calculate Variance");

    // Guassian Blur:
    gaussian_blur_execute <<<blocksPerGrid2d, blockSize2d >>> (cam.resolution, filter_dim, colVar, norVar, posVar, dev_blur_kernel, dev_tmp_image, dev_tmp_output_image, dev_gBuffer);
    cudaDeviceSynchronize();
    checkCUDAError("gaussian denoise iteration");
    cudaMemcpy(dev_tmp_image, dev_tmp_output_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_tmp_output_image);
    // denoiser_send_image_to_pbo << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_tmp_image);
    checkCUDAError("denoise pbo");

    cudaFree(dev_blur_kernel);
    cudaFree(dev_tmp_image);
    cudaFree(dev_tmp_output_image);
    cudaFree(dev_colNorm);
    cudaFree(dev_norNorm);
    cudaFree(dev_posNorm);
    checkCUDAError("denoise CUDA Free");
    free(blur_kernel);
}
