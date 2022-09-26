#pragma once
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "intersections.h"
#include <glm/gtc/epsilon.hpp>
#include "cfg.h"
#include "utility"
#include <glm/gtx/norm.hpp>
#include "microface.h"
#include "cudaUtils.cuh"
#include <stdlib.h>
#include "bvh.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */

__host__ __device__
void StratifiedSample1D(
    thrust::default_random_engine& rng, 
    float* samp, 
    const int& nSamples,
    bool jitter = true) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    float invNsamples = 1.0f / (nSamples );
    
    for (int i = 0; i < nSamples; i++) {
        float delta = jitter ? u01(rng) : 0.5f;
        samp[i] = min( (i + delta) * invNsamples, (float)0x1.fffffep-1);
    }
}


__host__ __device__
void StratifiedSample2D(
    thrust::default_random_engine& rng,
    glm::vec2* samp,
    const int& nSamples,
    bool jitter = true) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float d = 1.0 / nSamples;

    for (int i = 0; i < nSamples; i++) {
        float jx = jitter ? u01(rng) : 0.5f;
        float jy = jitter ? u01(rng) : 0.5f;
        samp[i].x = min((i + jx) * d, 1.0f - EPSILON);
        samp[i].y = min((i + jy) * d, 1.0f - EPSILON);
    }
}

// cosine weighted
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    
    
    
#if stratified_sampling
    const int nsamples = 256;
    thrust::uniform_real_distribution<float> u01(0, nsamples);
    
    glm::vec2 samp[nsamples];
    StratifiedSample2D(rng, samp, nsamples, true);
    int idx = int(u01(rng));
    float up = sqrt(samp[idx].x); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    idx = int(u01(rng));
    float around = samp[idx].y * TWO_PI;
#else
    thrust::uniform_real_distribution<float> u01(0, 1);
    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;
#endif // stratified_sampling
    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1;
    glm::vec3 perpendicularDirection2;
    CoordinateSys(normal, perpendicularDirection1, perpendicularDirection2);

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
vc3 perfectSpecularReflection(
    vc3& wiw,
    const vc3& wow,
    Float& pdf,
    const ShadeableIntersection& itsct,
    const Material& m,
    glm::vec3* textureArray,
    thrust::default_random_engine& rng) {

    wiw = glm::reflect(-wow, itsct.vtx.normal);
    vc3 f = m.specular.color / glm::abs(glm::dot(wiw, itsct.vtx.normal));
    if (m.specularTexture.valid == 1) {
        f *= sampleTexture(textureArray, itsct.vtx.uv, m.specularTexture);
    }
    pdf = 1.;
    return f;
}

__host__ __device__
vc3 imperfectSpecularReflection(
    vc3& wiw,
    const vc3& wow,
    Float& pdf,
    const ShadeableIntersection& itsct,
    const Material& m,
    vc3* textureArray,
    thrust::default_random_engine& rng) {
    
    
    const float n = m.specular.exponent;
    if (n >= 4096.0f) {
        return perfectSpecularReflection(wiw, wow, pdf, itsct, m, textureArray, rng);
    }
    else {
        thrust::uniform_real_distribution<float> u01(0, 1);
        // ref : https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling

        float mu_1 = u01(rng); // random number
        float mu_2 = u01(rng);
        float theta_s = acos( powf(u01(rng), 1.0 / (n + 1)) );
        float phi_s = 2.0 * PI * mu_2;

        float cos_phi_s = cos(phi_s);
        float sin_phi_s = sin(phi_s);
        float cos_theta_s = cos(theta_s);
        float sin_theta_s = sin(theta_s);

        glm::vec3 local_dir(
            cos_phi_s * sin_theta_s,
            sin_phi_s * sin_theta_s,
            cos_theta_s);

        // ref :: // https://stackoverflow.com/questions/20923232/how-to-rotate-a-vector-by-a-given-direction
        glm::vec3 perfect_reflect_dir = glm::reflect(-wow, itsct.vtx.normal);
        glm::vec3 base_z = glm::normalize(perfect_reflect_dir);
        glm::vec3 base_x = glm::normalize(glm::cross(base_z, glm::vec3(0.0f, 0.0f, 1.0f)));
        glm::vec3 base_y = glm::normalize(glm::cross(base_z, base_x));
        glm::mat3 base_m = glm::mat3(base_x, base_y, base_z);

        wiw = base_m * local_dir;
        vc3 f = m.specular.color / glm::abs(glm::dot(wiw, itsct.vtx.normal));
        if (m.specularTexture.valid == 1) {
            f *= sampleTexture(textureArray, itsct.vtx.uv, m.specularTexture);
        }
        pdf = 1.;
        return f;
    }
}

__host__ __device__
vc3 LambertBRDF(
    vc3& wiw,
    Float& pdf,
    const ShadeableIntersection& itsct,
    const Material& m,
    glm::vec3* textureArray,
    thrust::default_random_engine& rng) {
    vc3 f = m.color / glm::pi<Float>();
    wiw = calculateRandomDirectionInHemisphere(itsct.vtx.normal, rng);
    if (m.diffuseTexture.valid == 1) {
        f *= sampleTexture(textureArray, itsct.vtx.uv, m.diffuseTexture);
    }
    pdf = glm::dot(itsct.vtx.normal, wiw) / glm::pi<Float>();
    return f;
}

__host__ __device__
vc3 refraction(
    vc3& wiw,
    const vc3& wow,
    Float& pdf,
    const ShadeableIntersection& itsct,
    const Material& m,
    vc3* textureArray,
    thrust::default_random_engine& rng
) {
    // ref pbrt v3 func 528
    float etaA = 1.0; // air
    float etaB = m.indexOfRefraction;
    glm::vec3 refract_normal;

    bool isEntering =  glm::dot(-wow, itsct.vtx.normal) < 0.0;
    float eta = isEntering ? etaA / etaB : etaB / etaA;
    refract_normal = isEntering ? itsct.vtx.normal: -itsct.vtx.normal;

    glm::vec3 refract_dir = glm::refract(
        -wow,
        refract_normal,
        eta
    );

    vc3 f;
    if (glm::length(refract_dir) == 0) {
        // internal reflection
        // TODO
        f = perfectSpecularReflection(wiw, wow, pdf, itsct, m, textureArray, rng);
    }
    else {
        wiw = refract_dir;
        f = m.specular.color / glm::abs(glm::dot(wiw, itsct.vtx.normal));
        if (m.normalTexture.valid == 1) {
            f *= sampleTexture(textureArray, itsct.vtx.uv, m.specularTexture);
        }
        pdf = 1.;
        //pathSegment.ray.origin = intersect + 0.01f * ray_dir;
    }
    return f;

}

__host__ __device__
vc3 SchlickFresnel(
    vc3& wiw,
    const vc3& wow, 
    Float& pdf,
    const ShadeableIntersection& itsct,
    const Material& m,
    vc3* textureArray,
    thrust::default_random_engine& rng
) {
    // ref: pbrt v3 page 550
    thrust::uniform_real_distribution<float> u01(0, 1);
    // TODO do when ray shot from ball
    float etaA = 1;
    float etaB = m.indexOfRefraction;
    float R_0 = powf((etaA - etaB) / (etaA + etaB), 2.0);


    float cos_theta = abs(glm::dot(wow, itsct.vtx.normal));
    float R_theta = R_0 + (1 - R_0) * powf( (1 - cos_theta), 5.0f);

    if (R_theta < u01(rng)) {
        return refraction(wiw, wow, pdf, itsct, m, textureArray, rng);
    }
    else {
        return imperfectSpecularReflection(wiw, wow, pdf, itsct, m, textureArray, rng);
    }
}

__host__ __device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}


__device__ __host__
vc3 evalBsdf(
    const ShadeableIntersection& itsct,
    const glm::vec3& wow,
    const glm::vec3& wiw,
    const Material& m,
    BxDFType& sampledType,
    float& pdf,
    glm::vec3* textureArray,
    thrust::default_random_engine& rng
) {
    /***
    * wiw is given already and evaluate it
    * **/

    thrust::uniform_real_distribution<float> u01(0, 1);
    float p = u01(rng);

    bool on_same_side = glm::dot(wiw, itsct.vtx.normal) * glm::dot(wow, itsct.vtx.normal) > 0;

    vc3 f;
    if (p > m.hasReflective + m.hasRefractive) {
        if (m.dist.type == Flat) {
            // diffuse on cos sample
            Float cosIn = glm::max(glm::dot(itsct.vtx.normal, wiw), 0.0f);
            f = cosIn * m.color;
            if (m.diffuseTexture.valid == 1) {
                f *= sampleTexture(textureArray, itsct.vtx.uv, m.diffuseTexture);
            }
            pdf = cosIn / glm::pi<Float>();
        }
        else if (m.dist.type == TrowbridgeReitz) {
            f = microfaceBRDF_f(wow, wiw, itsct, m, textureArray);
            pdf = microfaceBRDF_pdf(m.dist, wow, wiw, itsct.vtx.normal);
        }
        
    }
    else if (m.hasReflective > 0 && m.hasRefractive > 0 && !on_same_side) {
        // fresnel
        f = vc3(0.);
        pdf = 0.;
    }
    else if (p > m.hasRefractive) {
        f = vc3(0.);
        pdf = 0.;
    }
    else {
        f = vc3(0.);
        pdf = 0.;
    }
    return f;
}

__device__ __host__
vc3 sampleBsdf(
    const ShadeableIntersection& itsct,
    const glm::vec3& wow,
    vc3& wiw,
    Float& pdf,
    const Material& m,
    glm::vec3* textureArray,
    thrust::default_random_engine& rng
) {
    /***
    * need to sample a wiw
    * **/
    thrust::uniform_real_distribution<float> u01(0, 1);
    float p = u01(rng);
    int t = static_cast<int>(m.dist.type);
    vc3 brdf;
    if (p > m.hasReflective + m.hasRefractive) {
        if (m.dist.type == Flat) {
            brdf = LambertBRDF(wiw, pdf, itsct, m, textureArray, rng);
        }
        else if (m.dist.type == TrowbridgeReitz) {
            brdf = microfaceBRDF_sample_f(wow, wiw, itsct, m, pdf, textureArray, rng);
#if _DEBUG
            if (!isfinite(brdf.x) || !isfinite(brdf.y) || !isfinite(brdf.z)) {
                printf("brdf infinite or Nan\n");
            }

            if (!isfinite(pdf)) {
                printf("pdf infinite or Nan and is %f, wow: %f, %f, %f, wiw: %f, %f, %f\n", wow.x, wow.y, wow.z, wiw.x, wiw.y, wiw.z, pdf);
            }
#endif
        }
        
    }
    else if (m.hasReflective > 0 && m.hasRefractive > 0) {
        // fresnel
        brdf = SchlickFresnel(wiw, wow, pdf, itsct, m, textureArray, rng);
    }
    else if (p > m.hasRefractive) {
        brdf = imperfectSpecularReflection(wiw, wow, pdf, itsct, m, textureArray, rng);
    }
    else {
        // refractive under construction
        brdf = refraction(wiw, wow, pdf, itsct, m, textureArray, rng);
    }
    return brdf;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        const ShadeableIntersection& itsct,
        const Material &m,
        glm::vec3* textureArray,
        thrust::default_random_engine &rng) {
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    vc3 brdf;
    vc3 lightIn;
    Float pdf;

    brdf = sampleBsdf(itsct, -pathSegment.ray.direction, lightIn, pdf, m, textureArray, rng);
    
    pathSegment.colorThroughput *= brdf * abs(glm::dot(itsct.vtx.normal, lightIn)) / pdf;
    pathSegment.ray.direction = lightIn;
    pathSegment.ray.origin = itsct.vtx.pos + lightIn * 1e-3f;
    //pathSegment.ray.origin = itsct.pos;
    pathSegment.remainingBounces--;
}

__host__ __device__
bool isBlack(const glm::vec3& v, float e = 1e-6) {
    return glm::all(glm::epsilonEqual(v, glm::vec3(0.), glm::vec3(1e-6)));
}

__host__ __device__
vc3 SphericalDirection(Float sinTheta, Float cosTheta,
    Float phi, const vc3& x, const vc3& y,
    const vc3& z) {
    return sinTheta * std::cos(phi) * x +
        sinTheta * std::sin(phi) * y + cosTheta * z;
}


__device__ __host__
ShadeableIntersection arealight_shape_sample(const ShadeableIntersection& light_itsct, const Geom& light_geom, const glm::vec2& xi, float& lightPdf) {
    // sample a pos based on the light intersection position
    
    ShadeableIntersection it;
    if (light_geom.type == SPHERE) {
        // sample pos on sphere visible on the given point
        
        // coordinate sys for sphere sampling
        glm::vec3 ctr = multiplyMVHomo(light_geom.geomT.transform, glm::vec4(0., 0., 0., 1.));
        glm::vec3 wc = glm::normalize(ctr - light_itsct.vtx.pos);
        glm::vec3 wcX, wcY;
        CoordinateSys(wc, wcX, wcY);
        // <<Sample uniformly on sphere if  is inside it>>
        // TODO

        // <<Sample sphere uniformly inside subtended cone>>
        //Float radius = light_geom.scale[0];
        Float radius = 1.;
        Float sinThetaMax2 = radius * radius / glm::distance2(ctr, light_itsct.vtx.pos);
        Float cosThetaMax = glm::sqrt(glm::max((Float)0, 1 - sinThetaMax2));
        Float cosTheta = (1 - xi[0]) + xi[0] * cosThetaMax;
        Float sinTheta = glm::sqrt(glm::max((Float)0, 1 - cosTheta * cosTheta));
        Float phi = xi[1] * 2 * glm::pi<Float>();

        // <<Compute angle  from center of sphere to sampled point on surface>>
        Float dc = glm::distance(light_itsct.vtx.pos, ctr);
        Float ds = dc * cosTheta -
            glm::sqrt(glm::max((Float)0,
                radius * radius - dc * dc * sinTheta * sinTheta));
        Float cosAlpha = (dc * dc + radius * radius - ds * ds) /
            (2 * dc * radius);
        Float sinAlpha = glm::sqrt(glm::max((Float)0, 1 - cosAlpha * cosAlpha));

        // <<Compute surface normal and sampled point on sphere>> 
        vc3 nObj = SphericalDirection(sinAlpha, cosAlpha, phi,
            -wcX, -wcY, -wc);
        vc3 pObj = (Float)1. * vc3(nObj.x, nObj.y, nObj.z);

        it.vtx.pos = multiplyMVHomo(light_geom.geomT.transform, vc4(pObj, 1.));
        it.vtx.normal = multiplyMV(light_geom.geomT.invTranspose, vc4(pObj, 0.));
        it.materialId = light_geom.materialid;
        it.geom_idx = light_geom.geom_idx;
        // TODO illuminated point inside sphere 
        
        lightPdf = 1.0f / (2.0f * glm::pi<float>() * (1 - cosThetaMax));
    }
    else if (light_geom.type == GeomType::CUBE) {
        // TODO Sample cube
    }
    
    return it;
}

__device__ __host__
glm::vec3 Light_Le(const ShadeableIntersection& itsct, const Geom& light_geom, const Material& mat, const glm::vec3& wiw) {

    if (light_geom.type == SPHERE || light_geom.type == CUBE) {
        return glm::dot(itsct.vtx.normal, wiw) > 0.f ? mat.color * mat.emittance : vc3(0.);
    }
    else {
        // TODO other light
    }
    return glm::vec3(0.);
}

__device__ __host__
glm::vec3 Light_sample_Li(const ShadeableIntersection& given_itsct, ShadeableIntersection& light_itsct, const Geom& light_geom, const Material& mat, const glm::vec2& xi, glm::vec3& wiw, float& lightPdf) {
    light_itsct = arealight_shape_sample(given_itsct, light_geom, xi, lightPdf);
    wiw = glm::normalize(light_itsct.vtx.pos - given_itsct.vtx.pos);
    return Light_Le(light_itsct, light_geom, mat, -wiw);
}

__device__ __host__
Float light_pdf(
    const ShadeableIntersection& light_itsct, const ShadeableIntersection& ref_itsct,
    const Geom& light_geom, const glm::vec3& wiw) {
    // area light
    if (light_geom.type == SPHERE) {
        Float area = 4 * glm::pi<Float>() * light_geom.geomT.scale[0] * light_geom.geomT.scale[0];
        return glm::distance2(light_itsct.vtx.pos, ref_itsct.vtx.pos) / (glm::abs(glm::dot(light_itsct.vtx.normal, (-wiw))) * area);
    }
    else if (light_geom.type == CUBE) {
        // TODO
        return 0.f;
    }
    else {
        return 1.;
    }
}

// reference pbrt v3 BVH traversal
__host__ __device__
int SceneIntersection(
    const Ray& ray,
    Primitive* primitives,
    LinearBVHNode* LBVHnodes,
    ShadeableIntersection& intersection
) {
    // TOCHECK for BVH traverse
    bool hit = false;
    bool outside = true;
    vc3 invDir = vc3(1.0) / ray.direction;
    vc3 dirIsNeg = glm::lessThan(invDir, vc3(0.));
    // <<Follow ray through BVH nodes to find primitive intersections>>=
    int toVisitOffset = 0, currentNodeIndex = 0; int primIndex = -1;
    int nodesToVisit[64];
    Float t_min = FLT_MAX;
    while (true) {
        const LinearBVHNode* node = &LBVHnodes[currentNodeIndex];
        //<< Check ray against BVH node >>
        if (aabbRayIntersectionTest(node->bounds, ray, invDir, dirIsNeg) != -1) {
            if (node->nPrimitives > 0) {
                // << Intersect ray with primitives in leaf BVH node >>
                // need to find the one with smallest t
                ShadeableIntersection itsct_tmp;
                for (int i = 0; i < node->nPrimitives; ++i) {
                    Float t_tmp = primitiveRayIntersectionTest(primitives[node->primitivesOffset + i], ray, itsct_tmp, outside);
                    if (t_tmp > 0 && t_tmp < t_min) {
                        t_min = t_tmp;
                        intersection = itsct_tmp;
                        primIndex = node->primitivesOffset + i;
                        hit = true;
                    }
                }
                            
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];

            }
            else {
                //<< Put far BVH node on nodesToVisit stack, advance to near node >>
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                }
                else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    // first child is immediately after the current node
                    currentNodeIndex += 1;
                }
            }
        }
        else {
            /*if (!hit) {
                printf("aabb not hit with ray o: (%f, %f, %f), d: (%f, %f, %f)\n",
                    ray.origin.x,
                    ray.origin.y,
                    ray.origin.z,
                    ray.direction.x,
                    ray.direction.y,
                    ray.direction.z
                );
            }*/
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    //printf("Intersect prim index: %d\n", primIndex);
    return primIndex;
}

__host__ __device__
int SceneIntersection(
    const Ray& ray,
    Geom* geoms,
    int geoms_size,
    GLTF_Model* models,
    Triangle* triangles,
    ShadeableIntersection& intersection
) {
    float t;
    glm::vec3 normal;
    glm::vec2 uv;

    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];
#if motion_blur
        glm::mat4 inv_transform_cache = geom.inverseTransform;
        glm::mat4 inv_transpose_cache = geom.invTranspose;
        glm::vec3 new_translate = pathSegment.ray.time * geom.velocity + geom.translation;
        glm::mat4 new_transform = dev_buildTransformationMatrix(new_translate, geom.rotation, geom.scale);
        geom.inverseTransform = glm::inverse(new_transform);
        // forget to update invTranspose
        /// ty john marcao
        geom.invTranspose = glm::inverseTranspose(new_transform);
#endif
        Vertex tmp_itsct{ glm::vec3(0), glm::vec3(0), glm::vec2(0) };
        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom.geomT, ray, tmp_itsct, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom.geomT, ray, tmp_itsct, outside);
        }
        else if (geom.type == BBOX) {
            t = meshIntersectionTest(
                geom,
                models,
                triangles,
                ray,
                tmp_itsct,
                outside);
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            normal = tmp_itsct.normal;
            uv = tmp_itsct.uv;
        }

#if motion_blur
        geom.inverseTransform = inv_transform_cache;
        geom.invTranspose = inv_transpose_cache;
#endif
    }

    if (hit_geom_index == -1)
    {
        intersection.t = -1.0f;
    }
    else
    {
        //The ray hits something
        intersection.t = t_min;
        intersection.vtx.pos = getPointOnRay(ray, t_min);
        intersection.materialId = geoms[hit_geom_index].materialid;
        intersection.vtx.normal = normal;
        intersection.vtx.uv = uv;
        
    }
    intersection.geom_idx = hit_geom_index;

    return hit_geom_index;
}

__device__ __host__
glm::vec3 EstimateDirect(
    const ShadeableIntersection& itsct,
    const PathSegment& pathSegment,
    const glm::vec3& wow,
    Material* materials,
    Geom* geoms, int geom_size, GLTF_Model* models, Triangle* triangles,
    Primitive* primitives, LinearBVHNode* LBVHnodes,
    glm::vec3* textureArray,
    const Geom& light_geom,
    thrust::default_random_engine& rng) {
    /// <summary>
    /// Estimate direct light given the light
    /// </summary>
    /// <param name="itsct"></param>
    /// <param name="pathSegment"></param>
    /// <param name="wow"></param>
    /// <param name="materials"></param>
    /// <param name="geoms"></param>
    /// <param name="geom_size"></param>
    /// <param name="models"></param>
    /// <param name="triangles"></param>
    /// <param name="textureArray"></param>
    /// <param name="light_geom"> the geom of the light </param>
    /// <param name="rng"></param>
    /// <returns></returns>

    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 Ld(0.);
    glm::vec3 wiw(0.);
    float lightPdf = 0., float scatteringPdf = 0.;
    glm::vec2 uLight(u01(rng), u01(rng));
    glm::vec2 uScattering(u01(rng), u01(rng));

    glm::vec3 itsct_p = itsct.vtx.pos;
    Material mat = materials[itsct.materialId];
    ShadeableIntersection sampled_light_itsct;
    glm::vec3 Li = Light_sample_Li(itsct, sampled_light_itsct, light_geom, materials[light_geom.materialid], uLight, wiw, lightPdf);
#if DirectLightSampleLight == 1
    if (lightPdf > 0. && !isBlack(Li)) {
        glm::vec3 f(0.);
        BxDFType sampledType;
        if (mat.isSurface) {
            f = evalBsdf(itsct, wow, wiw, mat, sampledType, scatteringPdf, textureArray, rng) * glm::abs(glm::dot(itsct.vtx.normal, wiw));
        }
        else {
            // TODO subsurface
        }

        if (!isBlack(Li)) {
            // test occlusion 
            Ray r{ sampled_light_itsct.vtx.pos - 1e-2f * wiw, -wiw };
            ShadeableIntersection intersection;
#if RAY_SCENE_INTERSECTION == BRUTE_FORCE
            int hit_geom_idx = SceneIntersection(r, geoms, geom_size, models, triangles, intersection);
#elif RAY_SCENE_INTERSECTION == HBVH
            int hit_geom_idx = SceneIntersection(r, primitives, LBVHnodes, intersection);
#endif
            if (hit_geom_idx == -1 || glm::distance2(intersection.vtx.pos, itsct.vtx.pos) > 1e-4) {
                // not hit the same light
                bool handleMedia = false;
                Li *= handleMedia;
            }

#if _DEBUG
            float3 it1 = make_float3(intersection.vtx.pos.x, intersection.vtx.pos.y, intersection.vtx.pos.z);
            float3 it2 = make_float3(itsct.vtx.pos.x, itsct.vtx.pos.y, itsct.vtx.pos.z);
#endif
            float weight = 1.;
            if (light_geom.type == DELTA) {
                Ld += f * Li / lightPdf;
            }
            else {
                weight = powerHeuristic(1., lightPdf, 1., scatteringPdf);
                Ld += f * Li * weight / lightPdf;
            }
        }
    }
    
#endif

#if DirectLightSampleBSDF == 1
    if (light_geom.type != DELTA) {
        glm::vec3 f(0.);
        bool sampledSpecular = false;
        if (mat.isSurface) {
            f = sampleBsdf(itsct, wow, wiw, scatteringPdf, mat, textureArray, rng);
            f *= glm::abs(glm::dot(itsct.vtx.normal, wiw));
            // TODO handle sampledSpecular
        }
        else {
            // TODO handle subsurface
        }

        if (!isBlack(f) && scatteringPdf > 0) {
            //  <<Find intersection and compute transmittance>> 
            glm::vec3 Tr(1.0);
            Ray r{ itsct_p + 1e-3f * wiw, wiw };
            ShadeableIntersection light_intersection;
            int hit_geom_idx = SceneIntersection(r, geoms, geom_size, models, triangles, light_intersection);

            // <<Add light contribution from material sampling>> 
            glm::vec3 Li(0.);
            float weight = 1.;
            if (hit_geom_idx == -1) {
                // TODO did not hit anything, handle infinite area light
            }
            else {
#if RAY_SCENE_INTERSECTION == BRUTE_FORCE
                bool hit_light = (hit_geom_idx == light_geom.geom_idx);
#elif RAY_SCENE_INTERSECTION == HBVH
                bool hit_light = (primitives[hit_geom_idx].geom_idx == light_geom.geom_idx);
#endif
                if (hit_light) {
                    Li = Light_Le(light_intersection, light_geom, materials[light_geom.materialid], -wiw);
                    if (!sampledSpecular) {
                        // get light pdf TODO only get pdf
                        lightPdf = light_pdf(light_intersection, itsct, light_geom, wiw);
                        if (lightPdf == 0) {
                            return Ld;
                        }
                        weight = powerHeuristic(1., scatteringPdf, 1., lightPdf);
                    }
                }
            }           
            Ld += f * Li * Tr * weight / scatteringPdf;
        }
       
    }
    
#endif

    return Ld;
}

__host__ __device__
void UniformSampleOneLight(
    PathSegment& pathSegment,
    const ShadeableIntersection& itsct,
    Material* materials,
    const glm::vec3& wow,
    const int& nLights,
    int *lightIDs,
    Geom* geoms, int geom_size, GLTF_Model* models, Triangle* triangles,
    Primitive* primitives, LinearBVHNode* LBVHnodes,
    vc3* textureArray,
    thrust::default_random_engine& rng
) {
    if (nLights == 0) {
        // no lights, no energy
        return;
    }
    thrust::uniform_int_distribution<int> dist(0, nLights-1);
    int chosen_light_id = lightIDs[dist(rng)];
    vc3 f = EstimateDirect(itsct, pathSegment, -pathSegment.ray.direction, materials,
        geoms, geom_size, models, triangles,
        primitives, LBVHnodes,
        textureArray,
        geoms[chosen_light_id], rng);
    pathSegment.colorSum += pathSegment.colorThroughput * (float)nLights * f;
}




