#pragma once
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "intersections.h"
#include "cfg.h"
#include "utility"

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

__host__ __device__ glm::vec3 crossDirection(glm::vec3 v) {
    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    if (abs(v.x) < SQRT_OF_ONE_THIRD) {
        return glm::vec3(1, 0, 0);
    }
    else if (abs(v.y) < SQRT_OF_ONE_THIRD) {
        return glm::vec3(0, 1, 0);
    }
    return glm::vec3(0, 0, 1);
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

    glm::vec3 directionNotNormal = crossDirection(normal);

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__forceinline__
__host__ __device__ glm::vec3 sampleTexture(glm::vec3* dev_textures, glm::vec2 uv, TextureDescriptor tex)
{
    //uv.y = 1.f - uv.y;
    uv = glm::mod(uv * tex.repeat, glm::vec2(1.f));
    if (tex.type == 0)
    {
        int x = glm::min((int)(uv.x * tex.width), tex.width - 1);
        int y = glm::min((int)(uv.y * tex.height), tex.height - 1);
        int index = y * tex.width + x;
        return dev_textures[tex.index + index];
    }
    else
    {
        int steps = 0;
        glm::vec2 z = glm::vec2(0.f);
        glm::vec2 c = (uv * 2.f - glm::vec2(1.f)) * 1.5f;
        c.x -= .5;

        for (steps = 0; steps < 100; steps++)
        {
            float x = z.x * z.x - z.y * z.y + c.x;
            float y = 2.f * z.x * z.y + c.y;

            z = glm::vec2(x, y);

            if (glm::dot(z, z) > 2.f)
                break;
        }

        float sn = float(steps) - log2(log2(dot(z, z))) + 4.0f; // http://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
        sn = glm::clamp(sn, 0.1f, 1.f);
        return glm::vec3(sn);
    }
}

__host__ __device__
void perfectSpecularReflection(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    glm::vec3 reflection_dir = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.color *= m.color;
    pathSegment.ray.direction = reflection_dir;
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
}

__host__ __device__
void imperfectSpecularReflection(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    
    
    const float n = m.specular.exponent;
    if (n >= 4096.0f) {
        perfectSpecularReflection(pathSegment, intersect, normal, m, rng);
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
        glm::vec3 perfect_reflect_dir = glm::reflect(pathSegment.ray.direction, normal);
        glm::vec3 base_z = glm::normalize(perfect_reflect_dir);
        glm::vec3 base_x = glm::normalize(glm::cross(base_z, glm::vec3(0.0f, 0.0f, 1.0f)));
        glm::vec3 base_y = glm::normalize(glm::cross(base_z, base_x));
        glm::mat3 base_m = glm::mat3(base_x, base_y, base_z);

        pathSegment.ray.direction = base_m * local_dir;
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + 0.001f * normal;

    }
}

__host__ __device__
void diffuseReflection(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    const glm::vec3& normal,
    const glm::vec2& uv,
    const Material& m,
    glm::vec3* textureArray,
    thrust::default_random_engine& rng) {
    glm::vec3 diffuse_dir = calculateRandomDirectionInHemisphere(normal, rng);

    //float costheta = glm::dot(normal, diffuse_dir);
    pathSegment.ray.direction = diffuse_dir;

    //pathSegment.color *= m.color * costheta;
    pathSegment.color *= m.color;
    if (m.diffuseTexture.valid == 1) {
        pathSegment.color *= sampleTexture(textureArray, uv, m.diffuseTexture);
        //pathSegment.color = glm::vec3(uv, 0.);
    }
    //pathSegment.color = glm::normalize(normal);
   // pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
    pathSegment.ray.origin = intersect + normal * 0.01f;
}

__host__ __device__
void refraction(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng
) {
    // ref pbrt v3 func 528
    float etaA = 1.0; // air
    float etaB = m.indexOfRefraction;
    glm::vec3 refract_normal;
    glm::vec3& ray_dir = pathSegment.ray.direction;

    bool isEntering =  glm::dot(ray_dir,  normal) < 0.0;
    float eta = isEntering ? etaA / etaB : etaB / etaA;
    refract_normal = isEntering ? normal : - normal;

    glm::vec3 refract_dir = glm::refract(
        ray_dir,
        refract_normal,
        eta
    );

    if (glm::length(refract_dir) == 0) {
        // internal reflection
        // TODO
        perfectSpecularReflection(pathSegment, intersect, normal, m, rng);
    }
    else {
        ray_dir = refract_dir;
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + 0.01f * ray_dir;
    }


}

__host__ __device__
void SchlickFresnel(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng
) {
    // ref: pbrt v3 page 550
    thrust::uniform_real_distribution<float> u01(0, 1);
    // TODO do when ray shot from ball
    float etaA = 1;
    float etaB = m.indexOfRefraction;
    float R_0 = powf((etaA - etaB) / (etaA + etaB), 2.0);

    float cos_theta = abs(glm::dot(pathSegment.ray.direction, normal));
    float R_theta = R_0 + (1 - R_0) * powf( (1 - cos_theta), 5.0f);

    if (R_theta < u01(rng)) {
        refraction(pathSegment, intersect, normal, m, rng);
    }
    else {
        imperfectSpecularReflection(pathSegment, intersect, normal, m, rng);
    }
}

__host__ __device__ float powerHeuristic(float pdf1, float pdf2) {
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return pdf1 / (pdf1 + pdf2);
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
        const glm::vec3& intersect,
        const glm::vec3& normal,
        const glm::vec2& uv,
        const Material &m,
        glm::vec3* textureArray,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_real_distribution<float> u01(0, 1);
    float p = u01(rng);
    // try diffuse
    //pathSegment.ray.origin = intersect;
    if (p > m.hasReflective + m.hasRefractive) {
        diffuseReflection(pathSegment, intersect, normal, uv, m, textureArray, rng);
    } 
    else if (m.hasReflective > 0 && m.hasRefractive > 0) {
        // fresnel
        SchlickFresnel(pathSegment, intersect, normal, m, rng);
    }
    else if (p > m.hasRefractive) {
        imperfectSpecularReflection(pathSegment, intersect, normal, m, rng);
    }
    else {
        // refractive under construction
        refraction(pathSegment, intersect, normal, m, rng);
    }
    
    pathSegment.remainingBounces--;
}
