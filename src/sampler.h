#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "glm/glm.hpp"
#include "cudahelper.h"
#include <cstring> // std:;memcpy
#include <vector>
#include <thrust/random.h>
#include "cudaUtils.cuh"
#include "Constants.h"

//ref: https://github.com/mmerchante/CUDA-Path-tracer
struct TextureDescriptor
{
    int valid;
    int type; // 0 bitmap, 1 procedural TODO
    int index;
    int width;
    int height;
    glm::vec2 repeat;
    TextureDescriptor() : valid(-1), type(0), index(-1), width(0), height(0), repeat(glm::vec2(1.f)) {};
};

__forceinline__
__host__ __device__ glm::vec3 sampleTexture(glm::vec3* dev_textures, glm::vec2 uv, const TextureDescriptor& tex)
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

#pragma region Distribution2D
struct Distribution1D
{
    Float* func, * cdf;
    int n;
    Float funcInt;

    Distribution1D() = default;

    Distribution1D(const Float* f, int n) : n(n) {
        /*cudaMallocManaged(&func, n * sizeof(Float));
        cudaMallocManaged(&cdf, (n + 1) * sizeof(Float));*/
        // assume f is on CPU
        // TODO memset
    }

    ~Distribution1D() {
        /*cudaFree(func);
        cudaFree(cdf);*/
    }

    __host__ __device__
    Float SampleContinuous(Float u, Float* pdf, int* off = nullptr) const {

    // TODO
        return -1;
    }

    __host__ __device__
    int SampleDiscrete(Float u, Float* pdf = nullptr,
            Float* uRemapped = nullptr) const {
        // TODO
        return -1;
    }

    __host__ __device__
        Float DiscretePDF(int index) const {
        return func[index] / (funcInt * n);
    }
};

struct Distribution2D {
    /// <summary>
    /// contains nv pConditionals([nu + 1, nv]), 1 pMarginalV([nu + 1, 1])
    /// </summary>
    /// 
    int valid = 0;
    int nu = -1, nv = -1;
    
    Float* pdf = nullptr;
    Float* cdf = nullptr;

    Distribution2D(): valid(0) { 
    };

    Distribution2D(const Float* func, const int& nu, const int& nv);

    __host__ __device__ __forceinline__
    Float SampleContinuous1D(const Float& u, Float& sampled_pdf, Float* v_pdf, Float* v_cdf, const Float& funcInt, const int& pdf_size, int *off = nullptr) const {
        // <<Find surrounding CDF segments and offset>> 
        int first = 0, len = pdf_size + 1;
        while (len > 0) {
            int half = len >> 1, middle = first + half;
            // <<Bisect range based on value of pred at middle>>
            if (v_cdf[middle] < u) {
                first = middle + 1;
                len -= half + 1;
            }
            else {
                len = half;
            }
        }
        int offset = glm::clamp(first - 1, 0, pdf_size + 1 - 2);
        if (off) *off = offset;
        //<< Compute offset along CDF segment >>
        Float du = u - v_cdf[offset];
        if ((v_cdf[offset + 1] - v_cdf[offset]) > 0)
            du /= (v_cdf[offset + 1] - v_cdf[offset]);
       // << Compute PDF for sampled offset >>
        if (pdf) *pdf = v_pdf[offset] / funcInt;

        //<< Return  corresponding to sample >>
        return (offset + du) / pdf_size;
    }

    __host__ __device__
    vc2 SampleContinuous(const vc2& u, Float& sampled_pdf) const {
        // TOCHECK
        vc2 pdfs;
        int v;

        Float* marginal_pdf = this->pdf + nu * nv;
        Float* marginal_cdf = this->cdf + (nu + 1) * nv;
        Float* funcInts = this->pdf + nu * nv + nv;
        Float d1 = this->SampleContinuous1D(u[1], pdfs.y, marginal_pdf, marginal_cdf, funcInts[nv], nv, &v);
        Float* pdf_1d = this->pdf + nu * v;
        Float* cdf_1d = this->cdf + (nu + 1) * v;
        Float d0 = this->SampleContinuous1D(u[0], pdfs.x, pdf_1d, cdf_1d, funcInts[v], nu);
        sampled_pdf = pdfs.x * pdfs.y;
        return vc2(d0, d1);
    }

    __host__ __device__
    Float Pdf(const vc2& p) const {
    // TOCHECK
    // The value of the PDF for a given sample value is computed as 
    // the product of the conditional and marginal PDFs for sampling it from the distribution.
        int iu = glm::clamp(int(p.x * nu), 0, nu - 1);
        int iv = glm::clamp(int(p.y * nv), 0, nv - 1);
        //printf("nu:%d, nv:%d, iu: %d, iv: %d\n", nu, nv,iu, iv);
        Float ret = pdf[nu * iv + iu] / pdf[(nu + 2) * nv];
        //printf("sample 2d pdf: %f\n", ret);
        return ret;
    }

    
};
#pragma endregion Distribution2D

__forceinline__
__host__ __device__ Float gtr1(const Float& cosHalf, const Float& a) {
    // Generalized Trowbridge-Reitz ,used by Clearcoat created by Burley
    if (a >= 1) {
        return InvPI;
    }

    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * cosHalf * cosHalf;
    return (a2 - 1.0f) / (PI * glm::log(a2) * t);
}

__forceinline__
__host__ __device__ Float gtr2(const Float& cosHalf, const Float& a) {
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * cosHalf * cosHalf;
    return a2 / (PI * t * t);
}

__forceinline__
__host__ __device__ vc3 sampleGTR1(const Float& rough, const vc2& xi, const vc3& n) {
    float a = glm::clamp(rough, 1e-6f, 1.f - 1e-3f);
    float a2 = a * a;

    float Hazimuth = xi[0] * TWO_PI;

    float cosTheta = glm::sqrt((1.0f - pow(a2, 1.0f - xi[0])) / (1.0f - a2));
    float sinTheta = glm::clamp(sqrt(1.0f - (cosTheta * cosTheta)), 0.0f, 1.0f);
    float sinPhi = glm::sin(Hazimuth);
    float cosPhi = glm::cos(Hazimuth);

    vc3 perpendicularDirection1;
    vc3 perpendicularDirection2;
    CoordinateSys(n, perpendicularDirection1, perpendicularDirection2);

    vc3 w = vc3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
    // to world
    return w.x * perpendicularDirection1 + w.y * perpendicularDirection2 + w.z * n;
}

__forceinline__
__host__ __device__ Float smithGGgx(const Float& cosOut, const Float& alphaG) {
    /// <summary>
    /// // Smith masking/shadowing term.
    /// </summary>
    /// <param name="cosOut"></param>
    /// <param name="alphaG"></param>
    /// <returns></returns>
    float a = alphaG * alphaG, b = cosOut * cosOut;
    return 2.0f * cosOut / (cosOut + glm::sqrt(a + b - a * b));
}

// cosine weighted
__forceinline__
__host__ __device__
glm::vec3 CosineSampleHemisphere(
    const glm::vec3& normal, thrust::default_random_engine& rng) {
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

namespace Fresnel {
    __host__ __device__ __forceinline__ Float SchlickWeight(const Float& cos) {
        // fresnel factor (1 - cosTheta)^5
        // TODO why clamp
        float m = glm::clamp(1.0f - cos, 0.0f, 1.0f);
        float sqm = m * m;
        return sqm * sqm * m;
    }

    __host__ __device__ __forceinline__ Float FrSchlick(const Float& R0, const Float& cos) {
        return glm::mix(SchlickWeight(cos), R0, 1.f);
    }

    __host__ __device__ __forceinline__
    Float Dielectric(const Float& cosThetaI, const Float& eta)
    {
        //actual Fresnel equation for the dielectric materials
        float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

        // Total internal reflection
        if (sinThetaTSq > 1.0)
            return 1.0;

        float cosThetaT = glm::sqrt(glm::max(1.0 - sinThetaTSq, 0.0));

        float rs = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);
        float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);

        return 0.5f * (rs * rs + rp * rp);
    }

    __host__ __device__ __forceinline__
        Float DisneyBlend(const Float& metallic, const Float& eta, const Float& cosInHalf, const Float& cosOutHalf) {
        Float metallicFresnel = SchlickWeight(cosInHalf);
        Float dielectricFresnel = Dielectric(abs(cosOutHalf), eta);
        return glm::mix(dielectricFresnel, metallicFresnel, metallic);
    }
};