#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "glm/glm.hpp"
#include "cudahelper.h"
#include <cstring> // std:;memcpy
#include <vector>

typedef float Float;
typedef glm::vec2 vc2;
typedef glm::vec3 vc3;
typedef glm::vec4 vc4;

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