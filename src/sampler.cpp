#include "sampler.h"

Distribution2D::Distribution2D(const Float* func, const int& _nu, const int& _nv) : valid(1), nu(_nu), nv(_nv) {
    int total_size = (nu * nv + nv + nv + 1);
    cudaMallocManaged(&pdf, total_size * sizeof(Float)); // [0, (nu + 1) * nv]: pdf, then PdfIntegral
    cudaMallocManaged(&cdf, total_size * sizeof(Float));
    checkCUDAError("cuda distribution error!\n");
    std::memcpy(pdf, (void*)func, nu * nv * sizeof(Float));
    std::memset(cdf, (Float)0, total_size * sizeof(Float));
    //<< Compute cdf >>
    const int wid = nu + 1; // dim of each cdf
    std::vector<Float> marginalPdf(nv);
    for (int v = 0; v < nv; ++v) { // each pConditionalV
        for (int i = 1; i < wid; i++) {
            cdf[v * wid + i] = cdf[v * wid + i - 1] + func[v * nu + i - 1] / nu;
        }
        // normalize
        Float& funcInt = pdf[wid * nv + v]; // integral of cdf, offset starting from (nu + 1) * nv, after all the conditional pdf and the margianl
        funcInt = cdf[wid * v + wid - 1]; // get the last 
        marginalPdf[v] = funcInt; // is this reference?
        if (funcInt == 0) {
            for (int i = 1; i < wid; ++i)
                cdf[v * wid + i] = Float(i) / Float(nu);
        }
        else {
            for (int i = 1; i < wid; ++i)
                cdf[v * wid + i] /= funcInt;
        }
    }
    // get margianl pdf, cdf, funcInt
    // pdf
    std::memcpy(pdf + nu * nv, marginalPdf.data(),(size_t)nv);
    // cdf
    for (int i = 1; i < nv + 1; i++) {
        //cdf[wid * nv + i] = cdf[wid * nv + i - 1] + cdf[nu * (nv + 1) + i] / nv;
        cdf[wid * nv + i] = cdf[wid * nv + i - 1] + pdf[nu * nv + i - 1] / nv;
    }
    Float& funcInt = pdf[total_size - 1];
    funcInt = cdf[wid * nv + nv];
    if (funcInt == 0) {
        for (int i = 1; i < nv + 1; ++i)
            cdf[wid * nv + i] = Float(i) / Float(nv);
    }
    else {
        for (int i = 1; i < nv + 1; ++i)
            cdf[wid * nv + i] /= funcInt;
    }
}
