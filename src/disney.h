#pragma once
#include "Constants.h"
#include "glm/glm.hpp"
#include "sampler.h"
#include "sceneStructs.h"
#include "Constants.h"
#include "microface.h"

// ref: 
// 1. step by step tutorial: https://schuttejoe.github.io/post/disneybsdf/
// 2. glsl implementation: https://github.com/knightcrawler25/GLSL-PathTracer



__host__ __device__ void disneyBrdfIsotropicClearcoat(
    float cosIn, float cosOut, float cosHalf, const Material& m, float fresnelHalf,
    float* color, float* density
) {
    // clearcoat
    float dClearcoat = gtr1(cosHalf, glm::mix(0.1f, 0.001f, m.disneyPara.clearcoatGloss));
    // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04.
    float fClearcoat = glm::mix(0.04f, 1.0f, fresnelHalf);
    float gClearcoat = smithGGgx(cosIn, 0.25f) * smithGGgx(cosOut, 0.25f);

    // the disney implmentation multiplies it by 0.25 here which results in clearcoat being too weak
    *color = m.disneyPara.clearcoat * gClearcoat * fClearcoat;
    *density = dClearcoat;
}



namespace PxrDisney {

    __host__ __device__ inline vc3 BrdfTint(const vc3& linearColor) {
        Float luminance = Math::luminance(linearColor);
        return luminance > 0.0f ? linearColor / luminance : vc3(1.0f);
    }

    __host__ __device__ vc3 evaluateSheen(const Material& m, const Float& dotHL)
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="dotHL"> dot of half vector and in</param>
        /// <returns></returns>
        vc3 tint = BrdfTint(m.color);
        return glm::mix(vc3(1.0f), tint, m.disneyPara.sheenTint) * Fresnel::SchlickWeight(dotHL) * m.disneyPara.sheen;
    }

    __host__ __device__ vc3 evaluateDiffuse(const Float& cosIn, const Float& cosOut, const Float& dotHalfOut, const Material& m, Float& pdf ) {
        pdf = 0.f;
        if (cosOut <= 0.f) {
            return vc3(0.f);
        }
        
        // Subsurface too hard, so we fake it
        Float Fss_90 = m.disneyPara.roughness * dotHalfOut * dotHalfOut;
        Float Fd_90 = 0.5f + 2.f * m.disneyPara.roughness * cosOut * cosOut;

        Float Fss_in  = 1.f + (Fss_90 - 1.f) * Fresnel::SchlickWeight(cosIn);
        Float Fss_out = 1.f + (Fss_90 - 1.f) * Fresnel::SchlickWeight(cosOut);

        Float Fss = (Float)1.25  * (Fss_in * Fss_out * ((Float)1 / (cosIn + cosOut) - 0.5f) + 0.5f) * cosOut;
        
        Float Fd_in = 1.f + (Fd_90 - 1.f) * Fresnel::SchlickWeight(cosIn);
        Float Fd_out = 1.f + (Fd_90 - 1.f) * Fresnel::SchlickWeight(cosOut);

        Float F_baseDiffuse = Fd_in * Fd_out * cosOut;
       
        vc3 F_lambert = m.color * InvPI;

        // cos sample
        pdf = cosOut * InvPI;

        return F_lambert * glm::mix(F_baseDiffuse, Fss, m.disneyPara.subsurface);
    }

    __host__ __device__ vc3 evaluateClearCoat(
        const vc3& wow, 
        const vc3& wiw,
        const vc3& whw, const vc3& n,
        const Material& m,
        Float& pdf
        ) {
        pdf = 0.f;
        vc3 ret(0.f);
        Float HdotO = glm::dot(whw, wow);
        Float Fc = glm::mix(Fresnel::SchlickWeight(HdotO), 1.f, 0.04f);
        Float a_g = glm::mix(0.1f, 0.001f, m.disneyPara.clearcoatGloss);
        Float HdotN = glm::dot(whw, n);
        Float Dc = gtr1(HdotN, a_g);
        // isotropic
        const MicroDistribution& dist{ MicroDistributionType::TrowbridgeReitz, vc2(0.25, 0.25) };
        Float Gc = 0.f;
        Float NdotIn = glm::dot(n, wiw);
        Float NdotOut = glm::dot(n, wow);
        if (NdotIn > 0.f && NdotOut > 0.f) {
            Float Gc = smithGGgx(glm::dot(n, wiw), 0.25f) * smithGGgx(glm::dot(n, wow), 0.25f);
            //Float Gc = distribution_G1(dist, wiw, n) * distribution_G1(dist, wow, n);
            //Float Gc = distribution_G(dist, wow, wiw, n);

            Float NdotH = glm::dot(n, whw);
            pdf = Dc * glm::abs(NdotH) / (4.f * glm::abs(HdotO) + 1e-6f);
            ret = m.color * Fc * Dc * Gc / (4.f * glm::abs(NdotIn) * glm::abs(NdotOut) + 1e-3f);
            /*vc3 tmp_ret = ret / pdf;
            printf("cc: %f, %f, %f, pdf: %f\n", tmp_ret.x, tmp_ret.y, tmp_ret.z, pdf);*/
        }
        return ret;
        
    }

#pragma region Specular
    // also metal
    __host__ __device__ vc3 evaluateSpecularReflection(
        const vc3& wow, const vc3& wiw, const vc3& whw,
        const vc3& specColor, const vc3& F,
        //const Float& eta,
        const MicroDistribution& dist,
        const ShadeableIntersection& itsct,
        const Material& m,
        Float& pdf
    ) {
        pdf = 0.f;
        Float NdotIn = glm::dot(itsct.vtx.normal, wiw);
        if (NdotIn <= 0.f) {
            return vc3(0);
        }
            
        Float NdotOut = glm::dot(itsct.vtx.normal, wow);
        Float HdotOut = glm::dot(whw, wow);
        Float HdotIn = glm::dot(whw, wiw);

        //Float fres = Fresnel::DisneyBlend(m.disneyPara.metallic, eta, HdotIn, HdotOut);
        //vc3 Fm = m.color + (vc3(1.f) - m.color) * Fresnel::SchlickWeight(glm::abs(HdotOut));
        //vc3 Fm = glm::mix(specColor, vc3(1.0f), fres);
        Float D = distribution_D(dist, whw, itsct.vtx.normal);
        Float G1 = distribution_G1(dist, wow, itsct.vtx.normal);
        Float G = G1 * distribution_G1(dist, wiw, itsct.vtx.normal);

        pdf = D * G1 / (4.f * glm::abs(HdotOut));// TODO use disney specular pdf

        return F * D * G / vc3(glm::abs(4.0f * NdotIn * NdotOut));
    }

    __host__ __device__ vc3 evaluateSpecularTransmission(
        const vc3& wow, const vc3& wiw, const vc3& wh,
        const vc3& specColor,
        const Float& eta,
        const MicroDistribution& dist,
        const ShadeableIntersection& itsct,
        const Material& m,
        Float& pdf
    ) {
        pdf = 0.f;
        Float NdotIn = glm::dot(itsct.vtx.normal, wiw);
        if (NdotIn >= 0.f) {
            return vc3(0);
        }
        Float NdotOut = glm::dot(itsct.vtx.normal, wow);
        Float HdotOut = glm::dot(wh, wow);
        Float HdotIn = glm::dot(wh, wiw);
        // here for fresnel we use dielectric
        Float F = Fresnel::Dielectric(glm::abs(glm::dot(wow, wh)), eta);
        Float D = distribution_D(dist, wh, itsct.vtx.normal);
        Float G1 = distribution_G1(dist, wow, itsct.vtx.normal);
        Float G = G1 * distribution_G1(dist, wiw, itsct.vtx.normal);

        Float denom = glm::dot(wiw, wh) + glm::dot(wow, wh) * eta;
        denom *= denom;
        // TOCHECK
        pdf = G1 * glm::max(0.f, NdotOut) * D * glm::abs(HdotIn) / (denom * NdotOut);

        // sqrt for albedo preservation: a ray can refract twice inside an object.
        vc3 f = glm::pow(m.color, vc3(0.5f)) * (1.f - F) * D * G * glm::abs(HdotOut * HdotIn) / (glm::abs(glm::dot(itsct.vtx.normal, wiw)) * denom);
        //printf("specTrans disney bsdf: %f, %f, %f, pdf: %f\n", f.x, f.y, f.z, pdf);
        return f;
    }

#pragma endregion 

    __host__ __device__
    void GetLobeProbabilities(
        const Material& mat, const Float& eta, const vc3& specCol, const Float& Fresnel, 
        Float& diffuseWt, Float& dielectricWt ,Float& metalWt, Float& specRefractWt, Float& clearcoatWt)
    {
        diffuseWt = (1.0 - mat.disneyPara.metallic) * (1.0 - mat.disneyPara.specularTrans) * Math::luminance(mat.color);
        dielectricWt = (1.0 - mat.disneyPara.metallic) * (1.0 - mat.disneyPara.specularTrans) * Math::luminance(glm::mix(specCol, vc3(1.0), Fresnel));
        metalWt = mat.disneyPara.metallic * Math::luminance(glm::mix(mat.color, vc3(1.0), Fresnel));
        clearcoatWt = 0.25 * mat.disneyPara.clearcoat;
        specRefractWt = (1.0 - mat.disneyPara.metallic) * mat.disneyPara.specularTrans;
        
        Float invTotalWt = 1.0 / (diffuseWt + dielectricWt + metalWt + specRefractWt + clearcoatWt + 1e-6f);

        diffuseWt /= invTotalWt;
        clearcoatWt /= invTotalWt;
        dielectricWt /= invTotalWt;
        metalWt /= invTotalWt;
        specRefractWt /= invTotalWt;
        
        /*printf("disnel parameter: anis: %f, sub: %f, rough: %f, metal: %f, speTran: %f, specTint: %f, sheen: %f, sheenTint: %f, cc: %f, ccG: %f \n Fresnel: %f, specCol: %f, %f %f\n weight: diff: %f, dielectric: %f, metal: %f, specRefract: %f, clearcoat: %f \n", 
            mat.disneyPara.anisotropic, mat.disneyPara.subsurface, mat.disneyPara.roughness, mat.disneyPara.metallic, 
            mat.disneyPara.specularTrans, mat.disneyPara.specularTint, mat.disneyPara.sheen, mat.disneyPara.sheenTint,
            mat.disneyPara.clearcoat, mat.disneyPara.clearcoatGloss,
            Fresnel, specCol.x, specCol.y, specCol.z,
            diffuseWt, dielectricWt, metalWt, specRefractWt, clearcoatWt
            );*/
    }

    __host__ __device__ __forceinline__ 
        vc3 GetSpecColor(
        const Material& m, const Float& eta) {
        vc3 ctint = m.color / (Math::luminance(m.color) + 1e-6f);
        Float F0 = (1.0f - eta) / (1.0f + eta);
        F0 = F0 * F0;
        //return glm::mix(F0 * glm::mix(vc3(1.0f), ctint, m.disneyPara.specularTint), m.color, m.disneyPara.metallic);
        return F0 * glm::mix(vc3(1.0f), ctint, m.disneyPara.specularTint);
    }

    __host__ __device__ __forceinline__
    MicroDistribution ANISO_ROUGH_to_Dist(const Material& m) {
        Float aspect = glm::sqrt(1 - 0.9f * m.disneyPara.anisotropic);
        Float ax = m.disneyPara.roughness * m.disneyPara.roughness / aspect;
        Float ay = m.disneyPara.roughness * m.disneyPara.roughness * aspect;
        MicroDistribution dist{ MicroDistributionType::TrowbridgeReitz, vc2(ax, ay) };
        return dist;
    }

    __host__ __device__ vc3 evaluateBSDF(
        const ShadeableIntersection& itsct,
        const Material& m,
        const vc3& wow,
        const vc3& wiw, 
        Float& pdf,
        thrust::default_random_engine& rng
        ) {
        pdf = (Float)0;

        thrust::uniform_real_distribution<Float> u01(0, 1);
        Float r0 = u01(rng),
              r1 = u01(rng);

        vc3 f(0), h;
        
        Float NdotIn = glm::dot(itsct.vtx.normal, wiw);
        Float NdotOut = glm::dot(itsct.vtx.normal, wow);
       
        Float eta = glm::dot(-wow, itsct.vtx.normal) < 0.0 ? 1.0 / m.indexOfRefraction : m.indexOfRefraction;
        if (NdotIn > 0.0)
            h = glm::normalize(wow + wiw);
        else {
            // For specular transmission, the half-angle vector is
            h = glm::normalize(wiw + wow * eta);
        }
            
        Float HdotIn = glm::dot(h, wiw);
        Float HdotOut = glm::dot(h, wow);
        
        // Lobe weights
        Float diffuseWt, metalWt, dielectricWt, specRefractWt, clearcoatWt;

        vc3 specCol = GetSpecColor(m, eta);
        Float fresnel = Fresnel::SchlickWeight(NdotOut);
        GetLobeProbabilities(m, eta, specCol, fresnel, diffuseWt, dielectricWt, metalWt, specRefractWt, clearcoatWt);
        // CDF for picking a lobe
        Float tmp_pdf = 0;

        bool isReflect = (NdotIn * NdotOut) > 0.f;
        if (diffuseWt > 0.f && isReflect) {
            vc3 diffuse_f = 
                evaluateDiffuse(glm::abs(NdotIn), glm::abs(NdotOut), glm::abs(HdotOut), m, tmp_pdf) +
                evaluateSheen(m, glm::abs(HdotIn));
            f += diffuse_f * (1.f - m.disneyPara.metallic) * (1.f - m.disneyPara.specularTrans);
            pdf += tmp_pdf * diffuseWt;
        }

        if (clearcoatWt > 0.f && isReflect) {
            f += evaluateClearCoat(wow, wiw, h, itsct.vtx.normal, m, tmp_pdf) * 0.25f * m.disneyPara.clearcoat;
            pdf += tmp_pdf * clearcoatWt;
        }

        MicroDistribution dist = ANISO_ROUGH_to_Dist(m);
        
        if (metalWt > 0.f && isReflect) { // TODO test NdotIn > 0.f && NdotOut > 0.f
            Float metallicFresnel = Fresnel::SchlickWeight(HdotIn); // TODO
            vc3 F = glm::mix(m.color, vc3(1.0), metallicFresnel);
            f += evaluateSpecularReflection(wow, wiw, h, specCol, F, dist, itsct, m, tmp_pdf) * metalWt;
            pdf += tmp_pdf * metalWt;
        }

        if (dielectricWt > 0.f && isReflect) { // TODO test NdotIn > 0.f && NdotOut > 0.f
            Float dielectricFresnel = Fresnel::Dielectric(glm::abs(HdotIn), eta);
            vc3 F = glm::mix(specCol, vc3(1.0), dielectricFresnel);; // TODO

            f += evaluateSpecularReflection(wow, wiw, h, specCol, F, dist, itsct, m, tmp_pdf) * (1.f - m.disneyPara.metallic) * (1.f - m.disneyPara.specularTrans);
            pdf += tmp_pdf * metalWt;
        }

        if (specRefractWt > 0.f && !isReflect) {
            f += evaluateSpecularTransmission(wow, wiw, h, specCol, eta, dist, itsct, m, tmp_pdf) * (1.f - m.disneyPara.metallic) * m.disneyPara.specularTrans;
            pdf += tmp_pdf * specRefractWt;
        }

        //return f * glm::abs(NdotIn);
        return f;
    }

    

    __host__ __device__ vc3 sampleBSDF(
        const ShadeableIntersection& itsct,
        const vc3& wow,
        vc3& wiw,
        Float& pdf,
        PrevSegmentInfo& prev,
        const Material& m,
        thrust::default_random_engine& rng) {
        pdf = 0.f;
        thrust::uniform_real_distribution<Float> u01(0, 1);
        Float r0 = u01(rng), 
              r1 = u01(rng);
        Float eta = glm::dot(-wow, itsct.vtx.normal) < 0.0 ? 1.0 / m.indexOfRefraction : m.indexOfRefraction;
        // Lobe weights
        Float diffuseWt, metalWt, dielectricWt, specRefractWt, clearcoatWt;

        vc3 specCol = GetSpecColor(m, eta);
        Float dotOutN = glm::dot(wow, itsct.vtx.normal);
        Float approxFresnel = Fresnel::SchlickWeight(dotOutN);
        GetLobeProbabilities(m, eta, specCol, approxFresnel, diffuseWt, dielectricWt, metalWt, specRefractWt, clearcoatWt);
        // CDF for picking a lobe
        Float cdf[5];
        cdf[0] = diffuseWt;
        cdf[1] = cdf[0] + clearcoatWt;
        cdf[2] = cdf[1] + dielectricWt;
        cdf[3] = cdf[2] + metalWt;
        cdf[4] = cdf[3] + specRefractWt;

        vc3 f, h; // bsdf, half vector
        Float NdotOut = glm::dot(itsct.vtx.normal, wow); 
        if (r0 < cdf[0]) {// Diffuse Reflection Lobe
            wiw = CosineSampleHemisphere(itsct.vtx.normal, rng);
            h = glm::normalize(wiw + wow);
            Float NdotIn = glm::dot(itsct.vtx.normal, wiw);
            Float HdotIn = glm::dot(h, wiw);
            Float HdotOut = glm::dot(h, wow);
            vc3 diff_f = evaluateDiffuse(glm::abs(NdotIn), glm::abs(NdotOut), glm::abs(HdotOut), m, pdf);
            // in 2015 disney bsdf they add some sheen(due to multiple scattering)
            vc3 sheen_f = evaluateSheen(m, HdotIn);
            f = diff_f + sheen_f;
            //printf("sample diffuse: %f, %f, %f, sheen: %f, %f, %f\n", diff_f.x, diff_f.y, diff_f.z, sheen_f.x, sheen_f.y, sheen_f.z);
            f *= (1.f - m.disneyPara.metallic) * (1.f - m.disneyPara.specularTrans);
            pdf *= diffuseWt;
        }
        else if (r0 < cdf[1]){
            // ClearCoat TODO
            vc2 xi(u01(rng), u01(rng));
            vc3 whw = sampleGTR1(m.disneyPara.clearcoatGloss, xi, itsct.vtx.normal);

            Float NdotH = glm::dot(whw, itsct.vtx.normal);
            whw *= 2 * (NdotH >= 0) - 1.f;
            wiw = glm::normalize(glm::reflect(-wow, whw));

            f = evaluateClearCoat(wow, wiw, whw, itsct.vtx.normal, m, pdf);
            f *= 0.25f * m.disneyPara.clearcoat;
            pdf *= clearcoatWt;
            //printf("clearcoat disney bsdf: %f, %f, %f, pdf: %f\n", f.x, f.y, f.z, pdf);
        }
        else {
            // Reflect
            MicroDistribution dist = ANISO_ROUGH_to_Dist(m);

            vc2 xi(u01(rng), u01(rng));
            // we can only sure that wh, wow is on the same hemisphere W.R.T to normal
            vc3 wh = distribution_sample_wh(dist, wow, itsct.vtx.normal, xi);

            /*Material tmp_m = m;
            tmp_m.disneyPara.anisotropic = 0.f;
            MicroDistribution tmp_dist = ANISO_ROUGH_to_Dist(tmp_m);
            vc3 tmp_wh = distribution_sample_wh(tmp_dist, wow, itsct.vtx.normal, xi);
            
            Float cos_diff = glm::dot(tmp_wh, wh);
            printf("aniso diff: %f\n", cos_diff);*/

            Float NdotH = glm::dot(wh, itsct.vtx.normal); 
            Float HdotOut = glm::dot(wh, wow);
            // if not the same semisphere as normal, flip it
           

            //Float fresnel = Fresnel::DisneyBlend(m.disneyPara.metallic, eta, HdotOut, HdotOut); // TODO 
            //Float F = 1.f - ((1.0 - fresnel) * m.disneyPara.specularTrans * (1.f - m.disneyPara.metallic));

            //printf("fresnel : %f, F: %f\n", fresnel, F);
            if (r0 < cdf[2]) {
                // dielectric
                wiw = glm::normalize(glm::reflect(-wow, wh));
                Float dielectricFresnel = Fresnel::Dielectric(glm::abs(glm::dot(wiw, wh)), eta);
                vc3 F = glm::mix(specCol, vc3(1.0), dielectricFresnel);; // TODO
                f = evaluateSpecularReflection(wow, wiw, wh, specCol, F, dist, itsct, m, pdf);
                f *= (1.f - m.disneyPara.metallic) * (1.f - m.disneyPara.specularTrans);
                pdf *= dielectricWt;
            }
            else if (r0 < cdf[3]) {
                // metal
                wiw = glm::normalize(glm::reflect(-wow, wh));
                Float metallicFresnel = Fresnel::SchlickWeight(glm::dot(wiw, wh)); // TODO
                vc3 F = glm::mix(m.color, vc3(1.0), metallicFresnel);
                f = evaluateSpecularReflection(wow, wiw, wh, specCol, F, dist, itsct, m, pdf);
                f *= m.disneyPara.metallic;
                pdf *= metalWt;
            }
            else
            {
                // refractive
                wiw = glm::normalize(glm::refract(-wow, wh, eta)); // TODO
                f = evaluateSpecularTransmission(wow, wiw, wh, specCol, eta, dist, itsct, m, pdf);
                f *= (1.f - m.disneyPara.metallic) * m.disneyPara.specularTrans;
                pdf *= specRefractWt;

                //vc3 tmp_f = f / pdf;
                //printf("specTrans disney tmp bsdf: %f, %f, %f, pdf: %f\n", tmp_f.x, tmp_f.y, tmp_f.z, pdf);
            } 
#if _DEBUG
            if (!isfinite(f.x) || !isfinite(f.y) || !isfinite(f.z)) {
                printf("reflect brdf infinite or Nan\n");
            }
#endif
            /*if (!isfinite(pdf) || glm::abs(pdf) < EPSILON) {
                printf("pdf infinite or Nan or zero and is %f, wow: (%f, %f, %f), wiw: (%f, %f, %f), wh: (%f, %f, %f)\n", pdf, wow.x, wow.y, wow.z, wiw.x, wiw.y, wiw.z, wh.x, wh.y, wh.z);
            }*/
        }
        //printf("disney bsdf: %f, %f, %f, pdf: %f\n", f.x, f.y, f.z, pdf);
        //return f * glm::abs(glm::dot(itsct.vtx.normal, wiw));
        return f;
    }
}
