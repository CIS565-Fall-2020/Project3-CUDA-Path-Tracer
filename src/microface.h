#pragma once

#include "intersections.h"
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/norm.hpp>
#include "interactions.h"
#include "cudaUtils.cuh"

__host__ __device__
Float sinPhi(const vc3& a, const vc3& normal) {
	// x / sinTheta
	Float sinTheta = glm::sqrt(glm::max((Float)0, 1 - glm::dot(a, normal) * glm::dot(a, normal)));

	vc3 x, y;
	CoordinateSys(normal, x, y);
	Float tmp = glm::dot(a, y);

	return (sinTheta == 0) ? 0 : glm::clamp(glm::dot(a, y) / sinTheta, -1.f, 1.f);
}

__host__ __device__
Float cosPhi(const vc3& a, const vc3& normal) {
	// x / sinTheta
	float3 f3_a = make_float3(a.x, a.y, a.z);
	float3 f3_n = make_float3(normal.x, normal.y, normal.z);

	Float sinTheta = glm::sqrt(glm::max((Float)0, (Float)1 - glm::dot(a, normal) * glm::dot(a, normal)));
	
	vc3 x, y;
	CoordinateSys(normal, x, y);
	Float tmp = glm::dot(a, x);

	return (sinTheta == 0) ? (Float)1 : glm::clamp(glm::dot(a, x) / sinTheta, -1.f, 1.f);
}

__host__ __device__
bool SameHemiSphere(const vc3& a, const vc3& b) {
	return glm::dot(a, b) > 0;
}

__host__ __device__
vc3 fresnel_evalulate(Float cosThetaI) {
	return vc3(1.);
}

__host__ __device__
vc3 distribution_sample_wh(const MicroDistribution& distrib, const vc3& wo, const vc2& xi) {
	// Samples the distribution of microfacet normals to generate one
	// about which to reflect wo to create a wi.

	//if (distrib.type == TrowbridgeReitz) {
		vc3 wh;
		float cosTheta = 0, phi = (2 * glm::pi<Float>()) * xi[1];
		if (distrib.alpha.x == distrib.alpha.y) {
			float tanTheta2 = distrib.alpha.x * distrib.alpha.x * xi[0] / (1.0f - xi[0]);
			cosTheta = 1 / glm::sqrt(1 + tanTheta2);
		}
		else {
			phi =
				glm::atan(distrib.alpha.y / distrib.alpha.x * tan(2 * glm::pi<Float>() * xi[1] + .5f * glm::pi<Float>()));
			if (xi[1] > .5f) phi += glm::pi<Float>();
			float sinPhi = sin(phi), cosPhi = cos(phi);
			const float alphax2 = distrib.alpha.x * distrib.alpha.x, alphay2 = distrib.alpha.y * distrib.alpha.y;
			const float alpha2 =
				1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
			float tanTheta2 = alpha2 * xi[0] / (1 - xi[0]);
			cosTheta = 1 / glm::sqrt(1 + tanTheta2);
		}
		float sinTheta =
			glm::sqrt(glm::max((float)0., (float)1. - cosTheta * cosTheta));

		wh = vc3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi),
			cosTheta);
		if (!SameHemiSphere(wo, wh)) wh = -wh;

		return wh;
	//}
	//return vc3(0.);
}

__host__ __device__
Float Lambda(const MicroDistribution& distrib, const vc3& wh, const vc3& normal) {
	// Shadowing - masking functions, which measures 
	// invisible masked microfacet area per visible microfacet area.
	//if (distrib.type == TrowbridgeReitz) {
		Float cosTheta = glm::dot(wh, normal);
		Float cos2Theta = cosTheta * cosTheta;
		Float tan2Theta = thrust::max(0., 1. - cos2Theta) / cos2Theta;
		Float absTanTheta = sqrt(tan2Theta);
		if (isfinite(absTanTheta)) return 0.;

		// Compute alpha for direction w
		Float cos_phi = cosPhi(wh, normal);
		Float sin_phi = sinPhi(wh, normal);
		Float alpha =
			sqrt(cos_phi * cos_phi * distrib.alpha.x * distrib.alpha.x + sin_phi * sin_phi * distrib.alpha.y * distrib.alpha.y);
		Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
		return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
	/*}
	return 0;*/
}

__host__ __device__
Float distribution_D(const MicroDistribution& distrib, const vc3& wh, const vc3& normal) {
	// normal disribution
	/*if (distrib.type == TrowbridgeReitz) {*/
		Float cosTheta = glm::dot(wh, normal);
		Float cos2Theta = cosTheta * cosTheta;
		Float tan2Theta = thrust::max(0., 1. - cos2Theta) / cos2Theta;
		if (isfinite(tan2Theta)) { // handle infinity and Nan 
			return (Float)0.;
		}
		Float cos4Theta = cos2Theta * cos2Theta;

		float c = cosPhi(wh, normal);
		float s = sinPhi(wh, normal);
		Float e = (c * c / (distrib.alpha.x * distrib.alpha.x) +
			s * s / (distrib.alpha.y * distrib.alpha.y)) / tan2Theta;
		printf("wh: %f, %f, %f, n: %f, %f, %f;  c: %d, s: %d, e: %d\n",wh.x, wh.y, wh.z, normal.x, normal.y, normal.z, c, s, e);

		return 1. / (glm::pi<Float>() * distrib.alpha.x * distrib.alpha.y * cos4Theta * (1 + e) * (1 + e));
	/*}
	return 0.;*/
}

__host__ __device__
Float distribution_pdf(const MicroDistribution& distrib, const vc3& wo, const vc3& wh, const vc3& normal) {
	return distribution_D(distrib, wh, normal) * glm::abs(glm::dot(wh, normal)) ;
}

__host__ __device__
Float distribution_G(const MicroDistribution& distrib, const vc3& wo, const vc3& wi, const vc3& normal) {
	Float l1 = Lambda(distrib, wo, normal);
	Float l2 = Lambda(distrib, wo, normal);
	return 1. / (1. + l1 + l2);
}

__host__ __device__
vc3 microfaceBRDF_f(
	const vc3& wo, const vc3& wi,
	const ShadeableIntersection& itsct,
	const Material& mat,
	glm::vec3* textureArray) {
	Float cosThetaO = glm::dot(wo, itsct.surfaceNormal), cosThetaI = glm::dot(wi, itsct.surfaceNormal);
	vc3 wh = wo + wi;
	// Handle degenerate cases for microfacet reflection
	if (cosThetaI == 0 || cosThetaO == 0) return vc3(0.f);
	if (glm::length2(wh) < 1e-4f) return vc3(0.f);

	
	float3 f3_wo = make_float3(wo.x, wo.y, wo.z);
	float3 f3_wi = make_float3(wi.x, wi.y, wi.z);
	float3 f3_n = make_float3(itsct.surfaceNormal.x, itsct.surfaceNormal.y, itsct.surfaceNormal.z);

	wh = glm::normalize(wh);
	float3 f3_wh = make_float3(wh.x, wh.y, wh.z);
	vc3 F = fresnel_evalulate(glm::dot(wi, wh));
	Float D = distribution_D(mat.dist, wh, itsct.surfaceNormal);
	Float G = distribution_G(mat.dist, wo, wi, itsct.surfaceNormal);

	vc3 c = mat.color;
	if (mat.diffuseTexture.valid == 1) {
		c *= sampleTexture(textureArray, itsct.uv, mat.diffuseTexture);
	}

	return c * D * G * F /
		(4 * cosThetaI * cosThetaO);
}

__host__ __device__
Float microfaceBRDF_pdf(const MicroDistribution& distrib, const vc3& wo, const vc3& wi, const vc3& normal) {

	if (!SameHemiSphere(wo, wi)) {
		return 0;
	}
	vc3 wh = glm::normalize(wo + wi);
	return distribution_pdf(distrib, wo, wh, normal) / (4. * glm::dot(wo, wh));
}

__host__ __device__
vc3 microfaceBRDF_sample_f(
	const vc3& wo, vc3& wi,
	const ShadeableIntersection& itsct,
	const Material& mat,
	Float& pdf, 
	vc3* textureArray,
	thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	vc2 xi(u01(rng), u01(rng));
	vc3 wh = distribution_sample_wh(mat.dist, wo, xi);
	wi = glm::reflect(-wo, wh);
	// test if on the sample hemisphere
	if (glm::dot(wi, wo) < 0) {
		return vc3(0.);
	}
	
	float3 f_wh = make_float3(wh.x, wh.y, wh.z);
	pdf = distribution_pdf(mat.dist, wo, wh, itsct.surfaceNormal) / (4. * glm::dot(wo, wh));
	return microfaceBRDF_f(wo, wi, itsct, mat, textureArray);
}


