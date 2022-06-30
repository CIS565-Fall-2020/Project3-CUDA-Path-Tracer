#pragma once

#include "intersections.h"
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/norm.hpp>
#include "interactions.h"
#include "cudaUtils.cuh"

__host__ __device__
Float sinPhi(const vc3& a, const vc3& normal) {
	// x / sinTheta
	Float sinTheta = sqrt(thrust::max((Float)0, 1 - glm::dot(a, normal) * glm::dot(a, normal)));

	vc3 x, y;
	CoordinateSys(normal, x, y);

	return (sinTheta == 0) ? 1 : glm::clamp(glm::dot(a, y) / sinTheta, -1.f, 1.f);
}

__host__ __device__
Float cosPhi(const vc3& a, const vc3& normal) {
	// x / sinTheta
	Float sinTheta = sqrt(thrust::max((Float)0, 1 - glm::dot(a, normal) * glm::dot(a, normal)));
	
	vc3 x, y;
	CoordinateSys(normal, x, y);

	return (sinTheta == 0) ? 1 : glm::clamp(glm::dot(a, x) / sinTheta, -1.f, 1.f);
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

	if (distrib.type == TrowbridgeReitz) {
		vc3 wh;
		float cosTheta = 0, phi = (2 * glm::pi<Float>()) * xi[1];
		if (distrib.alpha.x == distrib.alpha.y) {
			float tanTheta2 = distrib.alpha.x * distrib.alpha.x * xi[0] / (1.0f - xi[0]);
			cosTheta = 1 / sqrt(1 + tanTheta2);
		}
		else {
			phi =
				atan(distrib.alpha.y / distrib.alpha.x * tan(2 * glm::pi<Float>() * xi[1] + .5f * glm::pi<Float>()));
			if (xi[1] > .5f) phi += glm::pi<Float>();
			float sinPhi = sin(phi), cosPhi = cos(phi);
			const float alphax2 = distrib.alpha.x * distrib.alpha.x, alphay2 = distrib.alpha.y * distrib.alpha.y;
			const float alpha2 =
				1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
			float tanTheta2 = alpha2 * xi[0] / (1 - xi[0]);
			cosTheta = 1 / sqrt(1 + tanTheta2);
		}
		float sinTheta =
			sqrt(max((float)0., (float)1. - cosTheta * cosTheta));

		wh = vc3(sinTheta * cos(phi), sinTheta * sin(phi),
			cosTheta);
		if (!SameHemiSphere(wo, wh)) wh = -wh;

		return wh;
	}
	

	

	return vc3(0.);
}

__host__ __device__
Float Lambda(const MicroDistribution& distrib, const vc3& wh, const vc3& normal) {
	// Shadowing - masking functions, which measures 
	// invisible masked microfacet area per visible microfacet area.
	if (distrib.type == TrowbridgeReitz) {
		Float cosTheta = glm::dot(wh, normal);
		Float cos2Theta = cosTheta * cosTheta;
		Float tan2Theta = thrust::max(0., 1. - cos2Theta) / cos2Theta;
		Float absTanTheta = sqrt(tan2Theta);
		if (isfinite(absTanTheta)) return 0.;

		// Compute alpha for direction w
		Float alpha =
			sqrt(cosPhi(wh, normal) * cosPhi(wh, normal) * distrib.alpha.x * distrib.alpha.x + sinPhi(wh, normal) * sinPhi(wh, normal) * distrib.alpha.y * distrib.alpha.y);
		Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
		return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
	}
	return 0;
}

__host__ __device__
Float distribution_D(const MicroDistribution& distrib, const vc3& wh, const vc3& normal) {
	// normal disribution
	if (distrib.type == TrowbridgeReitz) {
		Float cosTheta = glm::dot(wh, normal);
		Float cos2Theta = cosTheta * cosTheta;
		Float tan2Theta = thrust::max(0., 1. - cos2Theta) / cos2Theta;
		if (isfinite(tan2Theta)) { // handle infinity and Nan 
			return (Float)0.;
		}
		Float cos4Theta = cos2Theta * cos2Theta;
		Float e = (cosPhi(wh, normal) * cosPhi(wh, normal) / (distrib.alpha.x * distrib.alpha.x) +
			sinPhi(wh, normal) * sinPhi(wh, normal) / (distrib.alpha.y * distrib.alpha.y)) / tan2Theta;

		return 1. / (glm::pi<Float>() * distrib.alpha.x * distrib.alpha.y * cos4Theta * (1 + e) * (1 + e));
	}
	return 0.;
}

__host__ __device__
Float distribution_pdf(const MicroDistribution& distrib, const vc3& wo, const vc3& wh, const vc3& normal) {
	return distribution_D(distrib, wh, normal) * glm::abs(glm::dot(wh, normal)) ;
}

__host__ __device__
Float distribution_G(const MicroDistribution& distrib, const vc3& wo, const vc3& wi, const vc3& normal) {
	return 1. / (1. + Lambda(distrib, wo, normal) + Lambda(distrib, wi, normal));
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
	/*if (cosThetaI == 0 || cosThetaO == 0) return Color3f(0.f);
	if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Color3f(0.f);*/

	wh = glm::normalize(wh);
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


