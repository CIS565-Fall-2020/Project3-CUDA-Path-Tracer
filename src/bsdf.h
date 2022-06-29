#pragma once

#include "intersections.h"
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/norm.hpp>

__host__ __device__
bool SameHemiSphere(const vc3& a, const vc3& b) {
	return glm::dot(a, b) > 0;
}

__host__ __device__
vc3 fresnel_evalulate(Float cosThetaI) {
	return vc3(0.);
}

__host__ __device__
vc3 distribution_sample_wh(const MicroDistribution& dist, const vc3& wo, const vc2& xi) {
	return vc3(0.);
}

__host__ __device__
Float distribution_pdf(const MicroDistribution& dist, const vc3& wo, vc3& wi) {
	return 0.;
}

__host__ __device__
Float distribution_D(const MicroDistribution& distrib, const vc3& wh) {
	return 0.;
}

__host__ __device__
Float distribution_G(const MicroDistribution& distrib, const vc3& wo, const vc3& wi) {
	return 0.;
}

__host__ __device__
vc3 microfaceBRDF_f(
	const MicroDistribution& distrib,
	const vc3& wo, const vc3& wi,
	const vc3& n) {
	Float cosThetaO = glm::dot(wo, n), cosTheta1 = glm::dot(wi, n);
	vc3 wh = wo + wi;
	// Handle degenerate cases for microfacet reflection
	/*if (cosThetaI == 0 || cosThetaO == 0) return Color3f(0.f);
	if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Color3f(0.f);*/

	wh = glm::normalize(wh);
	vc3 F = fresnel_evalulate(glm::dot(wi, wh));
	Float D = distribution_D(distrib, wh);
	Float G = distribution_G(distrib, wo, wi);
	return vc3(0.);
}

__host__ __device__
Float microfaceBRDF_pdf(const MicroDistribution& distrib, const vc3& wo, const vc3& wi) {

	if (!SameHemiSphere(wo, wi)) {
		return 0;
	}
	vc3 wh = glm::normalize(wo + wi);
	return distribution_pdf(distrib, wo, wh) / (4. * glm::dot(wo, wh));
}

__host__ __device__
vc3 microfaceBRDF_sample_f(
	const MicroDistribution& distrib,
	const vc3& wo, vc3& wi, 
	const vc3& normal,
	Float& pdf, 
	thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	vc2 xi(u01(rng), u01(rng));
	vc3 wh = distribution_sample_wh(distrib, wo, xi);
	wi = glm::reflect(-wo, wh);
	// test if on the sample hemisphere
	if (glm::dot(wi, wo) < 0) {
		return vc3(0.);
	}
	
	pdf = distribution_pdf(distrib, wo, wh) / (4. * glm::dot(wo, wh));
	return microfaceBRDF_f(distrib, wo, wi, normal);
}


