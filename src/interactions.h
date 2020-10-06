#pragma once

#include "intersections.h"

__host__ __device__ glm::vec3 crossDirection(glm::vec3 v) {
	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.
	if (abs(v.x) < SQRT_OF_ONE_THIRD) {
		return glm::vec3(1, 0, 0);
	} else if (abs(v.y) < SQRT_OF_ONE_THIRD) {
		return glm::vec3(0, 1, 0);
	}
	return glm::vec3(0, 0, 1);
}
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 sampleHemisphereCosine(glm::vec3 normal, glm::vec2 rand) {
	float up = sqrt(rand.x); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = rand.y * TWO_PI;

	glm::vec3 directionNotNormal = crossDirection(normal);

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 = glm::cross(normal, perpendicularDirection1);

	return
		up * normal + over * (
			cos(around) * perpendicularDirection1 +
			sin(around) * perpendicularDirection2
		);
}

__host__ __device__ glm::vec2 sampleUnitDiskUniform(glm::vec2 rand) {
	float angle = rand.x * 2.0f * glm::pi<float>(), radius = glm::sqrt(rand.y);
	return glm::vec2(glm::cos(angle), glm::sin(angle)) * radius;
}

__host__ __device__ float schlickFresnel(float cos) {
	float m = glm::clamp(1.0f - cos, 0.0f, 1.0f);
	float sqm = m * m;
	return sqm * sqm * m;
}
__host__ __device__ float gtr1(float cosHalf, float a) {
	float a2 = a * a;
	float t = 1.0f + (a2 - 1.0f) * cosHalf * cosHalf;
	return (a2 - 1.0f) / (glm::pi<float>() * glm::log(a2) * t);
}
__host__ __device__ float gtr2(float cosHalf, float a) {
	float a2 = a * a;
	float t = 1.0f + (a2 - 1.0f) * cosHalf * cosHalf;
	return a2 / (glm::pi<float>() * t * t);
}
__host__ __device__ float smithGGgxSpecular(float cosOut, float a) {
	float sqrA = a * a, sqrCosOut = cosOut * cosOut;
	return 1.0f / (cosOut + glm::sqrt(sqrCosOut + (1.0 - sqrCosOut) / sqrA));
}
__host__ __device__ float smithGGgxClearcoat(float cosOut, float alphaG) {
	float a = alphaG * alphaG, b = cosOut * cosOut;
	return 1.0f / (cosOut + glm::sqrt(a + b - a * b));
}

__host__ __device__ glm::vec3 disneyBrdfIsotropicDiffuse(
	float cosIn, float cosOut, float inDotHalf, const Material &m, glm::vec3 tint, float fresnelHalf
) {
	// diffuse fresnel
	float fresnelIn = schlickFresnel(cosIn), fresnelOut = schlickFresnel(cosOut);
	float fresnelDiffuse90 = 0.5f + 2.0f * inDotHalf * inDotHalf * m.disney.roughness;
	float fresnelDiffuse =
		glm::mix(1.0f, fresnelDiffuse90, fresnelIn) * glm::mix(1.0f, fresnelDiffuse90, fresnelOut);

	// sheen
	glm::vec3 sheen = glm::mix(glm::vec3(1.0f), tint, m.disney.sheenTint);
	glm::vec3 fSheen = fresnelHalf * m.disney.sheen * sheen;

	return ((1.0f / glm::pi<float>()) * fresnelDiffuse * m.baseColorLinear + fSheen) * (1.0f - m.disney.metallic);
}
__host__ __device__ void disneyBrdfIsotropicSpecular(
	float cosIn, float cosOut, float cosHalf, const Material &m, glm::vec3 tint, float fresnelHalf,
	glm::vec3 *color, float *density
) {
	// specular
	glm::vec3 spec0 = glm::mix(
		m.disney.specular * 0.08f * glm::mix(glm::vec3(1.0f), tint, m.disney.specularTint),
		m.baseColorLinear,
		m.disney.metallic
	);
	float a = glm::max(0.001f, m.disney.roughness * m.disney.roughness);
	float dSpecular = gtr2(cosHalf, a);
	glm::vec3 fSpecular = glm::mix(spec0, glm::vec3(1.0f), fresnelHalf);
	float gSpecular = smithGGgxSpecular(cosIn, a) * smithGGgxSpecular(cosOut, a);

	*color = fSpecular * gSpecular;
	*density = dSpecular;
}
__host__ __device__ void disneyBrdfIsotropicClearcoat(
	float cosIn, float cosOut, float cosHalf, const Material &m, float fresnelHalf,
	float *color, float *density
) {
	// clearcoat, IOR = 1.5
	float dClearcoat = gtr1(cosHalf, glm::mix(0.1f, 0.001f, m.disney.clearCoatGloss));
	float fClearcoat = glm::mix(0.04f, 1.0f, fresnelHalf);
	float gClearcoat = smithGGgxClearcoat(cosIn, 0.25f) * smithGGgxClearcoat(cosOut, 0.25f);

	*color = 0.25f * m.disney.clearCoat * gClearcoat * fClearcoat;
	*density = dClearcoat;
}

__host__ __device__ glm::vec3 disneyBrdfIsotropic(
	glm::vec3 lightOut, glm::vec3 normal, glm::vec3 lightIn, const Material &m
) {
	float cosIn = glm::dot(normal, lightIn), cosOut = glm::dot(normal, lightOut);
	if (cosIn < 0.0f || cosOut < 0.0f) {
		return glm::vec3(0.0f);
	}

	glm::vec3 halfVec = glm::normalize(lightIn + lightOut);
	float cosHalf = glm::dot(halfVec, normal), inDotHalf = glm::dot(lightIn, halfVec);

	float fresnelHalf = schlickFresnel(inDotHalf);
	float luminance = 0.3f * m.baseColorLinear.r + 0.6f * m.baseColorLinear.g + 0.1f * m.baseColorLinear.b;
	glm::vec3 tint = luminance > 0.0f ? m.baseColorLinear / luminance : glm::vec3(1.0f);

	glm::vec3 specularColor;
	float specularDensity, clearcoatColor, clearcoatDensity;
	disneyBrdfIsotropicSpecular(cosIn, cosOut, cosHalf, m, tint, fresnelHalf, &specularColor, &specularDensity);
	disneyBrdfIsotropicClearcoat(cosIn, cosOut, cosHalf, m, fresnelHalf, &clearcoatColor, &clearcoatDensity);

	return
		disneyBrdfIsotropicDiffuse(cosIn, cosOut, inDotHalf, m, tint, fresnelHalf) +
		specularColor * specularDensity + clearcoatColor * clearcoatDensity;
}

__host__ __device__ glm::vec3 sampleDisneySpecular(
	const Material &m, glm::vec3 rayDir, glm::vec3 normal, glm::vec2 rand, float *pdf
) {
	float gtr2Weight = 1.0f / (m.disney.clearCoat + 1.0f);
	glm::vec3 x = glm::normalize(glm::cross(crossDirection(normal), normal)), y = glm::cross(normal, x);
	glm::vec3 resultHalf;
	bool isSpecular = rand.x < gtr2Weight;
	if (isSpecular) {
		rand.x /= gtr2Weight;

		float g = glm::sqrt(rand.y / (1.0f - rand.y)) * m.disney.roughness * m.disney.roughness;
		float phi = 2.0f * glm::pi<float>() * rand.x;
		
		resultHalf = glm::normalize(normal + g * (glm::cos(phi) * x + glm::sin(phi) * y));
	} else {
		rand.x = (rand.x - gtr2Weight) / (1.0f - gtr2Weight);

		// sample GTR1 direction
		float phiH = 2.0f * rand.x * glm::pi<float>();
		float roughness2 = m.disney.roughness * m.disney.roughness;
		float cosThetaH = glm::sqrt(
			roughness2 >= 1.0f ?
			1.0f - rand.y :
			(1.0f - glm::pow(roughness2, 1.0f - rand.y)) / (1.0f - roughness2)
		);

		resultHalf =
			normal * cosThetaH +
			(glm::cos(phiH) * x + glm::sin(phiH) * y) * glm::sqrt(1.0f - cosThetaH * cosThetaH);
	}

	glm::vec3 result = glm::reflect(rayDir, resultHalf);
	float inDotHalf = glm::dot(result, resultHalf), cosHalf = glm::dot(normal, resultHalf);
	float d = glm::mix(
		gtr1(cosHalf, glm::mix(0.1f, 0.001f, m.disney.clearCoatGloss)),
		gtr2(cosHalf, glm::max(0.001f, m.disney.roughness * m.disney.roughness)),
		gtr2Weight
	);
	*pdf = d * cosHalf * 0.25f / inDotHalf;

	return result;
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
__host__ __device__ void scatterRay(
	PathSegment &path, glm::vec3 intersect, glm::vec3 geomNormal, glm::vec3 shadeNormal,
	Material m, glm::vec2 rand
) {
	path.ray.origin = intersect;

	if (m.type == MaterialType::emitter) {
		path.color *= m.baseColorLinear;
		path.remainingBounces = -1;
	} else {
		--path.remainingBounces;
		bool backface = glm::dot(geomNormal, path.ray.direction) > 0.0f;
		if (backface) {
			shadeNormal = -shadeNormal;
		}
		switch (m.type) {
		case MaterialType::diffuse:
			path.color *= m.baseColorLinear;
			path.ray.direction = sampleHemisphereCosine(shadeNormal, rand);
			break;
		case MaterialType::specularReflection:
			path.color *= m.baseColorLinear;
			path.ray.direction = glm::reflect(path.ray.direction, shadeNormal);
			break;
		case MaterialType::specularTransmission:
			{
				path.color *= m.baseColorLinear;
				float ior = m.specularTransmission.indexOfRefraction;
				if (!backface) {
					ior = 1.0f / ior;
				}
				float cosOut = -glm::dot(shadeNormal, path.ray.direction);

				float fresnel = 1.0f;
				float sinOut = glm::sqrt(1.0f - cosOut * cosOut);
				float sinIn = sinOut * ior;
				if (sinIn <= 1.0f) {
					float cosIn = glm::sqrt(1.0f - sinIn * sinIn);
					float rParl = (cosOut - ior * cosIn) / (cosOut + ior * cosIn);
					float rPerp = (ior * cosOut - cosIn) / (ior * cosOut + cosIn);
					fresnel = 0.5f * (rPerp * rPerp + rParl * rParl);
				}

				if (rand.x < fresnel) { // reflect
					path.ray.direction = glm::reflect(path.ray.direction, shadeNormal);
				} else { // refract
					path.ray.direction = glm::normalize(glm::refract(path.ray.direction, shadeNormal, ior));
				}
			}
			break;
		case MaterialType::disney:
			{
				glm::vec3 lightOut = -path.ray.direction;
				float diffuseWeight = 1.0f; /*glm::max(m.disney.roughness, 1.0f - m.disney.clearCoat);*/
				if (rand.x < diffuseWeight) { // diffuse
					rand.x /= diffuseWeight;
					path.ray.direction = sampleHemisphereCosine(shadeNormal, rand);
					path.color *= glm::pi<float>() / diffuseWeight;
				} else { // specular
					rand.x = (rand.x - diffuseWeight) / (1.0f - diffuseWeight);
					float pdf = 1.0f;
					path.ray.direction = sampleDisneySpecular(m, path.ray.direction, shadeNormal, rand, &pdf);
					path.color *= glm::dot(path.ray.direction, shadeNormal) / (pdf * (1.0f - diffuseWeight));
				}
				path.color *= disneyBrdfIsotropic(lightOut, shadeNormal, path.ray.direction, m);

				/*path.color = (path.ray.direction + 1.0f) * 0.5f;
				path.remainingBounces = -1;*/
			}
			break;
		}
		path.ray.origin += 0.01f * path.ray.direction;
	}
}
