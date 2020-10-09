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
__host__ __device__ float smithGGgx(float cosOut, float alphaG) {
	float a = alphaG * alphaG, b = cosOut * cosOut;
	return 1.0f / (cosOut + glm::sqrt(a + b - a * b));
}

__host__ __device__ glm::vec3 disneyBrdfTint(glm::vec3 linearColor) {
	float luminance = 0.3f * linearColor.r + 0.6f * linearColor.g + 0.1f * linearColor.b;
	return luminance > 0.0f ? linearColor / luminance : glm::vec3(1.0f);
}

__host__ __device__ float disneyDiffuseWeight(const Material &m) {
	/*return 1.0f / (glm::max(1.0f - m.disney.roughness, m.disney.clearcoat) + 1.0f);*/
	return 0.5f;
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
	float gSpecular = smithGGgx(cosIn, a) * smithGGgx(cosOut, a);

	*color = fSpecular * gSpecular;
	*density = dSpecular;
}
__host__ __device__ void disneyBrdfIsotropicClearcoat(
	float cosIn, float cosOut, float cosHalf, const Material &m, float fresnelHalf,
	float *color, float *density
) {
	// clearcoat
	float dClearcoat = gtr1(cosHalf, glm::mix(0.1f, 0.001f, m.disney.clearcoatGloss));
	float fClearcoat = glm::mix(0.04f, 1.0f, fresnelHalf);
	float gClearcoat = smithGGgx(cosIn, 0.25f) * smithGGgx(cosOut, 0.25f);

	// the disney implmentation multiplies it by 0.25 here which results in clearcoat being too weak
	*color = m.disney.clearcoat * gClearcoat * fClearcoat;
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
	glm::vec3 tint = disneyBrdfTint(m.baseColorLinear);

	glm::vec3 specularColor;
	float specularDensity, clearcoatColor, clearcoatDensity;
	disneyBrdfIsotropicSpecular(cosIn, cosOut, cosHalf, m, tint, fresnelHalf, &specularColor, &specularDensity);
	disneyBrdfIsotropicClearcoat(cosIn, cosOut, cosHalf, m, fresnelHalf, &clearcoatColor, &clearcoatDensity);

	return
		disneyBrdfIsotropicDiffuse(cosIn, cosOut, inDotHalf, m, tint, fresnelHalf) +
		specularColor * specularDensity + clearcoatColor * clearcoatDensity;
}

__host__ __device__ glm::vec3 sampleDisneySpecularHalf(const Material &m, glm::vec3 normal, glm::vec2 rand) {
	float specularWeight = 1.0f / (m.disney.clearcoat + 1.0f);
	glm::vec3 x = glm::normalize(glm::cross(crossDirection(normal), normal)), y = glm::cross(normal, x);
	glm::vec3 resultHalf;
	if (rand.x < specularWeight) {
		rand.x /= specularWeight;

		float g = glm::sqrt(rand.y / (1.0f - rand.y)) * m.disney.roughness * m.disney.roughness;
		float phi = 2.0f * glm::pi<float>() * rand.x;
		
		resultHalf = glm::normalize(normal + g * (glm::cos(phi) * x + glm::sin(phi) * y));
	} else {
		rand.x = (rand.x - specularWeight) / (1.0f - specularWeight);

		// sample GTR1 direction
		float phiH = 2.0f * rand.x * glm::pi<float>();
		float roughness = glm::mix(0.1f, 0.001f, m.disney.clearcoatGloss);
		float roughness2 = roughness * roughness;
		float cosThetaH = glm::sqrt(
			roughness2 >= 1.0f ?
			1.0f - rand.y :
			(1.0f - glm::pow(roughness2, 1.0f - rand.y)) / (1.0f - roughness2)
		);

		resultHalf =
			normal * cosThetaH +
			(glm::cos(phiH) * x + glm::sin(phiH) * y) * glm::sqrt(1.0f - cosThetaH * cosThetaH);
	}

	return resultHalf;
}


__host__ __device__ bool sampleBsdf(
	glm::vec3 lightOut, glm::vec3 shadeNormal, Material m, bool backface,
	glm::vec2 rand, glm::vec3 *lightInRes, glm::vec3 *bsdf, float *pdf
) {
	glm::vec3 lightIn;
	switch (m.type) {
	case MaterialType::specularReflection:
		lightIn = glm::reflect(-lightOut, shadeNormal);
		*bsdf = m.baseColorLinear / glm::abs(glm::dot(shadeNormal, lightIn));
		*pdf = 1.0f;
		break;
	case MaterialType::specularTransmission:
		{
			float ior = m.specularTransmission.indexOfRefraction;
			if (!backface) {
				ior = 1.0f / ior;
			}
			float cosOut = glm::dot(shadeNormal, lightOut);

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
				lightIn = glm::reflect(-lightOut, shadeNormal);
			} else { // refract
				lightIn = glm::refract(-lightOut, shadeNormal, ior);
			}
			*bsdf = m.baseColorLinear * fresnel / glm::abs(glm::dot(shadeNormal, lightIn));
			*pdf = fresnel;
		}
		break;

	case MaterialType::diffuse:
		lightIn = sampleHemisphereCosine(shadeNormal, rand);
		*bsdf = m.baseColorLinear / glm::pi<float>();
		*pdf = glm::dot(shadeNormal, lightIn) / glm::pi<float>();
		break;
	case MaterialType::disney:
		{
			glm::vec3 half;
			float diffuseWeight = disneyDiffuseWeight(m);

			if (rand.x < diffuseWeight) { // diffuse
				rand.x /= diffuseWeight;
				lightIn = sampleHemisphereCosine(shadeNormal, rand);
				half = glm::normalize(lightIn + lightOut);
			} else { // specular
				rand.x = (rand.x - diffuseWeight) / (1.0f - diffuseWeight);
				half = sampleDisneySpecularHalf(m, shadeNormal, rand);
				lightIn = glm::reflect(-lightOut, half);
			}

			float cosOut = glm::abs(glm::dot(shadeNormal, lightOut)), cosIn = glm::dot(shadeNormal, lightIn);
			if (cosIn < 0.0f) {
				return false;
			}
			float
				cosHalf = glm::dot(half, shadeNormal),
				inDotHalf = glm::dot(lightIn, half);

			float fresnelHalf = schlickFresnel(inDotHalf);
			glm::vec3 tint = disneyBrdfTint(m.baseColorLinear);

			glm::vec3 specularColor;
			float specularDensity, clearcoatColor, clearcoatDensity;
			glm::vec3 diffuse = disneyBrdfIsotropicDiffuse(cosIn, cosOut, inDotHalf, m, tint, fresnelHalf);
			disneyBrdfIsotropicSpecular(
				cosIn, cosOut, cosHalf, m, tint, fresnelHalf, &specularColor, &specularDensity
			);
			disneyBrdfIsotropicClearcoat(
				cosIn, cosOut, cosHalf, m, fresnelHalf, &clearcoatColor, &clearcoatDensity
			);

			*bsdf = diffuse + specularColor * specularDensity + clearcoatColor * clearcoatDensity;
			*pdf = glm::mix(
				glm::mix(
					clearcoatDensity, specularDensity, 1.0f / (m.disney.clearcoat + 1.0f)
				) * cosHalf * 0.25f / inDotHalf,
				cosIn / glm::pi<float>(),
				diffuseWeight
			);
		}
		break;
	}
	*lightInRes = lightIn;
	return true;
}

__host__ __device__ void evalBsdf(
	glm::vec3 lightIn, glm::vec3 shadeNormal, glm::vec3 lightOut, const Material &m, bool backface,
	glm::vec3 *bsdf, float *pdf
) {
	switch (m.type) {
	case MaterialType::diffuse:
		{
			float cosIn = glm::max(glm::dot(shadeNormal, lightIn), 0.0f);
			*bsdf = m.baseColorLinear * cosIn / glm::pi<float>();
			*pdf = cosIn / glm::pi<float>();
		}
		break;
	case MaterialType::disney:
		{
			float cosOut = glm::abs(glm::dot(shadeNormal, lightOut)), cosIn = glm::dot(shadeNormal, lightIn);
			if (cosIn < 0.0f) {
				*bsdf = glm::vec3(0.0f);
				*pdf = 0.0f;
				return;
			}
			glm::vec3 half = glm::normalize(lightIn + lightOut);
			float
				cosHalf = glm::dot(half, shadeNormal),
				inDotHalf = glm::dot(lightIn, half);

			float fresnelHalf = schlickFresnel(inDotHalf);
			glm::vec3 tint = disneyBrdfTint(m.baseColorLinear);

			glm::vec3 specularColor;
			float specularDensity, clearcoatColor, clearcoatDensity;
			glm::vec3 diffuse = disneyBrdfIsotropicDiffuse(cosIn, cosOut, inDotHalf, m, tint, fresnelHalf);
			disneyBrdfIsotropicSpecular(
				cosIn, cosOut, cosHalf, m, tint, fresnelHalf, &specularColor, &specularDensity
			);
			disneyBrdfIsotropicClearcoat(
				cosIn, cosOut, cosHalf, m, fresnelHalf, &clearcoatColor, &clearcoatDensity
			);

			*bsdf = diffuse + specularColor * specularDensity + clearcoatColor * clearcoatDensity;
			*pdf = glm::mix(
				glm::mix(
					clearcoatDensity, specularDensity, 1.0f / (m.disney.clearcoat + 1.0f)
				) * cosHalf * 0.25f / inDotHalf,
				cosIn / glm::pi<float>(),
				disneyDiffuseWeight(m)
			);
		}
		break;

	default:
		*bsdf = glm::vec3(0.0f);
		*pdf = 0.0f;
		break;
	}
}


__host__ __device__ int sceneIntersection(
	glm::vec3 p1, glm::vec3 p2, int e1, int e2,
	const Geom *geoms, const AABBTreeNode *tree, int treeRoot
) {
	float dist = 1.0f;
	Ray ray;
	ray.origin = p1;
	ray.direction = p2 - p1;
	glm::vec3 normalTokenUnused;
	return traverseAABBTree(ray, tree, treeRoot, geoms, e1, e2, &dist, &normalTokenUnused);
}
__host__ __device__ float balanceHeuristic(float pdf1, float pdf2) {
	return pdf1 / (pdf1 + pdf2);
}
__host__ __device__ float powerHeuristic(float pdf1, float pdf2) {
	pdf1 *= pdf1;
	pdf2 *= pdf2;
	return pdf1 / (pdf1 + pdf2);
}
__host__ __device__ void multipleImportanceSampling(
	PathSegment &path, glm::vec3 intersect, glm::vec3 geomNormal, glm::vec3 shadeNormal, const Material &m,
	glm::vec2 rand1, glm::vec2 rand2, int geomId, int numMisLights,
	const Geom *geoms, const Material *mats, const AABBTreeNode *tree, int treeRoot
) {
	if (m.type == MaterialType::emissive) {
		return;
	}

	const Geom &geom = geoms[geomId];
	glm::vec3 emission = mats[geom.materialid].baseColorLinear;

	bool backface = glm::dot(geomNormal, path.ray.direction) > 0.0f;
	if (backface) {
		shadeNormal = -shadeNormal;
	}

	glm::vec3 lightNormal = glm::cross(
		geom.triangle.vertices[1] - geom.triangle.vertices[0],
		geom.triangle.vertices[2] - geom.triangle.vertices[0]
	);
	float lightArea = glm::length(lightNormal);
	lightNormal /= lightArea;
	lightArea *= 0.5f;

	// sample light
	glm::vec3 lteLight(0.0f);
	float bsdfPdfLight = 0.0f, lightPdfLight = 0.0f;
	if (rand1.x + rand1.y > 1.0f) {
		rand1 = 1.0f - rand1;
	}
	glm::vec3 lightSample =
		geom.triangle.vertices[0] * rand1.x +
		geom.triangle.vertices[1] * rand1.y +
		geom.triangle.vertices[2] * (1.0f - rand1.x - rand1.y);
	if (sceneIntersection(intersect, lightSample, path.lastGeom, geomId, geoms, tree, treeRoot) == -1) {
		glm::vec3 lightInLight = lightSample - intersect;
		float sqrDistance = glm::length2(lightInLight);
		lightInLight = glm::normalize(lightInLight);
		glm::vec3 bsdf;
		evalBsdf(lightInLight, shadeNormal, -path.ray.direction, m, backface, &bsdf, &bsdfPdfLight);
		lightPdfLight = sqrDistance / (glm::abs(glm::dot(lightInLight, lightNormal)) * lightArea);
		lteLight = emission * bsdf * glm::abs(glm::dot(shadeNormal, lightInLight)) / lightPdfLight;
	}

	// sample bsdf
	glm::vec3 lteBsdf(0.0f), lightInBsdf, bsdfBsdf;
	float bsdfPdfBsdf = 0.0f, lightPdfBsdf = 0.0f;
	if (sampleBsdf(-path.ray.direction, shadeNormal, m, backface, rand2, &lightInBsdf, &bsdfBsdf, &bsdfPdfBsdf)) {
		Ray ray;
		ray.origin = intersect;
		ray.direction = lightInBsdf;
		glm::vec3 normalTokenUnused;
		float dist = FLT_MAX;
		int geomHit = traverseAABBTree(ray, tree, treeRoot, geoms, path.lastGeom, -1, &dist, &normalTokenUnused);
		if (geomHit == geomId) {
			lightPdfBsdf = dist * dist / (glm::abs(glm::dot(lightInBsdf, lightNormal)) * lightArea);
			lteBsdf = emission * bsdfBsdf * glm::abs(glm::dot(shadeNormal, lightInBsdf)) / bsdfPdfBsdf;
		}
	}

	if (bsdfPdfLight + lightPdfLight > 1e-6) {
		lteLight *= powerHeuristic(lightPdfLight, bsdfPdfLight);
	}
	if (bsdfPdfBsdf + lightPdfBsdf > 1e-6) {
		lteBsdf *= powerHeuristic(bsdfPdfBsdf, lightPdfBsdf);
	}

	path.colorAccum += path.colorThroughput * static_cast<float>(numMisLights) * (lteLight + lteBsdf);
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
	PathSegment &path, glm::vec3 intersect,
	glm::vec3 geomNormal, glm::vec3 shadeNormal, const Material &m,
	glm::vec2 rand, bool includeDirectLighting
) {
	path.ray.origin = intersect;

	if (m.type == MaterialType::emissive) {
		if (includeDirectLighting || m.emitterNoMis || path.prevBounceNoMis) {
			path.colorAccum += m.baseColorLinear * path.colorThroughput;
		}
		path.remainingBounces = 0;
	} else {
		--path.remainingBounces;
		bool backface = glm::dot(geomNormal, path.ray.direction) > 0.0f;
		if (backface) {
			shadeNormal = -shadeNormal;
		}

		glm::vec3 brdf, lightIn;
		float pdf;
		if (!sampleBsdf(-path.ray.direction, shadeNormal, m, backface, rand, &lightIn, &brdf, &pdf)) {
			path.remainingBounces = 0;
		}
		path.ray.direction = lightIn;
		path.colorThroughput *= brdf * glm::abs(glm::dot(shadeNormal, path.ray.direction)) / pdf;
	}
}
