#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal, thrust::default_random_engine& rng) {

	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

//Helper for Refraction 
__host__ __device__ glm::vec3 getRefractedDirection(PathSegment& pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material& m,
	thrust::default_random_engine&,
	bool& flipDir)
{
	float eta = 1.f / m.indexOfRefraction;
	float criticalAngle(0.f), theta(0.f);
	//Set the index of refraction and check for Total Internal Reflection 
	glm::vec3 rayDirection = pathSegment.ray.direction;
	glm::vec3 refractedDir(0.f);
	float cos_theta = glm::dot(rayDirection, normal);

	//Entering the object
	if (cos_theta <= 0.f)
	{
		theta = acos(glm::dot(-rayDirection, normal));
		criticalAngle = asin(eta);
		//Total internal reflection? 
		if (eta < 1.f && theta > criticalAngle)
		{
			refractedDir = glm::normalize(glm::reflect(rayDirection, -normal));
			flipDir = false;
		}
		else
		{
			refractedDir = glm::normalize(glm::refract(rayDirection, normal, eta));
			flipDir = true;
		}
	}
	//Exiting the object 
	else
	{
		theta = acos(glm::dot(rayDirection, normal));
		criticalAngle = asin(m.indexOfRefraction);
		//Total internal reflection? 
		if (m.indexOfRefraction < 1.f && theta > criticalAngle)
		{
			refractedDir = glm::normalize(glm::reflect(rayDirection, -normal));
			flipDir = true;
		}
		else
		{
			refractedDir = glm::normalize(glm::refract(rayDirection, -normal, eta));
			flipDir = false;
		}
	}
	return refractedDir;
}


__host__ __device__ glm::vec3 refract(PathSegment& pathSegment, glm::vec3 normal, const Material& m, thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1); 
	float R = 1.f; 
	float ior; 
	glm::vec3 sur_normal; 

	float cosi = glm::dot(pathSegment.ray.direction, normal); 
	if (cosi > 0.f)
	{
		ior = m.indexOfRefraction; 
		sur_normal = -normal; 
	}
	else
	{
		ior = 1.f / m.indexOfRefraction; 
		sur_normal = normal; 
	}

	//Total Internal Reflection 
	float sinr = (1.f / m.indexOfRefraction) * sqrtf(1.f - powf(cosi, 2)); 
	

	//Ray incoming 
	if (cosi > 0.f && sinr > 1.f)
	{
		R = 0.f; 
		pathSegment.color *= 0.f; 
	}
	else
	{
		float r0 = powf((1.f - ior) / (1.f + ior), 2.f); 
		R = r0 + (1 - r0) * powf((1 - fmaxf(0.f, cosi)), 5.f); 
		pathSegment.color *= m.specular.color; 
	}
	float rand = u01(rng); 
	if (rand < R)
	{
		return glm::refract(pathSegment.ray.direction, sur_normal, ior); 
	}
	else
	{
		return glm::reflect(pathSegment.ray.direction, normal); 
	}
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


float find_max(float a, float b)
{
	if (a > b) return a;
	else return b; 
}

__host__ __device__
void scatterRay(
	PathSegment& pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material& m,
	thrust::default_random_engine& rng) 
{
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	float randDist = u01(rng);

	//Pure diffuse shading 
	if (m.hasReflective == 0.f && m.hasRefractive == 0.f)
	{
		glm::vec3 randDirInHemisphere = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.origin = intersect + (EPSILON * normal);
		pathSegment.ray.direction = glm::normalize(randDirInHemisphere);
		pathSegment.color *= m.color;
	}
	//Reflective surface 
	else if (m.hasReflective >= randDist && m.hasRefractive == 0.f)
	{
		glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal); //For the incident vector I and surface orientation N, 
																				  //returns the reflection direction
		pathSegment.ray.origin = intersect + (EPSILON * normal);
		pathSegment.ray.direction = glm::normalize(reflectedDir);
		pathSegment.color *= m.specular.color;
	}
	//Refractive surface 
	else if (m.hasReflective + m.hasRefractive >= 1.f)
	{
		//bool flipDir(0);
		//pathSegment.ray.direction = glm::normalize(getRefractedDirection(pathSegment, intersect, normal, m, rng, flipDir));
		//if (flipDir) 
		//{
		//	pathSegment.ray.origin = intersect + (EPSILON * -normal);
		//}
		//else 
		//{
		//	pathSegment.ray.origin = intersect + (EPSILON * normal);
		//}
		//pathSegment.color *= m.specular.color;

		//glm::vec3 refracted_dir = refract(pathSegment, normal, m, rng);
		//pathSegment.ray.origin = intersect + (EPSILON * refracted_dir);
		//pathSegment.ray.direction = refracted_dir; 

		//PBRT REFRACT 

		glm::vec3 dir = glm::normalize(pathSegment.ray.direction);
		float cosThetaI = glm::dot(normal, -dir);
		float eta = 1.00029f;
		if (cosThetaI > 0.f)
		{
			eta /= m.indexOfRefraction;
		}
		else
		{
			normal *= -1;
			eta = m.indexOfRefraction / eta;
		}

		float sinThetaI = glm::max(float(0.f), float(1.f - cosThetaI * cosThetaI));
		float sinThetaT = eta * eta * sinThetaI;

		//Handle total internal reflection
		if (sinThetaT >= 1)
		{
			pathSegment.color *= m.color;
			pathSegment.ray.direction = glm::reflect(dir, normal);
		}
		else
		{
			float cosThetaT = std::sqrt(1 - sinThetaT);
			//pathSegment.ray.direction = eta * pathSegment.ray.direction + (eta * cosThetaI - cosThetaT) * normal;
			pathSegment.ray.direction = glm::refract(glm::normalize(pathSegment.ray.direction),normal, eta);
			pathSegment.color *= m.color;
		}
	}
 }



//__host__ __device__
//void scatterRay(
//	PathSegment& pathSegment,
//	glm::vec3 intersect,
//	glm::vec3 normal,
//	const Material& m,
//	thrust::default_random_engine& rng) {
//	// TODO: implement this.
//	// A basic implementation of pure-diffuse shading will just call the
//	// calculateRandomDirectionInHemisphere defined above.
//
//	thrust::uniform_real_distribution<float> u01(0, 1);
//	float randDist = u01(rng);
//
//	//Pure diffuse shading 
//	if (m.hasReflective == 0.f && m.hasRefractive == 0.f)
//	{
//		glm::vec3 randDirInHemisphere = calculateRandomDirectionInHemisphere(normal, rng);
//		pathSegment.ray.origin = intersect + (EPSILON * normal);
//		pathSegment.ray.direction = glm::normalize(randDirInHemisphere);
//		pathSegment.color *= m.color;
//	}
//	//Reflective surface 
//	else if (m.hasReflective >= randDist && m.hasRefractive == 0.f)
//	{
//		glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal); //For the incident vector I and surface orientation N, 
//																				  //returns the reflection direction
//		pathSegment.ray.origin = intersect + (EPSILON * normal);
//		pathSegment.ray.direction = glm::normalize(reflectedDir);
//		pathSegment.color *= m.specular.color;
//	}
//	//Refractive surface 
//	else if (m.hasReflective + m.hasRefractive >= 1.f)
//	{
//		float eta = 1.f / m.indexOfRefraction;
//		float criticalAngle(0.f), theta(0.f);
//		//Set the index of refraction and check for Total Internal Reflection 
//		glm::vec3 rayDirection = pathSegment.ray.direction;
//		glm::vec3 refractedDir(0.f);
//		float cos_theta = glm::dot(rayDirection, normal);
//
//		//Entering the object
//		if (cos_theta <= 0.f)
//		{
//			theta = acos(glm::dot(-rayDirection, normal));
//			criticalAngle = asin(eta);
//			//Total internal reflection? 
//			if (eta < 1.f && theta > criticalAngle)
//			{
//				refractedDir = glm::normalize(glm::reflect(rayDirection, -normal));
//				pathSegment.ray.origin = intersect + (EPSILON * normal);
//			}
//			else
//			{
//				refractedDir = glm::normalize(glm::refract(rayDirection, normal, eta));
//				pathSegment.ray.origin = intersect + (EPSILON * -normal);
//			}
//		}
//		//Exiting the object 
//		else
//		{
//			theta = acos(glm::dot(rayDirection, normal));
//			criticalAngle = asin(m.indexOfRefraction);
//			//Total internal reflection? 
//			if (m.indexOfRefraction < 1.f && theta > criticalAngle)
//			{
//				refractedDir = glm::normalize(glm::reflect(rayDirection, -normal));
//				pathSegment.ray.origin = intersect + (EPSILON * -normal);
//			}
//			else
//			{
//				refractedDir = glm::normalize(glm::refract(rayDirection, -normal, eta));
//				pathSegment.ray.origin = intersect + (EPSILON * normal);
//			}
//		}
//		 
//		pathSegment.ray.direction = refractedDir;
//	}
//
//}



//PBRT REFRACTIVE 

//__host__ __device__
//void scatterRay(
//	PathSegment& pathSegment,
//	glm::vec3 intersect,
//	glm::vec3 normal,
//	const Material& m,
//	thrust::default_random_engine& rng) 
//{
//	// TODO: implement this.
//	// A basic implementation of pure-diffuse shading will just call the
//	// calculateRandomDirectionInHemisphere defined above.
//
//	thrust::uniform_real_distribution<float> u01(0, 1);
//	float randDist = u01(rng);
//
//	//Pure diffuse shading 
//	if (m.hasReflective <= 0.01f && m.hasRefractive <= 0.01f)
//	{
//		glm::vec3 randDirInHemisphere = calculateRandomDirectionInHemisphere(normal, rng);
//		pathSegment.ray.origin = intersect + (EPSILON * normal);
//		pathSegment.ray.direction = glm::normalize(randDirInHemisphere);
//		pathSegment.color *= m.color;
//	}
//
//	//Reflective surface 
//	else if (m.hasReflective >= 0.01f && m.hasRefractive <= 0.01f)
//	{
//		glm::vec3 reflectedDir = glm::reflect(pathSegment.ray.direction, normal); //For the incident vector I and surface orientation N, 
//																				  //returns the reflection direction
//		pathSegment.ray.origin = intersect + (EPSILON * normal);
//		pathSegment.ray.direction = glm::normalize(reflectedDir);
//		pathSegment.color *= m.specular.color;
//	}
//	
//	//Refractive surface 
//	else if (m.hasRefractive >= 0.01f)
//	{
//		glm::vec3 refractDir(0.f), newNormal = normal; 
//		float r0 = powf((1.f - m.indexOfRefraction) / (1 + m.indexOfRefraction), 2.f); 
//		float r1 = r0 + (1.f - r0) * powf(1.f - (glm::dot(glm::normalize(pathSegment.ray.direction), normal)), 5.f); 
//		float eta = m.indexOfRefraction; 
//		bool incoming = false; 
//		if (glm::dot(glm::normalize(pathSegment.ray.direction), normal) < 0.f)
//		{
//			incoming = true; 
//		}
//		if (u01(rng) > r1)
//		{
//			refractDir = glm::reflect(pathSegment.ray.direction, normal);
//		}
//		else
//		{
//			if (!incoming)
//			{
//				newNormal *= -10.f; 
//			}
//			else
//			{
//				eta = 1.f / eta; 
//			}
//			refractDir = glm::refract(pathSegment.ray.direction, newNormal, eta); 
//			if (glm::length(refractDir) < 0.01f)
//			{
//				pathSegment.color *= 0.f; 
//				refractDir = glm::reflect(pathSegment.ray.direction, newNormal); 
//			}
//
//		}
//		pathSegment.color *= m.specular.color; 
//		pathSegment.ray.direction = refractDir; 
//		pathSegment.ray.origin = intersect + (EPSILON * refractDir); 
//	}
//}