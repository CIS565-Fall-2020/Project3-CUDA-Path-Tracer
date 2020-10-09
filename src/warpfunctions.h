#pragma once
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

class WarpFunctions
{
public:
	__host__ __device__ 
	inline static glm::vec2 squareToDiskConcentric(const glm::vec2& u)
	{
		// Map uniform random numbers to [-1, 1]^2
		glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

		// Handle degeneracy at the origin
		if (uOffset.x == 0.f && uOffset.y == 0.f)
			return glm::vec2(0);

		// Apply concentric mapping to point
		float theta, r;
		if (std::abs(uOffset.x) > std::abs(uOffset.y))
		{
			r = uOffset.x;
			theta = PI_OVER_4 * (uOffset.y / uOffset.x);
		}
		else
		{
			r = uOffset.y;
			theta = PI_OVER_2 - PI_OVER_4 * (uOffset.x / uOffset.y);
		}

		return r * glm::vec2(std::cos(theta), std::sin(theta));
	}

	__host__ __device__
	inline static glm::vec3 squareToSphereUniform(const glm::vec2& sample)
	{
		float z = 1 - 2 * sample[0];
		float r = sqrt(std::max(0.f, 1.f - z * z));
		float phi = TWO_PI * sample[1];
		return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
	}
};