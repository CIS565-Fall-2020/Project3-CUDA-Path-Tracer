#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <stb_image.h>

#include "image.h"

Image::Image(int x, int y) :
        xSize(x),
        ySize(y),
        pixels(new glm::vec3[x * y]) {
}

Image::~Image() {
    delete pixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) { 
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

Texture::Texture(const std::string& filename, float gamma, bool normalize): gamma(gamma), normalize(normalize)
{
    this->Load(filename);
}

void Texture::Load(const std::string& path)
{
	int channels = 0;
	float* rawPixels = stbi_loadf(path.c_str(), &this->xSize, &this->ySize, &channels, 3);

	if (channels == 3 || channels == 4)
	{
		glm::vec3 correction = glm::vec3(gamma);
		this->pixels = new glm::vec3[xSize * ySize];

		glm::vec3 accum = glm::vec3(0.f);

		if (normalize)
		{
			for (int i = 0; i < xSize * ySize; i++)
			{
				glm::vec3 color;
				color.x = rawPixels[i * channels];
				color.y = rawPixels[i * channels + 1];
				color.z = rawPixels[i * channels + 2];

				accum += color;
			}

			accum /= xSize * ySize;
		}

		for (int i = 0; i < xSize * ySize; i++)
		{
			glm::vec3 color;
			color.x = rawPixels[i * channels];
			color.y = rawPixels[i * channels + 1];
			color.z = rawPixels[i * channels + 2];

			if (normalize)
				color /= accum;

			this->pixels[i] = glm::pow(color, correction);
		}

		std::cout << "Loaded texture \"" << path << "\" [" << xSize << "x" << ySize << "|" << channels << "]" << std::endl;
	}
	else
	{
		std::cerr << "Error loading texture " << path << std::endl;
	}

	stbi_image_free(rawPixels);
}
