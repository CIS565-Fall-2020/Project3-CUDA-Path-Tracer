#pragma once

#include <glm/glm.hpp>

using namespace std;

class Image {
private:
    

public:
    int xSize;
    int ySize;
    glm::vec3* pixels;
    Image() = default;
    Image(int x, int y);
    ~Image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
};

class Texture : public Image {
public:
    float gamma;
    bool normalize;
    Texture(const std::string& filename, float gamma, bool normalize = false);
    void Load(const std::string& filename);
};
