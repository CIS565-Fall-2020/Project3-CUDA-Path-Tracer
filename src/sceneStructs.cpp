#include "sceneStructs.h"

aabbBounds::aabbBounds()
{
    bmin = vc3(std::numeric_limits<Float>::max());
    bmax = vc3(std::numeric_limits<Float>::lowest());
}

int aabbBounds::MaximumExtent() const
{
    vc3 d = this->bmax - this->bmin;
    if (d.x > d.y && d.x > d.z)
        return 0;
    else if (d.y > d.z)
        return 1;
    else
        return 2;
}

vc3 aabbBounds::Offset(const vc3& p) const
{
    return (p - bmin) / (bmax - bmin + vc3(1e-6));
}

Float aabbBounds::SurfaceArea() const
{
    vc3 d = bmax - bmin;
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
}
