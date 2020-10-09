#include "octree.h"

Octree::Octree()
{
    curTri = new Triangle();
}

Octree::Octree(int idx, glm::vec3 vert1, glm::vec3 vert2, glm::vec3 vert3) {
    curTri = new Triangle(idx);
    curTri->vert[0] = vert1;
    curTri->vert[1] = vert2;
    curTri->vert[2] = vert3;
}

Octree::Octree(float x1, float y1, float z1, float x2, float y2, float z2)
{
    if (x1 > x2 || y1 > y2 || z1 > z2) {
        return;
    }

    topLeftFront = glm::vec3(x1, y1, z1);
    bottomRightBack = glm::vec3(x2, y2, z2);

    curTri = nullptr;
	for (int i = 0; i < 8; i++) {
		children[i] = new Octree();
	}
}

void Octree::insert(int Idx, const glm::vec3 &vert1, const glm::vec3& vert2, const glm::vec3& vert3)
{
    glm::vec3 center = (vert1 + vert2 + vert3);
    center /= 3;


	float midx = (topLeftFront.x + bottomRightBack.x) / 2;
	float midy = (topLeftFront.y + bottomRightBack.y) / 2;
	float midz = (topLeftFront.z + bottomRightBack.z) / 2;

    int pos = -1;
    float x = center.x;
    float y = center.y;
    float z = center.z;

    // Checking the octant of 
    // the point 
    if (x <= midx) {
        if (y <= midy) {
            if (z <= midz)
                pos = TopLeftFront;
            else
                pos = TopLeftBottom;
        }
        else {
            if (z <= midz)
                pos = BottomLeftFront;
            else
                pos = BottomLeftBack;
        }
    }
    else {
        if (y <= midy) {
            if (z <= midz)
                pos = TopRightFront;
            else
                pos = TopRightBottom;
        }
        else {
            if (z <= midz)
                pos = BottomRightFront;
            else
                pos = BottomRightBack;
        }
    }

    // This node is an internal node
    if (children[pos]->curTri == nullptr) {
        children[pos]->insert(Idx, vert1, vert2, vert3);
        return;
    }

    // This node is a empty leaf node
    if (children[pos]->curTri->idx == -1) {
        delete children[pos];
        children[pos] = new Octree(Idx, vert1, vert2, vert3);
    }
    else {
        glm::vec3 vert1_ = children[pos]->curTri->vert[0];
        glm::vec3 vert2_ = children[pos]->curTri->vert[1];
        glm::vec3 vert3_ = children[pos]->curTri->vert[2];
        int idx_ = children[pos]->curTri->idx;

        delete children[pos];
        
        if (pos == TopLeftFront) {
            children[pos] = new Octree(topLeftFront.x,
                topLeftFront.y,
                topLeftFront.z,
                midx,
                midy,
                midz);
        }

        else if (pos == TopRightFront) {
            children[pos] = new Octree(midx + EPSILON,
                topLeftFront.y,
                topLeftFront.z,
                bottomRightBack.x,
                midy,
                midz);
        }
        else if (pos == BottomRightFront) {
            children[pos] = new Octree(midx + 1,
                midy + EPSILON,
                topLeftFront.z,
                bottomRightBack.x,
                bottomRightBack.y,
                midz);
        }
        else if (pos == BottomLeftFront) {
            children[pos] = new Octree(topLeftFront.x,
                midy + EPSILON,
                topLeftFront.z,
                midx,
                bottomRightBack.y,
                midz);
        }
        else if (pos == TopLeftBottom) {
            children[pos] = new Octree(topLeftFront.x,
                topLeftFront.y,
                midz + EPSILON,
                midx,
                midy,
                bottomRightBack.z);
        }
        else if (pos == TopRightBottom) {
            children[pos] = new Octree(midx + EPSILON,
                topLeftFront.y,
                midz + EPSILON,
                bottomRightBack.x,
                midy,
                bottomRightBack.z);
        }
        else if (pos == BottomRightBack) {
            children[pos] = new Octree(midx + EPSILON,
                midy + EPSILON,
                midz + EPSILON,
                bottomRightBack.x,
                bottomRightBack.y,
                bottomRightBack.z);
        }
        else if (pos == BottomLeftBack) {
            children[pos] = new Octree(topLeftFront.x,
                midy + EPSILON,
                midz + EPSILON,
                midx,
                bottomRightBack.y,
                bottomRightBack.z);
        }
        children[pos]->insert(idx_, vert1_, vert2_, vert3_);
        children[pos]->insert(Idx, vert1, vert2, vert3);
    }
}
