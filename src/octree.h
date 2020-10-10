#pragma once

#include "sceneStructs.h"
#include <queue>
#include <iostream>
#define EPSILON 0.000f

#define TopLeftFront 0 
#define TopRightFront 1 
#define BottomRightFront 2 
#define BottomLeftFront 3 
#define TopLeftBottom 4 
#define TopRightBottom 5 
#define BottomRightBack 6 
#define BottomLeftBack 7 

// Structure of a point 
struct Point {
    int x;
    int y;
    int z;
    Point()
        : x(-1), y(-1), z(-1)
    {
    }

    Point(int a, int b, int c)
        : x(a), y(b), z(c)
    {
    }
};

// Octree class 
class Octree {

    // if point == NULL, node is internal node. 
    // if point == (-1, -1, -1), node is empty. 
    Point* point;

    // Represent the boundary of the cube 
    Point* topLeftFront, * bottomRightBack;
    std::vector<Octree*> children;
    std::vector<int> triangles;

public:
    // Constructor 
    Octree()
    {
        // To declare empty node 
        point = new Point();
    }

    // Constructor with three arguments 
    Octree(int x, int y, int z)
    {
        // To declare point node 
        point = new Point(x, y, z);
    }

    // Constructor with six arguments 
    Octree(int x1, int y1, int z1, int x2, int y2, int z2)
    {
        // This use to construct Octree 
        // with boundaries defined 
        if (x2 < x1
            || y2 < y1
            || z2 < z1) {
            std::cout << "bounday poitns are not valid" << std::endl;
            return;
        }

        point = nullptr;
        topLeftFront
            = new Point(x1, y1, z1);
        bottomRightBack
            = new Point(x2, y2, z2);

        // Assigning null to the children 
        children.assign(8, nullptr);
        for (int i = TopLeftFront;
            i <= BottomLeftBack;
            ++i)
            children[i] = new Octree();
    }

    // Function to insert a point in the octree 
    void insert(int x,
        int y,
        int z)
    {

        // If the point already exists in the octree 
        if (find(x, y, z)) {
            std::cout << "Point already exist in the tree" << std::endl;
            return;
        }

        // If the point is out of bounds 
        if (x < topLeftFront->x
            || x > bottomRightBack->x
            || y < topLeftFront->y
            || y > bottomRightBack->y
            || z < topLeftFront->z
            || z > bottomRightBack->z) {
            std::cout << "Point is out of bound" << std::endl;
            return;
        }

        // Binary search to insert the point 
        int midx = (topLeftFront->x
            + bottomRightBack->x)
            / 2;
        int midy = (topLeftFront->y
            + bottomRightBack->y)
            / 2;
        int midz = (topLeftFront->z
            + bottomRightBack->z)
            / 2;

        int pos = -1;

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

        // If an internal node is encountered 
        if (children[pos]->point == nullptr) {
            children[pos]->insert(x, y, z);
            return;
        }

        // If an empty node is encountered 
        else if (children[pos]->point->x == -1) {
            delete children[pos];
            children[pos] = new Octree(x, y, z);
            return;
        }
        else {
            int x_ = children[pos]->point->x,
                y_ = children[pos]->point->y,
                z_ = children[pos]->point->z;
            delete children[pos];
            children[pos] = nullptr;
            if (pos == TopLeftFront) {
                children[pos] = new Octree(topLeftFront->x,
                    topLeftFront->y,
                    topLeftFront->z,
                    midx,
                    midy,
                    midz);
            }

            else if (pos == TopRightFront) {
                children[pos] = new Octree(midx + 1,
                    topLeftFront->y,
                    topLeftFront->z,
                    bottomRightBack->x,
                    midy,
                    midz);
            }
            else if (pos == BottomRightFront) {
                children[pos] = new Octree(midx + 1,
                    midy + 1,
                    topLeftFront->z,
                    bottomRightBack->x,
                    bottomRightBack->y,
                    midz);
            }
            else if (pos == BottomLeftFront) {
                children[pos] = new Octree(topLeftFront->x,
                    midy + 1,
                    topLeftFront->z,
                    midx,
                    bottomRightBack->y,
                    midz);
            }
            else if (pos == TopLeftBottom) {
                children[pos] = new Octree(topLeftFront->x,
                    topLeftFront->y,
                    midz + 1,
                    midx,
                    midy,
                    bottomRightBack->z);
            }
            else if (pos == TopRightBottom) {
                children[pos] = new Octree(midx + 1,
                    topLeftFront->y,
                    midz + 1,
                    bottomRightBack->x,
                    midy,
                    bottomRightBack->z);
            }
            else if (pos == BottomRightBack) {
                children[pos] = new Octree(midx + 1,
                    midy + 1,
                    midz + 1,
                    bottomRightBack->x,
                    bottomRightBack->y,
                    bottomRightBack->z);
            }
            else if (pos == BottomLeftBack) {
                children[pos] = new Octree(topLeftFront->x,
                    midy + 1,
                    midz + 1,
                    midx,
                    bottomRightBack->y,
                    bottomRightBack->z);
            }
            children[pos]->insert(x_, y_, z_);
            children[pos]->insert(x, y, z);
        }
    }

    // Function that returns true if the point 
    // (x, y, z) exists in the octree 
    bool find(int x, int y, int z)
    {
        // If point is out of bound 
        if (x < topLeftFront->x
            || x > bottomRightBack->x
            || y < topLeftFront->y
            || y > bottomRightBack->y
            || z < topLeftFront->z
            || z > bottomRightBack->z)
            return 0;

        // Otherwise perform binary search 
        // for each ordinate 
        int midx = (topLeftFront->x
            + bottomRightBack->x)
            / 2;
        int midy = (topLeftFront->y
            + bottomRightBack->y)
            / 2;
        int midz = (topLeftFront->z
            + bottomRightBack->z)
            / 2;

        int pos = -1;

        // Deciding the position 
        // where to move 
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

        // If an internal node is encountered 
        if (children[pos]->point == nullptr) {
            return children[pos]->find(x, y, z);
        }

        // If an empty node is encountered 
        else if (children[pos]->point->x == -1) {
            return 0;
        }
        else {

            // If node is found with 
            // the given value 
            if (x == children[pos]->point->x
                && y == children[pos]->point->y
                && z == children[pos]->point->z)
                return 1;
        }
        return 0;
    }
};




