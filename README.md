CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Keyi Yu
  * [LinkedIn](https://www.linkedin.com/in/keyi-linda-yu-8b1178137/)
* Tested on: Windows 10, i7-10750H @ 2.60GHz 16GB, GeForce RTX 2070 with Max-Q 8228MB (My Machine)


Contents
-------------------------------------
- [Introduction](#Introduction)
- [Basic Path Tracing](#Basic-Path-Tracing)
  - [Generate Rays from Camera](#Generate-Rays-from-Camera)
  - [Compute Intersections](#Compute-Intersections)
  - [Scatter Rays](#Scatter-Rays)
  - [Shade Rays](#Shade-rays)
- [Improvement and Performance Analysis](#Improvement-and-Performance-Analysis)
- [Debug](#Debug)
- [Reference](#Reference)

## Introduction
In this project, I implemented a CUDA-based path tracer capable of rendering globally-illuminated images very quickly.

![](img/basicscene1.png)

## Core Features
### Generate Rays from Camera
The first step is to cast a ray from the camera to each pixel on the screen. The key point is to represent the screen point in the world coordinate. One naive approach is to do the inverse transformation but matrix computation is inefficient. Another way is to use the method shown in the picture below(Prof Adam Mally's CIS560 Slides).

![](img/generateRays.png)

### Compute Intersections
Now we can compute the intersection between the ray and objects in the world coordinate. One naive approch is to go through all the objects in the world and for each ray, whether there is an intersection between them. We also need to find the t which represent the depth from the camera to the object because we can only see the object in front.

### Scatter Rays
If a ray hit an object, it will bounce and generate a new ray. I use Bidirectional Scattering Distribution Functions(BSDF) to compute the new ray including the color, the direction adn the start position. At present, I just use a naive approach and will improve it in the next part. The peusdo code is:
```
generate a random number n01 from [0, 1]
if n01 < specular exponent:
    new ray color *= 0.5 * specular color + 0.5 * diffuse color
    new ray direction = 0.5 * specular direction + 0.5 * diffuse direction
else :
    new ray color *= diffuse color
    new ray direction = diffuse direction
```

### Shade Material
This step is to shade all the rays. There are three cases in general:
```
if ray hits a light:
    ray color *= light color
    remaining bounces = 0
else if ray hits an object:
    if this ray is the last bounce:
        ray color = black
        remaining bounces = 0
    else
        scatter rays
        remaining bounces--
else if ray hits nothing
    ray color = black
    remaining bounces = 0
```

### Path tracing
With all the functions above, we can do the path tracing in the following steps:
```
generate rays from camera
while # of bounce < bounce limits:
    compute intersections to determine the color of each ray
    shade material
    bounce depth++
gather colors and apply to the image
```
The basic path tracing can be improved in several ways. [This section](#Improvement-and-Performance-Analysis) will show some techniques.

## New Features
### Refraction (Fresnel + Schlick's approximation)
I used Schlick's approximation to compute the fresnel value to determine whether to do refraction or reflection. I mainly follow the steps in this [tutorial](https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/refraction). One thing need to notice is that I should add an offset to the origin of the new ray. If I don't add the offset, diffuse model still works well but the refraction model doesn't. 

When the ray is in the material with the higher refractive index, there is no real solution to Snell’s law, and thus there is no refraction possible. 
```
cannot refract = true, if refraction_ratio * sin_theta > 1.0
```

Here's some pesudo code:
```
If cannot refract or fresnel value > random value:
    do reflection
else :
    do refraction
```

The following pictures shows results of three different materials: perfect reflection, diffuse and refraction. Some details are shown in the crops.

![](img/basicscenedivide.png)

### Depth of Field
Camera in real life is not a perfect pinhole camera.The size of the lens help us to create some interesting effects, such as depth of field effect.

NO DOF            |  DOF
:-------------------------:|:-------------------------:
![](img/basicscene1.png) | ![](img/dof0.5.png)

### Stochastic Sampled Antialiasing
Since it will iterate many times, I randomly add an offset at each iteration for each pixel when generating rays from camera. First bounce cache should not be used here.

NO ANTIALISAING            |  ANTIALISAING
:-------------------------:|:-------------------------:
![](img/non-antilias.png) | ![](img/antialis.png)

### Better hemisphere sampling methods
I used 2D Jittered Sampling to generate two random numbers between 0 to 1. Then I mapped them to the hemisphere with a cosine weighting.

|Pesudo Random Sampling |  Jittered Sampling (25) |  Jittered Sampling (256)|
|--|--|----|
|![](img/sample0.png) | ![](img/sample25.png) | ![](img/sample256.png) |

### Arbitary Mesh Loading with bounding volume intersection culling
My project can load arbitary models in obj format. I used [tinyobj](https://github.com/tinyobjloader/tinyobjloader) to load obj models into some attribute vectors. Then I implemented triangle-ray intersection test. 

I used a bounding box to do reduce the number of rays that have to be checked against the entire mesh by first checking rays against a volume that completely bounds the mesh. The [performace](#Bounding-volume-intersection-culling) analysis can be found here.

### Object Motion and motion blur
Instead of moving objects, I move camera at each iteration to create motion blur effets. 
```
camera view = (1 - t) * old camera view + t * new camera view
where t is a random number at each iteration
```

back and forth            |  left and right
:-------------------------:|:-------------------------:
![](img/motion-z.png) | ![](img/motion-x.png)

## Improvement and Performance Analysis
There are four techniques to improve performance: stream compaction, sorting by materials, cache the first bounce and bounding volume intersection culling.

### Stream Compaction
Since not all the rays will hit an object and bounce, we need to eliminate some useless rays to reduce computation. Stream compaction helps here. I use stream compaction to select those will bounce and put the rest aside. From the left figure, we can see that over 10K rays will be eliminated after each bounce. Right figure shows that the duration is decreasing each time more intuitively.


Remaining rays            |  Duration of path tracning
:-------------------------:|:-------------------------:
![](img/remaning-rays-with-stream-compaction.png) | ![](img/duration-of-each-bounce-with-sc.png)

### Sorting by materials
Because different materials/BSDF evaluations will  take different amounts of time to complete, some threads may have to wait until other threads terminate. It is inefficient. So I sort the rays/path segments so that rays/path interacting with the same material are contiguous in memory before shading.  It looks like this method doesn't improve much right now because there are few types of materials. 

### Cache the first bounce
As long as the camera doesn't move, the first ray casting from the camera to the screen doesn't change. So we can cache the first bounce into a buffer and re-use it in the sequential iterations. Likewise, the # of the objects is small, it is hard to see the improvement.

### Bounding volume intersection culling
Bounding boxes can be used for pre-checking intersection. So I don't need to do useless check with every triangle.

## Basic Scene and Mario Scene Performance

In the following figures, Horizontal axis are:

   NO = NO OPTIMIZATION

   SC = STREAM COMPACTION

   FBC = FIRST BOUNCE CACHE

   SM = SORT BY MATERIAL

   VC = VOLUME CULLING

   ALL = SC + FBC + SM + VC

   Obviouly, optimazation techniques are pretty important for ray tracing. In the basic scene, the naive rendering still performs well. In the Mario scene, optimization skills save 50% time. Stream compaction is the most important one among all four techniques. Volume culling also save much time. However, in my test, sorting by materials doesn't perform well.

Basic Scene           |  Mario Scene
:-------------------------:|:-------------------------:
![](img/basic_performance.png) | ![](img/mario_performance.png)



## Debug
1. remove_if vs partition

When I used remove_if to filter ending ray, the light is black. remove_if doesn't put the removed elements anywhere; the elements after the new end will have their old values, so some of the removed elements might be missing, and some of the preserved elements might be duplicated. [Answer on stackovwerflow](#https://stackoverflow.com/questions/3521352/difference-between-partition-and-remove-functions-in-c)  So I used partition to do stream compaction.

Bug            |  Fixed
:-------------------------:|:-------------------------:
![](bug/p1-2xx.png) | ![](bug/p1-2xxx.png)


2. Wrong specular color

Firstly, I only add diffuse color. When I try to add specular color, the result looks werid. Because not all rays should have specular colors, I need to compare a random number with the specular exponent to determine whether the ray has spuclar color.

Bug            |  Fixed
:-------------------------:|:-------------------------:
![](bug/p1-2x.png) | ![](bug/p1-2x.png)

3. thrust::device_ptr vs device raw ptr

 I need to convert raw ptr to thrust::device_ptr before using thrust functions. Otherwise, error occurs.

 4. last bounce termination

I noticed that my image is a little lighter than the reference image. At the beginning, I think it is a small problem and the rendering image may differ on different machines. However, I found images posted on Piazza were exacly the same with the reference image, so I realized that there was a bug in my code. Thanks for Tea Tran's reponse on Piazza, I found my bug. Although some rays still have remaining bounces when the depth reaches the maximum depth, the last ray should also be colored black because it means that the ray doesn't hit a light within the maximum bounce. If not, the last ray will be colored with other colors, which make my rendering image a litte lighter.

 Bug            |  Fixed
:-------------------------:|:-------------------------:
![](bug/lastbouncenotblack.png) | ![](bug/lastbounceblack.png)

5. Wrong refraction
Sadly, I spent two days to debug this problem and finally found that I need to add an offset to the new origin after each bounce. Looks like I need to consider the thickness of the material.

## Still trying
I am trying on the Octree structure. At first, I just implemented a tree structure using pointers but failed to transfer it to GPU. Then I tried to flatten the tree and used vector to store OctreeNode, then transfer the vector to GPU. Although it renders some stuff, it is obvious that some meshes are missing. I realized that I used the mass center of the triangle to determine whether a triangle is in a grid or not. This method doesn't work. Still need to put each vertices in those grids...

![](bug/octree.png)