CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Sydney Miller
  * [LinkedIn](https://www.linkedin.com/in/sydney-miller-upenn/)
* Tested on: GTX 222 222MB (CETS Virtual Lab)

Table of contents
=================
   * [Overview of Path Tracing](#Overview)
   * [Features Overview](#features)
      * [BSDF Evaluation: Diffuse, Specular-Reflective, Refractive](#bsdf-evaluation)
      * [Physically-based depth-of-field](#depth-of-field)
      * [Stochastic Sampled Antialiasing](#stochastic-sampled-antialiasing)
      * [Obj Mesh Loading](#obj-mesh-loading)
      * [Procedural Shapes](#procedural-shapes)
      * [Procedural Shapes](#procedural-shapes)
      * [Procedural Textures](#procedural-textures)
      * [Better Hemisphere Sampling](#better-hemisphere-sampling)
  * [Optimizations ](#optimizations)
    * [Stream Compaction](#stream-compaction)
    * [Materials Contigious in Memory](#materials-contigious-in-memory)
    * [Cache First Intersection](#cache-first-intersection)
    * [Mesh Bounding Box](#mesh-bounding-box)
  * [Performance Analysis](#performance-analysis)
    * [Steam Compaction with Open and Closed Scenes](#steam-compaction-with-open-an-closed-scenes)
    * [Chaching First Intersection with Varying Depths](#chaching-first-intersection-with-varying-depths)

# Overview

# Features

## BSDF Evaluation

| Diffuse | Reflective | Refractive |
| ------------- | ----------- |----------- |
| ![](img/renders/part1Diffuse.png)  | ![](img/renders/part1Specular.png) | ![](img/renders/refractive.png) |
| Diffuse Blue | Reflective Blue | Refractive Blue |
| ![](img/renders/blueDiffuse.png)  | ![](img/renders/blueSpecular.png) | ![](img/renders/blueGlass.png) |

I implemented diffuse, reflective, and refracting materials for my pathtracer. Diffuse materials will randomly choose the direction rays bounce next whereas reflective materials will reflect the ray across the normal, so there is only one option for the new direction of a ray. I implemented a refractive material with Frensel effects using Schlick's approximation. Each of these implementations in the scene above took around 76 ms per iteration.

## Physically-based Depth-of-Field
| No Depth of Field | Depth of Field |
| ------------- | ----------- |
| ![](img/renders/part1Specular.png)  | ![](img/renders/DOF.png) |

I implemented physically bassed Depth-of-Field to create a focus effect on an object in the scene with a blurred background. This effect is also known as a thin lens approximation, which simulates a lens with a thickness much smalled than the radius of curvature to create the effect. The depth of field render shown took about 77 ms per iteration.

## Stochastic Sampled Antialiasing
| No Antialiasing | Antialiasing |
| ------------- | ----------- |
| ![](img/renders/part1Diffuse.png)  | ![](img/renders/antialiasing.png) |

I implemented stochastic sampled antialiasing by randomly offseting the values used for a rays origin from the camera when calculating the rays direction. You can tell in the image above that the edges of the sphere in the antialiased image are smoother than the edges of the sphere in the non-antialiased image.

## Obj Mesh Loading
![](img/renders/star2500Samples.png)

I implemented OBJ mesh loading using [tiny obj](https://github.com/syoyo/tinyobjloader). The mesh is loaded into the path tracer as individual triangles. The mesh shown above has 2576 polygons and took an average of 664.454 ms per iteration.

## Procedural Shapes
| Box Border Signed Distance Function | Sphere with Displacement Signed Distance Function |
| ------------- | ----------- |
| ![](img/renders/sdf1.png)  | ![](img/renders/sdf2.png) |

I created two procedural shapes using signed distance functions and used ray marching to find the intersection of each ray with the shape. 

## Procedural Textures
| Based on Intersect Position | Based on Normal |
| ------------- | ----------- |
| ![](img/renders/proceduralTexture2.png)  | ![](img/renders/proceduralTexture1.png) |

I created two procedural textures that can be applied to any shape in the scene. One texture uses the intersection position to manipulate the color and another uses the normal of the intersection to manipulate the color.

## Better Hemisphere Sampling

I implemented stratified hemisphere sampling to get a better distribution of ray directions. Instead of rays being chosen to shoot through a randoom spot on the hemisphere, a cell is chosen based on the current iteration for the ray to shoot out of so that the hemisphere will be sampled from all areas. 

# Optimizations

## Stream Compaction

After each bounce of the rays, the array holding the rays will be partitioned so that the rays that have terminated will be at the end of the list. The next bounce in the current iteration will only consider rays that are not terminated, which are now next to each other in the array. We do this so all of the rays that are considered on the next bounce are next to each other in memory, meaning there won't be as many idle threads.

## Materials Contigious in Memory

## Cache First Intersection

## Mesh Bounding Box

# Performance Analysis

## Steam Compaction with Open and Closed Scenes

## Chaching First Intersection with Varying Depths





