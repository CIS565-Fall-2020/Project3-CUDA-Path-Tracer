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

## Physically-based Depth-of-Field
| No Depth of Field | Depth of Field |
| ------------- | ----------- |
| ![](img/renders/part1Specular.png)  | ![](img/renders/DOF.png) |

## Stochastic Sampled Antialiasing
| No Antialiasing | Antialiasing |
| ------------- | ----------- |
| ![](img/renders/part1Diffuse.png)  | ![](img/renders/antialiasing.png) |

## Obj Mesh Loading
![](img/renders/star2500Samples.png)

## Procedural Shapes
| ------------- | ----------- |
| ![](img/renders/sdf1.png)  | ![](img/renders/sdf2.png) |

## Procedural Textures
| ------------- | ----------- |
| ![](img/renders/proceduralTexture2.png)  | ![](img/renders/proceduralTexture1.png) |

## Better Hemisphere Sampling

# Optimizations

## Stream Compaction

## Materials Contigious in Memory

## Cache First Intersection

## Mesh Bounding Box

# Performance Analysis

## Steam Compaction with Open and Closed Scenes

## Chaching First Intersection with Varying Depths





