CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xuecheng Sun
* Tested on: Windows 10, R7 3700x @3.7GHz, RTX 2070 super 8GB

### Path Tracer Part 1

**Finished Feature:** 

	1. Specular and Diffuse Ray Scatter
 	2. Basic stream compaction of stop thread
 	3. First bounce caching

![](./img/PathTracerPart1.png)



**Performance Analysis**

1. Sorted material can reduce the divergence in a thread warp which can significantly boost the render speed.

2. First bounce cache always provide a constant value improvement with different depth setting

![](img/PathTracerDepth.png)