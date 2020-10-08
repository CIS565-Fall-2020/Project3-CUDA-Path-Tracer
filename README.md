CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xuecheng Sun
* Tested on: Windows 10, R7 3700x @3.7GHz, RTX 2070 super 8GB



### Customized Scene: The Arsenal

![](./img/Arsenel_Blue.png)

![](./img/Arsenel_White.png)

### Path Tracer Part 1

**Finished Feature:** 

	1. Specular and Diffuse Ray Scatter
 	2. Basic stream compaction of stop thread
 	3. First bounce caching

![](./img/PathTracerPart1.png)



**Benchmark for First Ray Cache**

1. Sorted material can reduce the divergence in a thread warp which can significantly boost the render speed.

2. First bounce cache always provide a constant value improvement with different depth setting

![](img/PathTracerDepth.png)

### Path Tracer Part 2 and 3

**1.  Different Materials**

Implemented perfect refractive, perfect reflective and Schlick's approximation fresnel materials:

![](./img/DiffMat.png)

**2. Depth of Field and AA**

![](./img/DiffMat_DOF_AA_Comp.png)

**3.Arbitrary Mesh(GLTF) Import**

​	**Extra Points: Texture Mapping: Base Color, Normal and Roughness**

![](./img/GLTF.png)

**4. Direct Light**

![](./img/directLighting.png)

**5. Advance Sampling Method: Halton Sequence**

**Halton Direct Light Sampling**

![](./img/HaltonDirect.png)

**Random Direct Light Sampling as Reference**

![](./img/RandomDirect.png)

Shadows under Halton Sampling are more regularized.

**6. Octree(Partially)**

I build the octree for primitives and triangle but find for mesh triangles, it always aggregates in the root node which finally slow down the rendering. 

### Benchmark for Bounding Box

![](./img/BoundingBoxBM.png)

For a very simple mesh such as cube the bounding box didn't help much for the rendering speed, but if we render a more complex mesh like pistol or duck which have more than 4000 triangles, the bounding box improved the performance greatly.



### Third Libraries

Tinygltf libraries: GLTF Loader