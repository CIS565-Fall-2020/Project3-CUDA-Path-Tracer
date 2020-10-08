CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ling Xie
  * [LinkedIn](https://www.linkedin.com/in/ling-xie-94b939182/), 
  * [personal website](https://jack12xl.netlify.app).
* Tested on: 
  * Windows 10, Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz 2.20GHz ( two processors) 
  * 64.0 GB memory
  * NVIDIA TITAN XP GP102

Thanks to [FLARE LAB](http://faculty.sist.shanghaitech.edu.cn/faculty/liuxp/flare/index.html) for this ferocious monster.

##### Cmake change

Add 

1. [PerformanceTimer.h](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/src/csvfile.hpp) : Measure performance by system time clock. 
2. [cfg.h](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/stream_compaction/radixSort.h),  as a configure
3. add ray tracing load example from [tiny_gltf](https://github.com/syoyo/tinygltf)

#### Intro

In this project, we try to implement the [ray tracing]() algorithm on CUDA. It's still under construction and more features will come soon.

### Current feature

##### BxDF 

- [x]  Diffuse
- [x] Specular
  - [x] perfect Specular
  - [x] imperfect Specular
- [x] Refraction
  - [x] Schlick Fresnel approximate 

##### Better Sample

- [x] stratified sampling ( Results not obvious though )

##### Visual effects

- [x] lens based depth of field
- [x] motion blur

##### IO

- [x] Load GLTF based on tiny_gltf ( kind of buggy though)
  - [x] bounding box per mesh

##### Speed Optimization

- [x] Path continuation/termination with thrust
- [x] First bounce cache
- [x] contiguous memory shuffle by material sort

##### [BSDF]()

diffuse ball

<a href="https://github.com/Jack12xl/Project0-Getting-Started/tree/master/images"><img src="https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/cornell.2020-09-30_03-21-12z.5000samp.png" height="400px"></a> 

perfect reflection

<a href="https://github.com/Jack12xl/Project0-Getting-Started/tree/master/images"><img src="https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/cornell_2020-09-30_08-17-02z_5000samp_32depth.png" height="400px"></a> 

Basic kernel to represent the physical lighting attribute of each material.





### First bounce cache

Basically, first bounce cache is a time-space trade-off that use memory to store the first intersection results for further use  instead of calculating intersections at each iterations.

Here shows the results based on my implementation.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/First_bounce_cache.svg)

Well, obviously cache can trigger more performance with same results.

**Warning** when those (motion blur,  material sort) with random ray are applied, we can not use first bounce cache here.

### **Material Sort**

Material sort tries to sort the shader working sequence on material ID, which can makes threads do the same job as much as possible. 

|                                | With material sort  | Without             |
| ------------------------------ | ------------------- | ------------------- |
| **Elapsed time per iteration** | 122.94 milliseconds | 62.412 milliseconds |

However, for scenes simple like Cornell box, the more contiguous memory read can not cover the overhead of sorting(we use thrust::sort, basically a O(n) radix sort). 



###  Acknowledgement

[CUDA PATH Tracer](https://github.com/jmarcao/CUDA-Path-Tracer) by [Jmarcao](https://github.com/jmarcao): for the test scene he provides.

[tiny_gltf](https://github.com/syoyo/tinygltf) by [syoyo](https://github.com/syoyo)

[Ray tracing in one weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

For fast look up

[Physically based rendering](http://www.pbr-book.org/)

The most comprehensive book I ever found.

[Comparing hemisphere sampling techniques for obscurance computation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.4690&rep=rep1&type=pdf)

for stratified sampling.