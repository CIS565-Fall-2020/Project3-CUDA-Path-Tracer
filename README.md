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

#### Intro

In this project, we try to implement the [ray tracing]() algorithm on CUDA. It's still under construction and more features will come soon.

### Current feature

- [x] A basic BSDF 
  - [x]  Diffuse
  - [x] Reflective
  - [ ] Refraction
- [x] Path continuation/termination
- [x] First bounce cache
- [x] contiguous memory shuffle by material sort

##### [BSDF]()

<a href="https://github.com/Jack12xl/Project0-Getting-Started/tree/master/images"><img src="https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/cornell.2020-09-30_03-21-12z.5000samp.png" height="400px"></a> 

<a href="https://github.com/Jack12xl/Project0-Getting-Started/tree/master/images"><img src="https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/cornell_2020-09-30_08-17-02z_5000samp_32depth.png" height="400px"></a> 

Basic kernel to represent the physical lighting attribute of each material.

### First bounce cache

Basically, first bounce cache is a time-space trade-off that use memory to store the first intersection results for further use  instead of calculating intersections at each iterations.

Here shows the results based on my implementation.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/First_bounce_cache.svg)

Well, obviously cache can trigger more performance with same results.

### **Material Sort**

Material sort tries to sort the shader working sequence on material ID, which can makes threads do the same job as much as possible. 

| With material sort  | Without             |
| ------------------- | ------------------- |
| 122.94 milliseconds | 62.412 milliseconds |

However, for scenes simple like Cornell box, the more contiguous memory read can not cover the overhead of sorting(we use thrust::sort, basically a O(n) radix sort). 