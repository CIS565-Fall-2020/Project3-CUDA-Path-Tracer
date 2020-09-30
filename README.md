CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Qiaosen Chen
  * [LinkedIn](https://www.linkedin.com/in/qiaosen-chen-725699141/), etc.
* Tested on: Windows 10, i5-9400 @ 2.90GHz 16GB, GeForce RTX 2060 6GB (personal computer).

## Mid-project Summary

![pathtracing first demo gif](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/pathtracing_first_demo.gif)

### Features

- Ray sorting by material
- Ideal Diffuse Shading & Bounce
- Perfect Specular Reflection
- Stream Compaction
- Cache first bounce  

### Rendered Images

- Cornell Box with a Specular Sphere

  ![Cornell Box with a Specular Sphere Pic](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2020-09-30_19-21-14z.5000samp.png)

- Cornell Box with a Diffuse Sphere

  ![Cornell Box with a Diffuse Sphere Pic](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2020-09-30_19-28-44z.5000samp.png)

### Performance Analysis

I used the [scene](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/scenes/cornell.txt), a Cornell box with a specular sphere, to do the following test. With the function ```cudaEventElapsedTime()``` provided by *cuda_runtime_api.h*, I could compute how long GPU takes to do pathtracing iterations by creating two ```cudaEvent_t``` variables, ```iter_event_start``` and ```iter_event_start```, one to record the start time of an iteration and the other to record the end time. After each iteration, I added the running time of this iteration to a variable ```gpu_time_accumulator```, to accumulate the total time of all the iterations. Finally, I could get the average time of each iteration.

- Not Sort VS Sort Based on Materials

  ![Material Sort VS Not Sort Pic](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/performance_inset_%20materialsort_comp.png)

  Not using sorting algorithm gets a much better performance than the one using sorting algorithm. In fact, using ```thrust::sort_by_key``` to make ```pathSegments``` with the same material are contiguous in memory would takes double time to finish each iteration. In my opinion, this result may be due to that very few materials are used in the scene. Only 5 materials are used at this test, so sorting ```pathSegments``` according to their materials consumes more resources and takes more time.

- Not Cache VS First Bounce Cache

  ![Cache VS Not Cache Pic](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/performance_inset_%20cache_comp.png)  

  As we can see, caching the data computed in the first bounce for later iterations achieves a better performance, although it only takes a very few milliseconds less than the one which doesn't store the first bounce. Besides, it's obvious that the running time of each iteration increases as the depth increases.