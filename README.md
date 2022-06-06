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
- [x] Anti-aliasing 

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

Basic kernel to represent the physical lighting attribute of each material.

Since I'm not good at setting up fascinating scenes, here I borrow scenes settings from [jmrcao](https://github.com/jmarcao) as a fast test for my algorithm. Thanks **jmrcao**!

diffuse ball

<a href="https://github.com/Jack12xl/Project0-Getting-Started/tree/master/images"><img src="https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/cornell.2020-09-30_03-21-12z.5000samp.png" height="400px"></a> 

##### Perfect specular reflection

<a href="https://github.com/Jack12xl/Project0-Getting-Started/tree/master/images"><img src="https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/mid-submit/img/cornell_2020-09-30_08-17-02z_5000samp_32depth.png" height="400px"></a> 

##### Imperfect specular

Combing diffuse scattering and specular reflection can bring more vivacity and realness to the material. Here we randomly decide to do diffuse or specular based on the material shininess property. From left to right, shininess ranges from **5** to **Inf**. 

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/imperfect_spec.png)

##### Refractive transmission 

Refraction basically refers to light transmitting from material to another material based on index of refraction(**IOR**). Here I assume the atmosphere is surrounded with Air (**IOR** = 1). After some critical angle for a certain **IOR**, it would cause total internal reflection.

Here, from left to right, the index of refraction keeps increasing from 1 (Air) to 2.42(diamond), whose setting is exactly the same as [jmarcao](https://github.com/jmarcao/CUDA-Path-Tracer#refractive-transmission-scattering-function). So it achieves a comparable result with former.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/refraction.png)



##### Fresnel effect

It refers to the phenomenon that reflections easily appear when infer angle is smaller. Here I use [Shlick's approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation) to fit the ideal curve that light would transmit through or simply reflect on expectation. 

Notice in the followed light starts to reflect in a narrower angle. After some critical angle, it directly transmits through the material.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/fresnel.png)

#### Stratified sampling

Compared with uniform sampling, stratified sampling basically subdivides area and then sample in each block, which could reduces clustering of samples and bring  little noise especially around the edge.

However, in my implementation, the results basically demonstrate comparable results between stratified and uniform sampling.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/sample_camparison.png)



#### Anti-aliasing

Here I implement the anti-aliasing by randomly jitter the ray shot from camera.

As you can see, the **left** is anti-aliasing, which a smoother edge than the right one.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/compare_jitter.png)



### Visual effects

##### Depth of field

To simulate a real camera(with circle of confusion), instead of shooting rays from camera center as default, here we shoot rays from a small concentric disk with given radius **r** and focal distance **f**. 

Here **r** = 0.5,  **f** = 15.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/dof.png)



**Motion blur**

To simulate a scene with the effect like time of exposure and moving objects, we randomly change the objects transform along its given speed. How much the motion would "blur " depends on its magnitude of speed.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/motion_blur.png)



### glTF mesh loading

Here we implement the glTF loading function, which can load the glTF format mesh.

![alt text](https://github.com/Jack12xl/public_file/raw/master/CIS565-GPU/PathTracer/test_gltf.png)

##### Bounding box

Though the loading is not that correct, still we implement the bounding box that works just fine. A bounding box is created to do ray intersection culling. Which could avoid unnecessary ray hit.

For the scene showed above(kirby in Cornell).

|                                    | With bbox | without bbox |
| ---------------------------------- | --------- | ------------ |
| Elapsed time(ms) in each iteration | 326.1     | 476.9        |



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



#### Blooper

##### Wrong GLTF model loading

Funny thing, I debugged this for almost 1.5 years haha.

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/gltf_load.svg)

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/bloopers/failed_dof.png)

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/bloopers/wrong1.png)

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/bloopers/wrong2.png)

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/bloopers/wrong3.png)

![alt text](https://github.com/Jack12xl/Project3-CUDA-Path-Tracer/blob/master/img/bloopers/wrong4.png)



###  Acknowledgement

[CUDA PATH Tracer](https://github.com/jmarcao/CUDA-Path-Tracer) by [Jmarcao](https://github.com/jmarcao): for the test scene he provides.

[tiny_gltf](https://github.com/syoyo/tinygltf) by [syoyo](https://github.com/syoyo)

[Ray tracing in one weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

For fast look up

[Physically based rendering](http://www.pbr-book.org/)

The most comprehensive book I ever found.

[Comparing hemisphere sampling techniques for obscurance computation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.4690&rep=rep1&type=pdf)

for stratified sampling.

#### 

