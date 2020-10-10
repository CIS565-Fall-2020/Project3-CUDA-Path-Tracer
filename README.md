

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
- Refraction with Frensel effects using [Schlick's approximation](https://en.wikipedia.org/wiki/Schlick's_approximation).
- Physically-based depth-of-field
- Stochastic Sampled Antialiasing
- Arbitrary mesh loading and rendering glTF files with toggleable bounding volume intersection culling
- Better hemisphere sampling methods
- Direct lighting
- Motion blur by averaging samples at different times in the animation (Extra Credit)

### Rendered Images

- **Specular surface reflection & Diffuse surface reflection & Refraction**

  | Specular Surface                                             | Diffuse Surface                                              | Refraction                                                   |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2020-09-30_19-21-14z.5000samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2020-09-30_19-28-44z.5000samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell_glassball.2020-10-08_03-59-42z.5000samp.png) |

- **glTF mesh loading**

  | Venus                                                        | Spear Bearer                                                 | Sparta                                                       |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-10_01-36-04z.4948samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-10_01-14-21z.5000samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-10_02-02-34z.5000samp.png) |

- **Physically-based depth-of-field (Focal Distance = 10)**

  | No Depth-of-fieldLens Radius                                 | With Depth-of-fieldLens Radius                               |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-07_19-30-29z.5000samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-08_20-37-26z.5000samp.png) |

- **Stochastic Sampled Antialiasing**

  | With Anti-Aliasing                                           | Without Anti-Aliasing                                        |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/Anti.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/NoAnti.png) |

- **Stratified sampling method**

  | Random sampling                                              | Stratified sampling                                          |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/Anti.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/NoAnti.png) |

  It's hard to find out some obvious differences between the naive random sampling and the stratified sampling.

- **Direct Lighting**

  | Indirect Lighting                                            | Direct Lighting                                              |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-10_03-37-27z.4147samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-10_03-43-41z.726samp.png) |

- **Motion Blur**

  | With Motion Blur                                             | Without Motion Blur                                          |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-10_03-37-27z.4147samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-10_03-31-35z.4226samp.png) |

## Performance Analysis

I used the [scene](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/scenes/cornell.txt), a Cornell box with a specular sphere, to do the following test. With the function ```cudaEventElapsedTime()``` provided by *cuda_runtime_api.h*, I could compute how long GPU takes to do pathtracing iterations by creating two ```cudaEvent_t``` variables, ```iter_event_start``` and ```iter_event_start```, one to record the start time of an iteration and the other to record the end time. After each iteration, I added the running time of this iteration to a variable ```gpu_time_accumulator```, to accumulate the total time of all the iterations. Finally, I could get the average time of each iteration.

- Not Sort VS Sort Based on Materials

  ![Material Sort VS Not Sort Pic](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/performance_inset_%20materialsort_comp.png)

  Not using sorting algorithm gets a much better performance than the one using sorting algorithm. In fact, using ```thrust::sort_by_key``` to make ```pathSegments``` with the same material are contiguous in memory would takes double time to finish each iteration. In my opinion, this result may be due to that very few materials are used in the scene. Only 5 materials are used at this test, so sorting ```pathSegments``` according to their materials consumes more resources and takes more time.

- Not Cache VS First Bounce Cache

  ![Cache VS Not Cache Pic](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/performance_inset_%20cache_comp.png)  

  As we can see, caching the data computed in the first bounce for later iterations achieves a better performance, although it only takes a very few milliseconds less than the one which doesn't store the first bounce. Besides, it's obvious that the running time of each iteration increases as the depth increases.

- **Not Bounding Box VS Bounding Box Intersection Culling**

  For the [Venus scene](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/scenes/cornell_venus.txt), apparently the situation with bounding box intersection culling intersection outperforms the situation without bounding box.

  ![Not Bounding Box VS Bounding Box Intersection Culling](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/performance_inset_%20bbox_comp.png)

## Bloopers

| Strange direct lighting                                      | Incorrect mesh surface normal calculation                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-09_00-32-44z.5000samp.png) | ![](https://github.com/giaosame/Project3-CUDA-Path-Tracer/blob/master/img/rendered_images/cornell.2020-10-09_23-15-50z.6samp.png) |