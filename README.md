CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Weiyu Du
* Tested on: CETS Virtual Lab

### Part 2
### Refraction

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/refract.png" width=300/>

### Depth of Field

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/dof_close.png" width=300/>
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/dof_far.png" width=300/></nobr>

### Antialiasing

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/antialiasing.png" width=300/>

### Arbitrary Mesh Loader

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/wahoo.png" width=300/>
Performance comparison regarding bounding volume interseciton culling (measured in time per iteration):

| OBJ file | bounding volume intersection culling | naive implementation |
| ---      | ---                                  | ---                  |
| Sphere   | 98.122 | 129.479 |
| Wahoo    | 1068.55 | 1453.84 |
| Stanford Bunny | 11970.6 | 22964.9 |

### Stratified Sampling

1) Comparison of stratified sampling (10x10 grid, left) and uniform random sampling (right) at 5000 iterations

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/strat_5000.png" width=300/><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/ref_5000.png" width=300/></nobr>

2) Comparison of stratified sampling (10x10 grid, left) and uniform random sampling (right) at 100 iterations

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/strat_100iter_10x10.png" width=300/><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/ref_100iter_10x10.png" width=300/></nobr>

### Motion Blur
1) Defined motion

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/defined_motion1.png" width=300/><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/defined_motion2.png" width=300/></nobr>

2) Real time camera motion

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/real_time_motion.png" width=300/>

### Part 1
### Render Result
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/mid-project-submission/img/render_res.png" width=300/>

### Analysis
1) Plot of elapsed time per iteration versus max ray depth (timed when sorting_material set to true)
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/mid-project-submission/img/hw3_plot.png"/>

- We expected that sorting the rays/path segments by material should improve the performance, because this will make the threads more likely to finish at around the same time, reducing waiting time for threads in the same warp. However, in reality we found that rendering without sorting is actually significantly faster. This may because that there isn't a variety of different materials in the scene. Since we're sorting the entire set of rays, this operation takes much more time than it saves.
- From the plot above we see that increasing max ray depth results in longer run time per iteration. Rendering using first bounce cache is consistently faster than rendering without cache, though not by a large margin. This is expected as we save time by avoiding the initial intersection computation.
