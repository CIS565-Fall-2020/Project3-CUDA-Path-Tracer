CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Hanyu Liu
  - [personal website](http://liuhanyu.net/) 
* Tested on: Windows 10, Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz 16.0GB, GeForce GTX 1050 (Personal)

### Renders

![](.\img\diffuse_4430samp.png)

Diffuse Cornell Box @4430 Samples



![](.\img\specular_2071samp.png)

Specular Cornell Box @2071 Samples



### Materials Sorting

For a scene with objects with many different materials, it is faster to have segments with the same material in contiguous memory in order to group similar runtimes together, reducing waiting time for threads in the same warp. However, if there are not many materials in the scene, the time it takes to sort the materials exceeds the benefit we get in performance, thus making the render time slower.



### Caching First Intersections

Increasing the max ray depth will slow the performance and result in rendering time. If we cache the first bounce, we avoid doing the first intersection calculation, which speeds up the rendering. 











