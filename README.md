CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**
* Sydney Miller
  * [LinkedIn](https://www.linkedin.com/in/sydney-miller-upenn/)
* Tested on: GTX 222 222MB (CETS Virtual Lab)


### README

### Part 1
![gif](img/renders/part1.gif)
![diffuse](img/renders/part1Diffuse.png)
![specular](img/renders/part1Specular.png)

#### Sorting Intersections and Path Segements Based on Material
When a scene is large and there are many materials, this feature increases performance. Since there is branching involved with every different material, it is faster if threads all in the same warp branch the same, otherwise the threads would have to wait for the diverging threads. 

#### Performance Benefit Across Max Ray Depths with Caching the First Bounce Intersectioon 
These experiments were run with 500 iterations and used a Cuda timer. The runtime increases as depth increases, but caching the first intersection increases the runtime for all of the depths testes.
![chart](img/part1Chart.png)

#### Bloopers
![blooper1](img/bloopers/cornell.2020-09-26_19-39-25z.17samp.png)
![blooper2](img/bloopers/cornell.2020-09-27_14-32-37z.8samp.png)
![blooper3](img/bloopers/sortingIntersectionsButNotPaths.png)
