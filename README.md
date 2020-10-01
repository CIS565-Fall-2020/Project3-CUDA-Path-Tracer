CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Dayu Li
* Tested on: Windows 10, i7-10700K @ 3.80GHz 16GB, GTX 2070 8150MB (Personal laptop)

## Features

##### Mid-Submission: 

* A shading kernel with BSDF evaluation for:
  * Ideal Diffuse surfaces.
  * Perfectly reflective (mirrored) surfaces.
* Path continuation/termination using Stream Compaction.
* A toggleable option to sort by materials (Press M button).
* A toggleable option to cache the first bounce intersections for re-use across all subsequent iterations. Provide performance benefit analysis across different max ray depths (Press C button).

![](img/1.png)

## Performance Analysis
### Optimization analysis
The graph below shows how the average runtime with different opmization methods.
