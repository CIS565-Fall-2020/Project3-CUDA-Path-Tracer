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
The graph below shows how the average runtime with different opmization methods. Suprisingly, the naive method provides best performance in terms of processing time per iteration. When the sorting by material method is applied, the cost becomes 10 times as long as the naive method costs. This is because that the scene we used for test is simple and there are only a few meterials. Sorting materials is thus very time consuming.   

![](img/2.png)
