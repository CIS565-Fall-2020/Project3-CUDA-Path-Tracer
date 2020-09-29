CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Han Yan
* Tested on: CETS Virtual lab

## Part 1

### Sort by materials

For cornell.txt, since there are only a few kinds of materials, sorting the rays by material does not boost the performance (even slows it down a little bit because it has to call sort function in every depth and iteration).

But I expect that as the number of materials grwos larger, sorting would have some effect on the overall performance.

### Cache the first bounce/intersections

I ran performance analysis with varying depth 4, 8, 12, with and without caching. Here are the total time taken in each case by the kernel function 'computeIntersections'. The improvement on performance diminishes as the depth increases. This makes sense because we are only caching the first intersection, which could become negligible with a large depth.
