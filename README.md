CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Weiyu Du
* Tested on: CETS Virtual Lab

### Render Result
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/mid-project-submission/img/render_res.png" width=600/>

### Analysis
1) Plot of elapsed time per iteration versus max ray depth (timed when sorting_material set to true)
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/mid-project-submission/img/hw3_plot.png"/>

- We expected that sorting the rays/path segments by material should improve the performance, because this will make the threads more likely to finish at around the same time, reducing waiting time for threads in the same warp. However, in reality we found that rendering without sorting is actually significantly faster. This may because that there isn't a variety of different materials in the scene. Since we're sorting the entire set of rays, this operation takes much more time than it saves.
- From the plot above we see that increasing max ray depth results in longer run time per iteration. Rendering using first bounce cache is consistently faster than rendering without cache, though not by a large margin. This is expected as we save time by avoiding the initial intersection computation.
