CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Janine Liu
  * [LinkedIn](https://www.linkedin.com/in/liujanine/), [personal website](https://www.janineliu.com/).
* Tested on: Windows 10, i7-10750H CPU @ 2.60GHz 16GB, GeForce RTX 2070 8192 MB (personal computer)

This project revolved around a GPU-based pathtracer, its optimizations, and several distinct features.

## Performance Analysis Methods

Using the Performance Timer class provided in the [previous assignment](https://github.com/j9liu/Project2-Stream-Compaction/), I surrounded my `pathtrace` call with calls to start and stop the CPU timer. I then took the average of all these iterations to determine what the average iteration time would be. To save time, I limited the number of these recorded iterations to 20% of the total number of samples.

For the Stream Compaction section, I simply collected data for the first iteration of the pathtracing process.

## Optimizations

To improve performance, I implemented some optimizations with the intent to make each pathtracing iteration faster. These optimizations include:
* stream compaction,
* material sorting, and
* first bounce caching.

These measurements are taken from the pathtracing process on the default Cornell Box image, show below. A chart of the performance changes as a result of these optimizations is shown at the end of the section.

![](img/presentable/diffuse.png)

### Stream Compaction

Stream compaction is an algorithm that removes elements from an array that meet a certain condition and reorganizes the elements so they are contiguous in place. Here, I use the `stable_partition` function in Thrust's library to target the rays who have no bounces left and need to be terminated, removing them from the rest of the paths to be considered.

Unfortunately, I do not have a reliable baseline comparison without stream compaction because my pathtracer acts erroneously without it. Some bloopers associated with this bug can be found in the bloopers section below. In any case, the stream compaction optimization produces a correct result that is anticipated to improve performance. It removes rays that are no longer active from the list of paths to trace, so we do not needlessly trace terminated paths. In addition, the function puts the active rays closer together in memory, which should make accessing global memory much faster since the accesses will be continguous, instead of random. Below is a graph showcasing the decrease in rays with each subsequent bounce.

![](img/graphs/stream_compaction.png)

It's important to note that this optimization works best with open scenes; these scenes contain many rays that terminate early because they stop intersecting with geometry. In a closed scene, none of the rays terminate apart from those that hit a light source, because the enclosed space causes them to continually bounce and hit geometry. For comparison, I used a zoomed in version of the Cornell box and placed a wall behind the camera so the space was truly enclosed.

![](img/presentable/diffuse_close_up.png)

Using this scene, as opposed to the default one, creates a noticeable difference in performance between the two scene types. This is demonstrated below.

![](img/graphs/stream_compaction_2.png)

Indeed, the average iteration time for the open scene was **21.6799ms**, while that of the closed scene was **34.7032ms**.

### Material Sorting

Each material in the scene has a unique ID that scene intersections reference whenever they collide with a material. Continguous intersections in memory can have different materials between them, thus leading to random memory access in the materials' global memory bank. To improve memory access, intersections who share the same material can be sorted based on material ID, so intersections with the same materials are coalesced in memory.

The chart, pictured later, will actually demonstrate an increase in time when material sorting is used. This is because there are few materials in the default Cornell scene, and not enough to optimize to justify the overhead of sorting repeatedly. It is expected that this will improve when there are much more materials in the scene to manage. Scenes with more materials will experience greater latency with unsorted, random memory access.

### Caching First Bounce

Since the target pixel coordinates don't change with each iteration, the camera shoots the same rays into the scene and hits the same geometry every time, before using random sampling to scatter into different parts of the scene. Thus, the rays' intersections from the first bounce can be stored to improve performance, since they won't be repeatedly calculated with each iteration. A plot of the cache's performance against varying levels of maximum ray depth is shown below.

![](img/graphs/cache.png)

For the most part, using this optimization shaves off around 0.5ms from the average iteration speed, except for the outlier towards the end. Given 5000 iterations for the Cornell image, this amounts to a 0.5ms x 5000 = 2500ms = **2.5 second** difference in performance, which does not seem like much. I'm wondering if this requires more geometry in the scene (and thus more potential intersections) for this to be substantially optimized.

The performance of all the optimizations combined is shown below.

![](img/graphs/optimization_graph.png)

## Features

## Refractive Materials
## Depth of Field

## Bugs and Bloopers
### Stream Compaction-less Issues

My pathtracer cannot render images properly without using stream compaction. Here are two bloopers from when I was trying to debug this issue:

![](img/bloopers/without_stream_compaction2.png)
![](img/bloopers/without_stream_compaction.png)

The intense brightness of these renders comes from overcounting the dead rays in the final product. But despite consistent checks that the rays aren't terminated, the images produced are still taking too much light into account, light that shouldn't exist with the terminated rays. It may have to do with shading and overcounting *intersection* data; I would need to spend more time to figure out how.