CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Dayu Li
* Tested on: Windows 10, i7-10700K @ 3.80GHz 16GB, GTX 2070 8150MB (Personal laptop)  
* I used 2 extra days on this project

<p align="center">
    <img src="img/8.png" width="800"/>
</p>
## Features

##### Mid-Submission: 

* A shading kernel with BSDF evaluation for:
  * Ideal Diffuse surfaces.
  * Perfectly reflective (mirrored) surfaces.
* Path continuation/termination using Stream Compaction.
* A toggleable option to sort by materials (Press M button).
* A toggleable option to cache the first bounce intersections for re-use across all subsequent iterations. Provide performance benefit analysis across different max ray depths (Press C button).

##### Final-Submission: 
* Refraction (e.g. glass/water) with Frensel effects using Schlick's approximation.
* Physically-based depth-of-field (by jittering rays within an aperture).
* Stochastic Sampled Antialiasing. (1/2 visual effect)
* Arbitrary mesh loading and rendering:
  * tinyObj is used for loading obj files.
  * triangle intersection function glm::intersectRayTriangle.
  * Bounding volume intersection. AABB interction dectecting method.
* Halton hemisphere sampling methods.
* Direct Light. (2/2 visual effect)
* Motion Blur. (Extra credit)

## Effects
* A shading kernel with BSDF evaluation for:
  * Ideal Diffuse surfaces. (left)
  * Perfectly reflective (mirrored) surfaces. (right)

<p align="center">
    <img src="img/1.png" width="600"/>
</p>

* Refraction (e.g. glass/water) with Frensel effects using Schlick's approximation. (Mid sphere below)
  
<p align="center">
   <img src="img/3.png" width="300" height="300"/>
   <img src="img/4.png" width="300" height="300" />
</p>    

* Physically-based depth-of-field (with lens radius 0.5f and focal distance 5.5f.
  * To toggle the depth-of-field setting, press D key while the program is running
  
<p align="center">
   <img src="img/9.png" width="300" height="300"/>
   <img src="img/10.png" width="300" height="300" />
</p>  

* Stochastic Sampled Antialiasing. (1/2 visual effect)
  * Anti aliasing makes the meshes more smooth in their outlines and make the ray bouncing more accurate
  * Difference can be seen obviously in the refelctions of top light on the surface of the refraction sphere (right one)
  * To toggle the anti-aliasing setting, press A key while the program is running
  
<p align="center">
   <img src="img/9.png" width="300" height="300" />
   <img src="img/11.png" width="300" height="300" />
</p>  

* Arbitrary mesh loading and rendering:

<p align="center">
    <img src="img/8.png" alt="mesh" width="600"/>
</p>

* Direct Lighting. (2/2 visual effect):
  * Taking a final ray directly to a random point on an emissive object acting as a light source.
  * With the direct lighting method, the rendered image should be lighter than path tracing due to the direct light components.
  * With the same limited iterations, the direct light effect should make the rendering faster than the indirect path tracing, below are comarations in 1, 10 and 100 iterations
  
  <p align="center">
   <img src="img/NDL_1.png" width="300" height="300" />
   <img src="img/DL_1.png" width="300" height="300" />
   <img src="img/NDL_10.png" width="300" height="300" />
   <img src="img/DL_10.png" width="300" height="300" />
   <img src="img/NDL_100.png" width="300" height="300" />
   <img src="img/DL_100.png" width="300" height="300" />
</p>  

* Motion blur
 * Very straight forward, combine the view vector linearly with a motion vector of the camera will generate motion blur effects.
 * Below are Z (0,0,0.5) and Y (0,0.1,0) motion blur effects
 
 <p align="center">
   <img src="img/Motion_1.png" width="300" height="300" />
   <img src="img/Motion_2.png" width="300" height="300" />
</p>  

## Performance Analysis
### Optimization analysis
* The graph below shows how the average runtime with different opmization methods. Suprisingly, the naive method provides best performance in terms of processing time per iteration. When the sorting by material method is applied, the cost becomes 10 times as long as the naive method costs. This is because that the scene we used for test is simple and there are only a few meterials. Sorting materials is thus very time consuming.   

<p align="center">
    <img src="img/2.png" alt="mesh" width="600"/>
</p>

* Try with the complicated mesh scene, this time the stream compaction makes a great difference, average iteration time was dropped from 210 ms to 140 ms, which is a great improve in efficience. However, applying cache-first-iteration, as well as the material sort is not obviously improving the iteration time. This is because that the material applied in the scenario is still not too much. Sorting material ids is not saving time but wasting.

<p align="center">
    <img src="img/Column1.png" alt="mesh" width="600"/>
</p>

* Try with a scene consist of more than 20 materials, this time sort_by_material is making some difference: average iteration time dropped by ~10 ms. Still not as effective as stream compaction when applied to a complicated scene.

<p align="center">
    <img src="img/Column2.png" alt="mesh" width="600"/>
</p>
