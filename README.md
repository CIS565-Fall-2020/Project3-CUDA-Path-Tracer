CUDA Path Tracer
================
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**
* Haorong Yang
* [LinkedIn](https://www.linkedin.com/in/haorong-henry-yang/)
* Tested on: Windows 10 Home, i7-10750H @ 2.60GHz 16GB, GTX 2070 Super Max-Q (Personal)

<img src="img/rongbig1959.png" width="650">  

3D Model Credit: [Rồng by @Husky on Sketchfab](https://sketchfab.com/3d-models/rong-b61cffbfbe66495b97a9a101b1859bbc?cursor=bz0xJnA9MjQ5)



## Features:  
### Graphics 
  * Bidirectional Scattering Distribution Functions: Ideal Diffuse, Specular Reflection, Refraction
  * Physically-based depth-of-field (by jittering rays within an aperture)
  * Stochastic Sampled Antialiasing
  * Arbitrary Mesh loading  
  
### Optimization
  * Path termination using stream compaction
  * Sorting pathSegments by material type
  * Acceleration by caching first bounce

### Physically-Based Depth of Field
No Depth of Field          |   With Depth of Field
:-------------------------:|:-------------------------:
<img src="img/nodepth.png" width="500">| <img src="img/cornell5000samp.png" width="500"> |


### Stochastic Sampled Antialiasing
No Anti Aliasing           |  With Anti Aliasing
:-------------------------:|:-------------------------:
<img src="img/no_anti_alias.PNG" width="500">| <img src="img/antialias.PNG" width="500"> |

### Arbitrary Mesh Loading
Avocado           |       Duck           |         Rồng
:-------------------------:|:-------------------------:|:-----------------------
<img src="img/avocado2396.png" width="330">| <img src="img/duck315.png" width="330">| <img src="img/rong.png" width="330">


3D Model Credit: [Khronos Group GLTF Sample Models](https://github.com/KhronosGroup/glTF-Sample-Models), [Rồng by @Husky on Sketchfab](https://sketchfab.com/3d-models/rong-b61cffbfbe66495b97a9a101b1859bbc?cursor=bz0xJnA9MjQ5)



### Optimization
Sorting the ray/path segments by material type will increase performance by making memory access contiguous hence more efficient; when there are a lot of materials, but not so much when there are limited materials, for example, in the conrell box test scene.


### Bloopers
<img src="img/blooper1.png" width="230"> <img src="img/blooper2.png" width="230">  <img src="img/blooper3.png" width="230"> <img src="img/blooper4.png" width="230">

### References
* [PBRT] Physically Based Rendering, Second Edition: From Theory To Implementation. Pharr, Matt and Humphreys, Greg. 2010.
* CIS565 Slides
* https://learnopengl.com/PBR/Theory
* https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/refraction
* http://wwwx.cs.unc.edu/~rademach/xroads-RT/RTarticle.html#:~:text=Its%20color%20is%20given%20by,way%20out%20into%20the%20scene.
