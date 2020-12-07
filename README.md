CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Name: Gizem Dal
  * [LinkedIn](https://www.linkedin.com/in/gizemdal), [personal website](https://www.gizemdal.com/)
* Tested on: Predator G3-571 Intel(R) Core(TM) i7-7700HQ CPU @ 2.80 GHz 2.81 GHz - Personal computer

<img src="img/renders/cornell_4.png" alt="sneak peek" width=800>

## Project Description ##

This is a CUDA-based path tracer capable of rendering globally-illuminated images very quickly with the power of parallelization.

**Features**

* Shading Kernel with BSDF Evaluation
  * Uniform diffuse
  * Perfect specular reflective (mirror)
  * Perfect specular refractive (glass)
  * Fresnel dielectric
  * Imperfect specular (glossy)
* Path Continuation/Termination with Stream Compaction
* Toggleable continuous storage of paths and intersections by material type
* Toggleable first bounce intersection cache to be used by subsequent iterations
* Anti-aliasing rays with sub-pixel samples
* Arbitrary GLTF mesh loading with toggleable bounding volume intersection culling
* Camera depth of field
* SDF-based Tanglecube and Bounding Box
* Procedural texture
* Stratified sampling
* Hierarchical Spatial Structure - Octree (In progress)

## Material Overview ##

Material shading is split into different BSDF evaluation functions based on material type. Supported materials include diffuse, mirror, glass, fresnel dielectric and glossy materials. Diffuse material scattering is computed by using cosine-weighted samples within a hemisphere. Fresnel dielectric materials are defined in the scene file with an index of refraction, which is used by BSDF evaluation to compute the probability of refracting versus reflecting the scatter ray. Mirror material scattering function reflects the ray along the surface normal while glossy reflection happens within a lobe computed by the specular exponent of the material.

**Speculars**

Fresnel dielectric | Perfect refractive | Perfect specular
:---: | :---: | :---: 
<img src="img/renders/ico_alias_1500_dielec.png" alt="sneak peek" width=300> | <img src="img/renders/ico_glass_1500.png" alt="sneak peek" width=300> | <img src="img/renders/ico_mirror_1500.png" alt="sneak peek" width=300>

**Imperfect Specular Reflection**

Exponent = 0 | Exponent = 5 | Exponent = 12 | Exponent = 50 | Exponent = 500
:---: | :---: | :---: |  :---: | :---:
<img src="img/renders/glossy_0.png" alt="Glossy 0" width=200> | <img src="img/renders/glossy_5.png" alt="Glossy 5" width=200> | <img src="img/renders/glossy_12.png" alt="Glossy 12" width=200> | <img src="img/renders/glossy_50.png" alt="Glossy 50" width=200> | <img src="img/renders/glossy_500.png" alt="Glossy 500" width=200>

Lower specular exponent values give results that are closer to diffuse scattering while larger specular exponent values result in larger highlights and more mirror-like surfaces.

## Features Overview ##

**GLTF Mesh Loading**
[Icosahedron](https://people.sc.fsu.edu/~jburkardt/data/obj/icosahedron.obj) | [Magnolia](https://people.sc.fsu.edu/~jburkardt/data/obj/magnolia.obj) | [Duck](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/Duck)
:---: | :---: | :---:
<img src="img/renders/ico_1500.png" alt="sneak peek" width=300> | <img src="img/renders/magnolia_1000.png" alt="sneak peek" width=300> | <img src="img/renders/duck_1200.png" alt="sneak peek" width=300>

In order to bring the mesh data into C++, I used the [tinygltf](https://github.com/syoyo/tinygltf/) library. I used the VBOs from the imported data to create the mesh triangles and store triangle information per arbitrary mesh.

Bounding volume intersection culling is applied at the ray-geometry intersection test to reduce the number of rays that have to be checked against the entire mesh by first checking rays against a volume that completely bounds the mesh. This feature is implemented as toggleable for performance analysis purposes. Pressing the 'B' key while running the GPU renderer will enable this feature.

In order to smoothen the triangles on round GLTF meshes, the intersection normal is computed from the barycentric interpolation of the 3 normals from the triangle vertices.

**Depth of Field**

Focal Distance = 30, Lens Radius = 2.5 | Focal Distance = 20, Lens Radius = 2.5
:---: | :---:
<img src="img/renders/DOF_focal30.png" alt="sneak peek" width=600> | <img src="img/renders/DOF_focal20.png" alt="sneak peek" width=600>

The scene camera can be set to enable focal distance and lens radius to get a depth of field effect. Geometries located at the focal distance within the lens radius will stay in focus while other geometry around the scene will be distorted.

**Anti-aliasing**

Anti-aliasing enabled | Anti-aliasing disabled
:---: | :---: |
<img src="img/renders/ico_alias_1500_dielec.png" alt="anti-aliasing enabled" width=500> | <img src="img/renders/ico_no_alias_1500_dielec.png" alt="anti-aliasing disabled" width=500>

Using anti-aliasing for subpixel sampling results in smoother geometry edges in the render results. It is important to note that anti-aliasing and first bounce cache do not work together, since the pixel samples will differ per iteration and cached first bounces from the first iteration won't match the generated ray direction in further iterations. In order to provide flexibility, I set first bounce cache usage as a toggleable feature rather than the default, so that anti-aliasing could be enabled if the first bounce cache is not used.

**SDF-Based Implicit Surfaces**

Tanglecube | Bounding Box
:---: | :---:
<img src="img/renders/tanglecube_500.png" alt="tanglecube" width=500> | <img src="img/renders/bound_box_500.png" alt="bounding box" width=500>

Ray-geometry intersection for implicit surfaces is computed by special signed distance functions and ray marching. The sign of the SDF return value determines whether the ray is outside, on or inside the implicit geometry. Ray marching is used for following the ray direction in small increments, passing the current position on the ray to the SDF function and deciding whether an intersection occured or the marching should be terminated if the maximum marching distance is reached.

**Procedural Textures**

[FBM](https://thebookofshaders.com/13/) | [Noise](https://thebookofshaders.com/edit.php#11/wood.frag)
:---: | :---:
<img src="img/renders/fb_tex.png" alt="fbm" width=500> | <img src="img/renders/noise_tex.png" alt="noise" width=500>

I closely followed the procedural texture implementations from the link provided on top of the render images. I find the *Book of Shaders* noise texture implementations to be pretty useful for generating aesthetically pleasing procedural textures from fragment data. The two procedural textures currently supported by the renderer include Fractal Brownian Motion texture which benefits from a loop of adding noise to create a fractal looking noise pattern, and Wood Noise with a swirl effect.

**Stratified Sampling**

Stratified (4 samples) | Random (4 samples)
:---: | :---:
<img src="img/renders/STRATIFIED_4.png" alt="stratified 4" width=600> | <img src="img/renders/RANDOM_4.png" alt="random 4" width=600>

Stratified (4 samples close up) | Random (4 samples close up)
:---: | :---:
<img src="img/renders/STRATIFIED_4_close.png" alt="stratified 25" width=600> | <img src="img/renders/RANDOM_4_close.png" alt="random 25" width=600>

Although it isn't very visible at larger sample sizes, using stratified samples compared to random samples results in slightly more converged results at very small iteration steps.

**Hierarchical Spatial Structure - Octree (In progress)**

I started implementing a hierarchical spatial structure named Octree. The purpose of this data structure is to contain the scene geometry within children nodes (at most 8 children nodes per node) by using 3D volume bounding boxes with the goal of eliminating naive geometry iteration in the ray-scene intersection test, thus improve the rendering performance. Due to time constraints, this feature is not completed yet though it is still in the works.

## Insights ##

One interesting observation I have is that using material sort results in more stable render results compared to naive approach. The two images below, both rendered with 4950 iterations, are renders from the same camera position. The render on the left is taken by sorting rays by material type while the one on the right is rendered by the naive approach.

Material sort enabled | Material sort disabled
:---: | :---: |
<img src="img/readme.png" alt="anti-aliasing enabled" width=300> | <img src="img/readme_nosort.png" alt="anti-aliasing disabled" width=300>

## Performance Analysis ##

I used a GPU timer to conduct my performance analysis on different features and settings in the renderer. This timer is wrapped up as a performance timer class, which uses the CUDAEvent library, in order to measure the time cost conveniently. The timer is started right before the iteration loop and is terminated once a certain number of iterations is reached. The measured time excludes the initial cudaMalloc() and cudaMemset() operations for the path tracer buffers, but still includes the cudaGLMapBuffer operations.

**Stream Compaction**

Using stream compaction to terminate paths which don't hit any geometry in the scene can be very beneficial for open scenes where some portion of the rays will shoot to void from the start. In order to show the impact of stream compaction on render path segments, I have created 3 test scenes with different layouts. All 3 test scenes are rendered with 1500 iterations.

Closed Cornell | Open Cornell I | Open Cornell II
:---: | :---: | :---:
<img src="img/renders/dode_closed_1500.png" alt="Closed Cornell" width=300> | <img src="img/renders/dode_1500.png" alt="Open Cornell I" width=300> | <img src="img/renders/dode_out_1500.png" alt="Open Cornell II" width=300>

For stream compaction analysis, I timed a single iteration of the renderer and recorded the number of remaning paths after each path termination in the 3 test scenes. I used a ray depth of 8, such that once this depth is reached all remaning paths would be terminated.

<img src="img/stream_graph.png" alt="stream compaction graph" width=1200>

I also recorded the total runtime of 1 iteration per test scene.

Scene | Measured runtime (in seconds)
:---: | :---: 
Closed Cornell | 0.260587
Open Cornell I | 0.146422
Open Cornell II | 0.106178

From this data, we can conclude that using stream compaction for path termination is very beneficial performance wise for open scenes where some rays could shoot to void and not hit any geometry within the scene. Although terminating as many paths as possible when needed is a significant performance improvement, it might not be possible to terminate many rays in closed scenes such as Closed Cornell where rays cannot escape.

**Volume Intersection Culling**

I used 4 arbitrary mesh examples to analyze the peformance benefits of enabling volume intersection culling for complex meshes. The table below shows the mesh examples used for this analysis and how many triangles they contain.

Mesh | Number of triangles
:---: | :---: 
Icosahedron | 20
Magnolia | 1372
Duck | 4212
Stanford Bunny | 69630

I measured the total runtime of 6 iterations in 1 test scene per mesh. I used simple open cornell box scenes where no other arbitrary meshes except the subject mesh is included. The runtime measurements yielded the following results.

<img src="img/volume_graph.png" alt="Volume Intersection Culling graph" width=1000>

Using volume intersection culling for simpler arbitrary meshes such as Icosahedron or Dodecahedron doesn't provide a significant performance improvement for a small number of iterations, however as the total number of iterations increases it can provide more significant efficiency. Although we do not observe a significant performance improvement with Icosahedron, we can see that using volume intersection culling improves scene intersection test performance significantly for scenes where much more complex shapes such as Stanford Bunny is present. Just with 6 iterations, using volume intersection culling has saved about 29 seconds.

**First Bounce Cache with SDF-based Implicit Surfaces**

Although simple scenes may not benefit from using the first bounce cache for subsequent iterations, scenes with SDF-based implicit surfaces where each path segment is ray marched against the procedural surface for intersection test benefit from the first bounce cache significantly. I have tested the impact of first bounce cache on two implicit surfaces currently supported by the renderer, Tanglecube and 3D Bounding Box, to compare the renderer performance against no usage of cache. This test involves only two iterations, where the first bounce from iteration 1 is stored to be used in iteration 2. For simplicity, the maximum ray marching distance is set to 20 units for both shapes. I picked the ray march steps per shape based on the observed results from different step sizes.

Shape | Ray march step
:---: | :---: 
Tanglecube | 0.0001
Bounding Box | 0.001

<img src="img/cache_graph.png" alt="First Bounce Cache graph" width=900>

As shown in the graph, using a first bounce cache improves the performance of each iteration significantly by 1-2 seconds depending on the shape and ray march step. The current ray marching implementation does not use sphere marching, however it could be a performance improvement to consider in the future. Another potential performance improvement could come from defining bounding boxes for implicit surfaces such that the ray will be tested against the surface only if it falls within the bounds. This could be achieved by the octree implementation.

**Procedural Textures**

Current procedural textures supported by the renderer make many calls to noise helper functions that call many glm math functions. In order to analyze the potential impacts of procedural textures on runtime, I created 3 simple test scenes with a sphere where the diffuse material of the sphere uses FBM, Wood noise and no texture separately and compared the total runtimes of 100 and 500 iterations.

<img src="img/texture_graph.png" alt="Texture graph" width=900>

Although the difference is not very significant due to the small number of iterations, using the FBM texture seems to be slightly less efficient than using Wood Noise or no texture at all. Since FBM functions usually call their helpers the octave amount of times, it is possible that these subsequent function calls could slow down the performance.

## Bloopers ##

<img src="img/blooper_1.png" alt="b" width=300> <img src="img/blooper_2.png" alt="b" width=300> 

