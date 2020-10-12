CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

## SIREESHA PUTCHA 
	
* <img src= "img/Logos/linkedin.png" alt = "LinkedIn" height = "30" width = "30">   [ LinkedIn ](https://www.linkedin.com/in/sireesha-putcha/)

* <img src= "img/Logos/facebook.png" alt = "Fb" height = "30" width = "30">  [ Facebook ](https://www.facebook.com/sireesha.putcha98/)

* <img src= "img/Logos/chat.png" alt = "Portfolio" height = "30" width = "30">   [ Portfolio ](https://sites.google.com/view/sireeshaputcha/home)

* <img src= "img/Logos/mail.png" alt = "Mail" height = "30" width = "30">  [ Mail ](sireesha@seas.upenn.edu)


* Tested on personal computer - Microsoft Windows 10 Pro, 
Processor : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 2601 Mhz, 6 Core(s), 12 Logical Processor(s)
 
GPU : NVIDIA GeForce RTX 2060

## OUTPUT 

<p float="left">
 <img src="/img/renders/unicorn_1.png" height = "400" width = "400" />
 <img src="/img/renders/deer_int.png" height = "400" width = "400" />
<img src="/img/renders/bokeh_2.png" height = "400" width = "400" />
</p>

## PATH TRACER 

Path tracing is a type of ray tracing which takes a random sampling of all of the rays to create the final image. This results in sampling a variety of different types of lighting, but especially global illumination.
we see things because light emitted by light sources such as the sun bounces off of the surface of objects. When light rays bounce only once from the surface of an object to reach the eye, we speak of direct illumination. But when light rays are emitted by a light source, they can bounce off of the surface of objects multiple times before reaching the eye. 
This is what we call indirect illumination because light rays follow complex paths before entering the eye.

 <p float="left">
 <img src="/img/dlandindl.png"/>
 <img src="/img/pathtracer2.png"/>
</p>

## FEATURES 

### Diffuse Reflective and Refractive Objects 

Implemented materials like diffuse, specular with reflective and refractive surfaces. For diffuse, the given ray can bounce off the hemisphere covering the intersection point with equal probability.
So, we sample a point on this hemisphere randomly and set the reflected ray direction to be so. For reflective surfaces, the ray bounces off the surface with with the same angle as the angle of 
incidence from the normal, i.e, angle of reflection should be the same as angle of incidence from the normal. For refractive surfaces, the ray passes through the point of intersection. But, due to total
internal reflection, if the angle of incidence is less than critical angle, the surface behaves as a reflective surface and the ray reflects out. 
In the images below, you can see the spheres and cubes of different materials. 
<p float="left">
  <img src="/img/renders/myboxdl.png" height = "400" width = "400" />
  <img src="/img/renders/cornell1.png" height = "400" width = "400" />
  <img src="/img/renders/refr_1.png" height = "400" width = "400" /> 
  <img src="/img/renders/refr_2.png" height = "400" width = "400" /> 
</p>


### Depth Of Field 
Depth of field is an effect produced by the camera in which the lens focuses on objects only at a certain distance called focal length and blurs the rest of the objects. This is implemented in the pathtracer by sampling a point
from the lens as we were sampling it from a square plane and setting that as the ray origin and the direction based on the point of focus. 

<p float="left">
  <img src="/img/renders/dof_1.png" height = "400" width = "400" />
  <img src="/img/renders/dof_2.png" height = "400" width = "400" /> 
  <img src="/img/renders/refr_dof.png" height = "400" width = "400" /> 
</p>

#### - Bokeh 
Bokeh effect can be achieved by using a shaped lens on a regular camera. In our virtual camera, we have to map the points we sample to a particular shape in order to achieve it.

<p float = "left">
 <img src="/img/renders/bokeh_1.png" height = "400" width = "400" />
 <img src="/img/renders/bokeh_2.png" height = "400" width = "400" />
</p>

### Motion Blur 

<p float="left">
  <img src="/img/renders/motion_blur_1.png" height = "400" width = "400" />
  <img src="/img/renders/motion_blur_2.png" height = "400" width = "400" /> 
</p>



### OBJ Mesh Loading
Used tiny_obj_loader to import obj mesh models into the scene. Once we have the triangle data, we check against each triangle in the mesh for intersection with our ray. Since this 
would lead to numerous computations which are pointless, we cull the number of rays to be tested against all triangles by first checking the ray against the bounding box (min-max corner box)
, which is just one simple additional check, to see if the ray intersects the mesh at all. 

I have imported the following models into my path tracer. All of these were rendered within 20 mins. 

#### 1) Charizard 

<p float="left">
  <img src="/img/renders/charizard_1.png" height = "400" width = "400" />
  <img src="/img/renders/charizard_2.png" height = "400" width = "400" />
</p>


#### 2) Deer 

- Outside the box vs Inside the box 
 
 <p float="left">
 <img src="/img/renders/deer_ext.png" height = "400" width = "400" />
 <img src="/img/renders/deer_int.png" height = "400" width = "400" />
</p>


#### 3) Unicorn 

 <p float="left">
 <img src="/img/renders/unicorn_1.png" height = "400" width = "400" />
 <img src="/img/renders/unicorn_2.png" height = "400" width = "400" />
</p>


### Direct Lighting 
In our path tracer, at each bounce, the direction of the next ray is determined based on the material it hits. We do not make any computations to check if the point in space is directly
visible to light or not. In direct lighting, we make the last bounce hit the light by selecting a point randomly on a randomly selected light and setting the direction of the 
final ray to be the sampled point. This way, we will know if the object is directly visible to the the light or not. 
We accomplish this by checking the remaining bounces and if it is the last one, instead of sampling the bsdf for the direction, we sample the light instead. This lights up the scene
as shown below. 

With vs. Without 
 <p float="left">
 <img src="/img/renders/myboxdl.png" height = "400" width = "400" />
 <img src="/img/renders/myboxnodl.png" height = "400" width = "400" />
</p>

### Stochastic Sampled Antialiasing
Anti aliasing is a pixel level operation of adding a small random offset to the current ray direction to get an averaged color for pixels in order to avoid aliasing. 
The images below show the effect. 

 <p float="left">
 <img src="/img/renders/myboxnodl.png" height = "400" width = "400" />
 <img src="/img/renders/aaproof.png" height = "200" width = "400" />
</p>

 <p float="left">
 <img src="/img/renders/myboxnoaa.png" height = "400" width = "400" />
 <img src="/img/renders/noaaproof.png" height = "200" width = "400" />
</p>


## OPTIMIZATIONS 

### Sorting Rays by materials 
We make the rays or path segments contiguous in memory by sorting them based on the type of material the ray currently hits. The graph below depicts the 
time taken to trace different depths with and without sorting. I have used 3 different materials in the scene while measuring the performance. 
<img src="/img/raysort.png"/>

### Stream Compacting dead rays 
We use stream compaction to eliminate the dead or terminated rays from the pool of rays. The graph below shows how stream compacting rays improves the render time 
for different number of iterations. 
<img src="/img/sc.png"/>

### Caching first bounce 
We store the initial ray bounces in a special cache buffer for reusing it in all the next iterations. The graph below depicts the effect of caching at different iterations
<img src="/img/caching.png"/>


## Bloopers 
These are all the silly renders I came across while working on my path tracer.
 <p float="left">
 <img src="/img/bloop_refr.png" height = "400" width = "400" />
 <img src="/img/bloop_sc.png" height = "400" width = "400" />
 <img src="/img/bloop_dl_2.png" height = "400" width = "400" />
<img src="/img/bloop_obj.png" height = "400" width = "400" />
 <img src="/img/bloop_aa_cache.png" height = "400" width = "400" />
<img src="/img/blooper3.png" height = "400" width = "400" />
</p>


