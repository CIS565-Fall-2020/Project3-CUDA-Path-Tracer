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

## FEATURES 

### Diffuse Reflective and Refractive Objects 

<p float="left">
  <img src="/img/renders/myboxdl.png" height = "400" width = "400" />
  <img src="/img/renders/cornell1.png" height = "400" width = "400" />
  <img src="/img/renders/refr_1.png" height = "400" width = "400" /> 
  <img src="/img/renders/refr_2.png" height = "400" width = "400" /> 
</p>


### Depth Of Field 

<p float="left">
  <img src="/img/renders/dof_1.png" height = "400" width = "400" />
  <img src="/img/renders/dof_2.png" height = "400" width = "400" /> 
  <img src="/img/renders/refr_dof.png" height = "400" width = "400" /> 
</p>

### Motion Blur 

<p float="left">
  <img src="/img/renders/motion_blur_1.png" height = "400" width = "400" />
  <img src="/img/renders/motion_blur_2.png" height = "400" width = "400" /> 
</p>



### OBJ Mesh Loading

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

With vs. Without 
 <p float="left">
 <img src="/img/renders/myboxdl.png" height = "400" width = "400" />
 <img src="/img/renders/myboxnodl.png" height = "400" width = "400" />
</p>

### Anti Aliasing

 <p float="left">
 <img src="/img/renders/myboxnodl.png" height = "400" width = "400" />
 <img src="/img/renders/aaproof.png" height = "200" width = "400" />
</p>

 <p float="left">
 <img src="/img/renders/myboxnoaa.png" height = "400" width = "400" />
 <img src="/img/renders/noaaproof.png" height = "200" width = "400" />
</p>

## PATH TRACER 

Path tracing is a type of ray tracing. The path tracing algorithm then takes a random sampling of all of the rays to create the final image. 
This results in sampling a variety of different types of lighting, but especially global illumination.
we see things because light emitted by light sources such as the sun bounces off of the surface of objects. When light rays bounce only once from the surface of an object to reach the eye, we speak of direct illumination. But when light rays are emitted by a light source, they can bounce off of the surface of objects multiple times before reaching the eye. 
This is what we call indirect illumination because light rays follow complex paths before entering the eye.

<img src="/img/dlandindl.png"/>

 <p float="left">
 <img src="/img/pathtracer1.png" height = "300" width = "300" />
 <img src="/img/pathtracer2.png" height = "300" width = "300" />
</p>


## OPTIMIZATIONS 

### Sorting Rays by materials 
<img src="/img/raysort.png" height = "400" width = "400" />

### Steam Compacting dead rays 
<img src="/img/sc.png" height = "400" width = "400" />

### Caching first bounce 
<img src="/img/caching.png" height = "400" width = "400" />


## Bloopers 

 <p float="left">
 <img src="/img/bloop_refr.png" height = "300" width = "300" />
 <img src="/img/bloop_sc.png" height = "300" width = "300" />
 <img src="/img/bloop_dl.png" height = "300" width = "300" />
<img src="/img/bloop_obj.png" height = "300" width = "300" />
 <img src="/img/bloop_aa_cache.png" height = "300" width = "300" />
<img src="/img/blooper3" height = "300" width = "300" />
</p>


