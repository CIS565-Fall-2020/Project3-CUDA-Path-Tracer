CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Tushar Purang

  * [Linkedin](https://www.linkedin.com/in/tpurang/)

  * [Portfolio](http://tushvr.com/)

  * [Youtube Channel](https://www.youtube.com/channel/UC9ZTxWcJjCSAJDC54dPNbKw?view_as=subscriber)

* Tested on: Windows 10, i7-7700HQ @ 2.80GHz 16GB, GTX 1060 6GB (Personal laptop)


I have implemented a path tracer for this project. I also implemented a BSDF-kernel based shader that shades rays depending on the surfaces that it hits.If an object is specular then a specular reflection and scattering is done based on the probability set in the shader.

 Diffuse surfaces use cos-weghted scatering function to generate rays in random direction. Stream compaction is implimented using partitioning function in thrust library to eliminate terminated rays.

Here are some renders showcasing some of these main features that I have added.

<img src="img\cornell.2021-06-03_08-34-14z.71samp.png" style="zoom:80%;" />



<img src="\img\cornell.2021-06-10_01-26-46z.95samp.png" style="zoom:80%;" />

<img src="\img\cornell.2021-06-10_01-13-06z.90samp.png" style="zoom:80%;" />

I implemented first-bounce caching for the first part of the project. Sorting intersections based on materials before shading also optimized the program. Its effectiveness stems from the fact that GPU can be efficiently utilized because of less branching since adjacent paths are generally of the same material. For this path tracer, sorting resulted in performance drop since most of the materials are same and sorting was redundant.