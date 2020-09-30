CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ling Xie
  * [LinkedIn](https://www.linkedin.com/in/ling-xie-94b939182/), 
  * [personal website](https://jack12xl.netlify.app).
* Tested on: 
  * Windows 10, Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz 2.20GHz ( two processors) 
  * 64.0 GB memory
  * NVIDIA TITAN XP GP102

Thanks to [FLARE LAB](http://faculty.sist.shanghaitech.edu.cn/faculty/liuxp/flare/index.html) for this ferocious monster.

##### Cmake change

Add 

1. [PerformanceTimer.h](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/src/csvfile.hpp) : Measure performance by system time clock. 
2. [radixSort.h](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/stream_compaction/radixSort.h), [radixSort.cu](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/stream_compaction/radixSort.cu) for [Part 6](