#pragma once

#define camera_jittering 0
#define cache_first_bounce 0
#define material_sort 0

#define BRUTE_FORCE 0
#define HBVH 1

#if material_sort
	#define material_sort_ID 0
#endif

#define dof 0

#define stratified_sampling 0

#define DirectLightPass 1
#define DirectLightSampleLight 1
#define DirectLightSampleBSDF 1

#define InDirectLightPass 1

#define RAY_SCENE_INTERSECTION HBVH // 0 no acceleration, 1 for LBVH

#define usebbox 1
#define motion_blur 0