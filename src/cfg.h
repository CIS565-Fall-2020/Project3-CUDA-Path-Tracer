#pragma once

#define camera_jittering 0
#define cache_first_bounce 0
#define material_sort 0

#if material_sort
	#define material_sort_ID 0
#endif

#define dof 0

#define stratified_sampling 0

#define DirectLightPass 1
#define DirectLightSampleLight 1
#define DirectLightSampleBSDF 0

#define InDirectLightPass 1

#define usebbox 1
#define motion_blur 0