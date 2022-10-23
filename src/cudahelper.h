#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "cfg.h"
#include <stdio.h>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char* msg, const char* file, int line);
