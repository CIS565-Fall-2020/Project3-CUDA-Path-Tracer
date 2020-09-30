#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void naiveScanParallel(int n, int power, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index >= power) {
                odata[index] = idata[index - power] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        // Convert inclusive to exclusive
        __global__ void convert(int n, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (index == 0) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // TODO
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* dev_arr1;
            int* dev_arr2;
            int direction = 1;
            int* tmp = new int[n];

            cudaMalloc((void**)&dev_arr1, n * sizeof(int));
            checkCUDAError("dev_arrr1 failed!");
            cudaMalloc((void**)&dev_arr2, n * sizeof(int));
            checkCUDAError("dev_arrr2 failed!");

            cudaMemcpy(dev_arr1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();

            int ceil = ilog2ceil(n);
            for (int d = 1; d <= ceil; d++) {
                int power = 1 << (d - 1);
                if (direction == 1) {
                    naiveScanParallel << <fullBlocksPerGrid, threadsPerBlock >> > (n, power, dev_arr1, dev_arr2);
                }
                else {
                    naiveScanParallel << <fullBlocksPerGrid, threadsPerBlock >> > (n, power, dev_arr2, dev_arr1);
                }
                /*
                printf("level %d \n", d);
                for (int i = 0; i < n; i++) {
                    printf("%3d  ", tmp[i]);
                }
                printf("\n");
                */
                direction *= -1;
            }
            if (direction == 1) {
                convert << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_arr1, dev_arr2);
                /*
                printf("result %d \n");
                for (int i = 0; i < n; i++) {
                    printf("%3d  ", odata[i]);
                }
                printf("\n");
                */
            }
            else {
                convert << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_arr2, dev_arr1);
                /*
                printf("result %d \n");
                for (int i = 0; i < n; i++) {
                    printf("%3d  ", odata[i]);
                }
                printf("\n");
                */
            }
            timer().endGpuTimer();
            if (direction == 1) {
                cudaMemcpy(odata, dev_arr2, n * sizeof(int), cudaMemcpyDeviceToHost);
            }
            else {
                cudaMemcpy(odata, dev_arr1, n * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(dev_arr1);
            cudaFree(dev_arr2);

            delete[] tmp;

        }
    }
}
