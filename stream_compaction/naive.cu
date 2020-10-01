#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int* dev_vec1, int* dev_vec2) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (index >= (1 << (d - 1))) {
                    dev_vec2[index] = dev_vec1[index - (1 << (d - 1))] + dev_vec1[index];
                } else {
                    dev_vec2[index] = dev_vec1[index];
                } 
            } 
        }

        __global__ void kernInsertIdentity(int n, int* dev_vec1, int* dev_vec2) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (index == 0) {
                    dev_vec2[index] = 0;    // Insert identity element
                } else {
                    dev_vec2[index] = dev_vec1[index - 1];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_vec1;
            int* dev_vec2;
            int size = n * sizeof(int);
            cudaMalloc((void**)&dev_vec1, size);
            cudaMalloc((void**)&dev_vec2, size);
            cudaMemcpy(dev_vec1, idata, size, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            for (int d = 1; d <= ilog2ceil(n); d++) {
                // Launch kernel for inclusive scan
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, d, dev_vec1, dev_vec2);
                // Ping-pong buffers
                int* temp = dev_vec1;
                dev_vec1 = dev_vec2;
                dev_vec2 = temp;
            }
            // Launch kernel for shifting elements right and inserting identity element
            kernInsertIdentity << <fullBlocksPerGrid, blockSize >> > (n, dev_vec1, dev_vec2);
            // Ping-pong buffers again
            int* temp = dev_vec1;
            dev_vec1 = dev_vec2;
            dev_vec2 = temp;
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_vec1, size, cudaMemcpyDeviceToHost);
            cudaFree(dev_vec1);
            cudaFree(dev_vec2);
        }
    }
}
