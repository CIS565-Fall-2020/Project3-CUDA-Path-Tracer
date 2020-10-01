#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <device_launch_parameters.h>
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        

        __global__ void kernReduce(int nPadded, int d, int* dev_vec_padded) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < nPadded) {
                if (index % (1 << (d + 1)) == 0) { // (int)fmodf(index, 1 << (d + 1))
                    dev_vec_padded[index + (1 << (d + 1)) - 1] += dev_vec_padded[index + (1 << d) - 1];
                }
            }
        }

        __global__ void downSweep(int nPadded, int d, int* dev_vec_padded) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < nPadded) {
                if (index % (1 << (d + 1)) == 0) {
                    int t = dev_vec_padded[index + (1 << d) - 1];
                    dev_vec_padded[index + (1 << d) - 1] = dev_vec_padded[index + (1 << (d + 1)) - 1];
                    dev_vec_padded[index + (1 << (d + 1)) - 1] += t;
                }
            }

        }

        void scan(int n, int *odata, const int *idata) {
            int paddedSize = 1 << ilog2ceil(n);
            int nPadded = n;
            if (paddedSize > n) {
                nPadded = paddedSize;
            }

            int* dev_vec_padded;
            cudaMalloc((void**)&dev_vec_padded, nPadded * sizeof(int));
            cudaMemset(dev_vec_padded, 0, nPadded * sizeof(int));
            cudaMemcpy(dev_vec_padded, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 fullBlocksPerGridPadded((nPadded + blockSize - 1) / blockSize);
            
            // Reduce/Up-Sweep
            for (int d = 0; d <= ilog2ceil(nPadded) - 1; d++) {
                kernReduce << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_vec_padded);
            }
            // Set Root To Zero
            cudaMemset(dev_vec_padded + nPadded - 1, 0, sizeof(int));

            // Down-Sweep
            for (int d = ilog2ceil(nPadded) - 1; d >= 0; d--) {
                downSweep << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_vec_padded);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_vec_padded, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_vec_padded);
        }


        
        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */

        __global__ void kernMakeBool(int num, int* dev_vec) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < num && dev_vec[index] != 0) {
                dev_vec[index] = 1;
            }
        }

        __global__ void kernScatter(int n, int* dev_idata, int* dev_indices, int* dev_result) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n && dev_idata[index] != 0) {
                dev_result[dev_indices[index]] = dev_idata[index];
            }
        }
        __global__ void kernCheckNonZeroNum(int n, int* dev_result, int* num) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (dev_result[index] == 0 && dev_result[index - 1] != 0) {
                    *num = index;
                }
            }
        }

        int compact(int n, int *odata, const int *idata) {
            int paddedSize = 1 << ilog2ceil(n);
            int nPadded = n;
            if (paddedSize > n) {
                nPadded = paddedSize;
            }

            int* dev_indices;
            cudaMalloc((void**)&dev_indices, nPadded * sizeof(int));

            int* dev_bool;
            cudaMalloc((void**)&dev_bool, nPadded * sizeof(int));
            
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, nPadded * sizeof(int));
            cudaMemset(dev_idata, 0, nPadded * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* dev_result;
            cudaMalloc((void**)&dev_result, n * sizeof(int));

            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 fullBlocksPerGridPadded((nPadded + blockSize - 1) / blockSize);
            
            // Convert to 0s and 1s
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGridPadded, blockSize >> > (nPadded, dev_bool, dev_idata);
            cudaMemcpy(dev_indices, dev_bool, nPadded * sizeof(int), cudaMemcpyDeviceToDevice);

            // Reduce/Up-Sweep
            for (int d = 0; d <= ilog2ceil(nPadded) - 1; d++) {
                kernReduce << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_indices);
            }
            // Set Root To Zero
            cudaMemset(dev_indices + nPadded - 1, 0, sizeof(int));

            // Down-Sweep
            for (int d = ilog2ceil(nPadded) - 1; d >= 0; d--) {
                downSweep << <fullBlocksPerGridPadded, blockSize >> > (nPadded, d, dev_indices);
            }
            
            // Scatter
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_result, dev_idata, dev_bool, dev_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_result, n * sizeof(int), cudaMemcpyDeviceToHost);
            int nonZeroNum;
            cudaMemcpy(&nonZeroNum, dev_indices + nPadded - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_indices);
            cudaFree(dev_bool);
            cudaFree(dev_idata);
            cudaFree(dev_result);

            return nonZeroNum;
        }
    }
}
