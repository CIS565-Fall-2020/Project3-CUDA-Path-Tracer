#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // up sweep
        __global__ void upSweep(int n, int d, int* data, int dist, int distHalf) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            if (index >= n || index % dist != 0) {
                return;
            }

            int toUpdate = index + dist - 1;
            int toGet = index + distHalf - 1;

            data[toUpdate] += data[toGet];
        }

        // up sweep efficient
        __global__ void upSweepEfficient(int n, int d, int* data, int stride, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || index >= n / stride) {
                return;
            }

            int toUpdate = ((index + 1) * stride) - 1;
            int toGet = toUpdate - offset;

            data[toUpdate] += data[toGet];
        }

        // down sweep
        __global__ void downSweep(int n, int d, int* data, int dist, int distHalf) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || index % dist != 0) {
                return;
            }

            int t_index = index + distHalf - 1;
            int replace_index = index + dist - 1;

            int t = data[t_index];
            data[t_index] = data[replace_index];
            data[replace_index] += t;
        }

        // down sweep efficient
        __global__ void downSweepEfficient(int n, int d, int* data, int stride, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || index >= n / stride) {
                return;
            }

            int replace_index = n - 1 - (index * stride);
            int t_index = replace_index - offset;
            

            int t = data[t_index];
            data[t_index] = data[replace_index];
            data[replace_index] += t;
        }

        // set n-1 to power of 2 values equal to 0
        __global__ void setZeros(int n, int power_of_2, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < power_of_2 && index >= n - 1) {
                data[index] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int power_of_2 = 1;
            while (power_of_2 < n) {
                power_of_2 *= 2;
            }

            // create array of size power of 2
            int* data;

            cudaMalloc((void**)&data, power_of_2 * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc data failed!");

            // fill array and pad end with 0's
            std::unique_ptr<int[]>padded_array{ new int[power_of_2] };
            cudaMemcpy(padded_array.get(), idata, sizeof(int) * n, cudaMemcpyHostToHost);
            for (int i = n; i < power_of_2; i++) {
                padded_array[i] = 0;
            }

            cudaMemcpy(data, padded_array.get(), sizeof(int) * power_of_2, cudaMemcpyHostToDevice);

            // kernel values
            int blockSize = 128;
            dim3 fullBlocksPerGrid((power_of_2 + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // up-sweep
            for (int d = 0; d <= ilog2(power_of_2) - 1; d++) {
                int dist = pow(2, d + 1);
                int distHalf = pow(2, d);
                upSweep << <fullBlocksPerGrid, blockSize >> > (power_of_2, d, data, dist, distHalf);
                /*int stride = pow(2, d+1);
                int offset = pow(2, d);
                upSweepEfficient << <fullBlocksPerGrid, blockSize >> > (power_of_2, d, data, stride, offset);*/
            }


            // set the last value to 0
            setZeros << <fullBlocksPerGrid, blockSize >> > (n, power_of_2, data);

            // down-sweep
            for (int d = ilog2(power_of_2) - 1; d >= 0; d--) {
                int dist = pow(2, d + 1);
                int distHalf = pow(2, d);
                downSweep << <fullBlocksPerGrid, blockSize >> > (power_of_2, d, data, dist, distHalf);
                /*int stride = pow(2, d + 1);
                int offset = pow(2, d);
                downSweepEfficient << <fullBlocksPerGrid, blockSize >> > (power_of_2, d, data, stride, offset);*/
            }
            timer().endGpuTimer();

            // set the out data to the scanned data
            cudaMemcpy(odata, data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(data);
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
        int compact(int n, int* odata, const int* idata) {
            // malloc necessary space oon GPU
            int* gpu_idata;
            int* bools;
            int* scanned_data;
            int* scattered_data;

            cudaMalloc((void**)&gpu_idata, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc gpu_idata failed!");
            cudaMemcpy(gpu_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            cudaMalloc((void**)&bools, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc bools failed!");

            cudaMalloc((void**)&scanned_data, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc scanned_data failed!");

            cudaMalloc((void**)&scattered_data, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc scattered_data failed!");

            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            //timer().startGpuTimer();
            // change to zeros and ones
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, bools, gpu_idata);

            // exclusive scan data
            scan(n, scanned_data, bools);

            // scatter
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, scattered_data, gpu_idata, bools, scanned_data);
            cudaMemcpy(odata, scattered_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int num = n;
            for (int i = 0; i < n; i++) {
                if (odata[i] == 0) {
                    num = i;
                    break;
                }
            }
            //timer().endGpuTimer();

            // return last index in scanned_data
            std::unique_ptr<int[]>scanned_cpu{ new int[n] };
            cudaMemcpy(scanned_cpu.get(), scanned_data, sizeof(int) * num, cudaMemcpyDeviceToHost);
            return num;
        }
    }
}
