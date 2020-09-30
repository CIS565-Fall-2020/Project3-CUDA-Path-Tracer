#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        int* dev_extend;

        int* dev_map;
        int* dev_scan;
        int* dev_scatter;
        int* dev_data;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            int power = 1 << (d + 1);
            int step = 1 << d;
            if (index != 0 && (index + 1) % power == 0) {
                data[index] += data[index - step];
            }
        }

        __global__ void kernUpSweepOptimized(int n, int stride, int* data, int start) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }

            int index = k * stride + start;
            data[index] += data[index - stride / 2];
            

        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            int power = 1 << (d + 1);
            int step = power >> 1;

            if (index != 0 && (index + 1) % power == 0) {
                int t = data[index - step];
                data[index - step] = data[index];
                data[index] += t;
            }
        }

        __global__ void kernDownSweepOptimized(int n, int stride, int* data, int start) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }

            int index = start + k * stride;
            int power = stride;
            if (index != 0 && (index + 1) % power == 0) {
                int t = data[index - stride / 2];
                data[index - stride / 2] = data[index];
                data[index] += t;
            }
        }

        __global__ void kernExtendArr(int extendNum, int n, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= extendNum) {
                return;
            }
            if (index >= n) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index];
            }
        }

        __global__ void kernMap(int n, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] = idata[index] == 0 ? 0 : 1;
        }

        __global__ void kernSetValue(int n, int value, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == n) {
                data[index] = value;
            }
            else {
                return;
            }
        }

        __global__ void kernSetValueOptimized(int n, int value, int* data, int stride) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            data[index + stride - 1] = value;
        }

        __global__ void kernScatter(int n, int* idata, int* scan, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (idata[index] != 0) {
                odata[scan[index]] = idata[index];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO

            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Expand non power-2 to power-2
            int ceil = ilog2ceil(n);
            int num = 1 << ceil;
            int* extendData = new int[num];
            int* tmp = new int[num];

            cudaMalloc((void**)&dev_extend, num * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            kernExtendArr << <fullBlocksPerGrid, threadsPerBlock >> > (num, n, dev_data, dev_extend);

            for (int d = 0; d <= ceil; d++) {
                kernUpSweep << <fullBlocksPerGrid, threadsPerBlock >> > (num, d, dev_extend);
            }

            kernSetValue << <fullBlocksPerGrid, threadsPerBlock >> > (num - 1, 0, dev_extend);

            for (int d = ceil - 1; d >= 0; d--) {
                kernDownSweep << <fullBlocksPerGrid, threadsPerBlock >> > (num, d, dev_extend);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_extend, n * sizeof(int), cudaMemcpyDeviceToHost);


            
            /*
            printf("_________________test____________________\n");
            for (int i = 0; i < n; i++) {
                printf("%3d  ", odata[i]);
            }
            */
            cudaFree(dev_extend);
            cudaFree(dev_data);

            delete[] tmp;
            delete[] extendData;
        }

        void scanOptimized(int n, int* odata, const int* idata) {
            dim3 threadsPerBlock(blockSize);


            // Expand non power-2 to power-2
            int ceil = ilog2ceil(n);
            int num = 1 << ceil;
            int* extendData = new int[num];
            int* tmp = new int[num];

            cudaMalloc((void**)&dev_extend, num * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("dev_arrr failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);


            dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            kernExtendArr << <fullBlocksPerGrid, threadsPerBlock >> > (num, n, dev_data, dev_extend);

            for (int d = 1; d <= ceil; d++) {
                int threadNum = 1 << (ceil - d);
                int stride = 1 << d;
                int start = stride - 1;
                fullBlocksPerGrid = (threadNum + blockSize - 1) / blockSize;

                kernUpSweepOptimized << <fullBlocksPerGrid, threadsPerBlock >> > (threadNum, stride, dev_extend, start);
            }

            kernSetValueOptimized << <1, threadsPerBlock >> > (1, 0, dev_extend, num);

            for (int d = ceil - 1; d >= 0; d--) {
                int threadNum = 1 << (ceil - d - 1);
                int stride = 1 << (d + 1);
                int start = stride - 1;
                fullBlocksPerGrid = (threadNum + blockSize - 1) / blockSize;

                kernDownSweepOptimized << <fullBlocksPerGrid, threadsPerBlock >> > (threadNum, stride, dev_extend, start);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_extend, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_extend);
            cudaFree(dev_data);

            delete[] tmp;
            delete[] extendData;
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
        int compact(int n, int *odata, const int *idata) {


            int ceil = ilog2ceil(n);
            int num = 1 << ceil;

            int* host_scan = new int[num];
            int* tmp = new int[num];

            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);


            cudaMalloc((void**)&dev_map, num * sizeof(int));
            checkCUDAError("dev_map failed!");


            cudaMalloc((void**)&dev_scan, num * sizeof(int));
            checkCUDAError("dev_scan failed!");


            cudaMalloc((void**)&dev_scatter, num * sizeof(int));
            checkCUDAError("dev_scatter failed!");

            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("dev_data failed!");

            cudaMalloc((void**)&dev_extend, num * sizeof(int));
            checkCUDAError("dev_extend failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            // Extend non-power of 2
            kernExtendArr << <fullBlocksPerGrid, threadsPerBlock >> > (num, n, dev_data, dev_extend);

            // map
            kernMap << <fullBlocksPerGrid, threadsPerBlock >> > (num, dev_extend, dev_scan);

            // scan
            for (int d = 0; d <= ceil; d++) {
                kernUpSweep << <fullBlocksPerGrid, threadsPerBlock >> > (num, d, dev_scan);
            }

            kernSetValue << <fullBlocksPerGrid, threadsPerBlock >> > (num - 1, 0, dev_scan);


            for (int d = ceil - 1; d >= 0; d--) {
                kernDownSweep << <fullBlocksPerGrid, threadsPerBlock >> > (num, d, dev_scan);              
            }
            // scatter
            kernScatter << <fullBlocksPerGrid, threadsPerBlock >> > (num, dev_extend, dev_scan, dev_scatter);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_scatter, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_scan, dev_scan, num * sizeof(int), cudaMemcpyDeviceToHost);


            cudaFree(dev_extend);
            cudaFree(dev_data);
            cudaFree(dev_map);
            cudaFree(dev_scan);
            cudaFree(dev_scatter);

            int count = host_scan[n - 1];
            if (1 << ceil != n) {
                count = host_scan[n];
            }
            delete[] host_scan;
            delete[] tmp;
            
            return count;
        }
    }
}
