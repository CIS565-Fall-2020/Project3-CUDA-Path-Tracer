#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include "common.h"
#include "efficient.h"
#include "thrust.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* in, int pow2_d, int pow2_d1) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) return;
            if (index % pow2_d1 == 0) {
                in[index + pow2_d1 - 1] += in[index + pow2_d - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* in, int pow2_d, int pow2_d1) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) return;
            if (index % pow2_d1 == 0) {
                int left = in[index + pow2_d - 1];
                in[index + pow2_d - 1] = in[index + pow2_d1 - 1];
                in[index + pow2_d1 - 1] += left;
            }
        }
        
        /**
        * Helper method to calculate the distance from the nearest power of 2 greater than or equal to n
        */
        int distanceFromPowTwo(int n) {
            int pos = ceil(log2(n));
            return int(1 << pos) - n;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* input;
            int numItems = n;
            int zerosToPad = distanceFromPowTwo(n);
            if (zerosToPad == 0) {
                cudaMalloc((void**)&input, numItems * sizeof(int));
                cudaMemcpy(input, idata, sizeof(int) * numItems, cudaMemcpyHostToDevice);
            }
            else {
                numItems += zerosToPad;
                cudaMalloc((void**)&input, numItems * sizeof(int));
                cudaMemcpy(input + zerosToPad, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
                cudaMemset(input, 0, zerosToPad * sizeof(int));
            }
            dim3 fullBlocksPerGrid((numItems + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // up sweep
            for (int d = 0; d <= ilog2ceil(numItems) - 1; ++d) {
                int pow2_d = 1 << d;
                int pow2_d1 = 1 << (d + 1);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (numItems, d, input, pow2_d, pow2_d1);
            }
            // down sweep
            cudaMemset(input + numItems - 1, 0, sizeof(int));
            for (int d = ilog2ceil(numItems) - 1; d >= 0; --d) {
                int pow2_d = 1 << d;
                int pow2_d1 = 1 << (d + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (numItems, d, input, pow2_d, pow2_d1);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, input + zerosToPad, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(input);
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
            int* input; // input array
            int* output; // output array
            int* bools; // boolean array
            int* scanned; // scan result
            int zerosToPad = distanceFromPowTwo(n);
            int numItems = n + zerosToPad;
            cudaMalloc((void**)&input, numItems * sizeof(int));
            cudaMemcpy(input + zerosToPad, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(input, 0, zerosToPad * sizeof(int));
            cudaMalloc((void**)&output, numItems * sizeof(int));
            cudaMalloc((void**)&bools, numItems * sizeof(int));
            cudaMalloc((void**)&scanned, numItems * sizeof(int));

            dim3 fullBlocksPerGrid((numItems + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (numItems, bools, input);
            // do scan here
            cudaMemcpy(scanned, bools, sizeof(int) * numItems, cudaMemcpyDeviceToDevice);
            for (int d = 0; d <= ilog2ceil(numItems) - 1; ++d) {
                int pow2_d = 1 << d;
                int pow2_d1 = 1 << (d + 1);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (numItems, d, scanned, pow2_d, pow2_d1);
            }
            // down sweep
            cudaMemset(scanned + numItems - 1, 0, sizeof(int));
            for (int d = ilog2ceil(numItems) - 1; d >= 0; --d) {
                int pow2_d = 1 << d;
                int pow2_d1 = 1 << (d + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (numItems, d, scanned, pow2_d, pow2_d1);
            }
            StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (numItems, output, input, bools, scanned);
            timer().endGpuTimer();
            cudaMemcpy(odata, output, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int result = 0;
            for (int i = 0; i < n; ++i) {
                if (odata[i] == 0) continue;
                result++;
            }
            cudaFree(input);
            cudaFree(output);
            cudaFree(bools);
            cudaFree(scanned);
            return result;
        }
    }
}
