#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {


        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // one iteration of inclusive scan
        __global__ void iteration(int n, int d, const int* idata, int* odata, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            } else {
                odata[index] = idata[index];
            }
        }

        // turns inclusive scan to exclusive scane
        __global__ void inclusiveToExclusive(int n, const int* idata, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index + 1 >= n) {
                return;
            }
            if (index == 0) {
                odata[0] = 0;
            }
            odata[index + 1] = idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int power_of_2 = 1;
            while (power_of_2 < n) {
                power_of_2 *= 2;
            }

            // create arrays of size power of 2
            int* data_1;
            int* data_2;

            cudaMalloc((void**)&data_1, power_of_2 * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc data_1 failed!");
            cudaMalloc((void**)&data_2, power_of_2 * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc data_2 failed!");

            // fill array and pad end with 0's
            cudaMemcpy(data_1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // call kernel
            int blockSize = 128;
            dim3 fullBlocksPerGrid((power_of_2 + blockSize - 1) / blockSize);

            //timer().startGpuTimer();
            for (int d = 1; d <= ilog2ceil(n); d++) {
                int offset = pow(2, d - 1);
                iteration << <fullBlocksPerGrid, blockSize >> > (power_of_2, d, data_1, data_2, offset);
                int* temp = data_1;
                data_1 = data_2;
                data_2 = temp;
            }

            inclusiveToExclusive << <fullBlocksPerGrid, blockSize >> > (power_of_2, data_1, data_2);
            int* temp = data_1;
            data_1 = data_2;
            data_2 = temp;
            //timer().endGpuTimer();

            // set the out data to the scanned data
            cudaMemcpy(odata, data_1, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(data_1);
            cudaFree(data_2);
        }
    }
}
