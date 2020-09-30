#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int d, int* in1, int* in2, int* out, int pow2_d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) return;
            if (index >= pow2_d) {
                out[index] = in1[index - pow2_d] + in1[index];
            }
            in2[index] = out[index];
        }

        __global__ void kernShiftArray(int n, int* in1, int* in2) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) return;
            if (index == 0) in2[index] = 0;
            else {
                in2[index] = in1[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // first allocate buffers and define kernel parameters
            int* input;
            int* input_temp;
            int* output;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            cudaMalloc((void**)&input, n * sizeof(int));
            cudaMalloc((void**)&input_temp, n * sizeof(int));
            cudaMalloc((void**)&output, n * sizeof(int));
            cudaMemcpy(input, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(input_temp, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(output, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // append identity to beginning and shift array
            kernShiftArray <<< fullBlocksPerGrid, blockSize >> > (n, input, input_temp);
            std::swap(input_temp, input);
            // make ilog2ceil(n) kernel calls for scan
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                int pow2 = 1 << (d - 1);
                kernNaiveScan <<< fullBlocksPerGrid, blockSize >> > (n, d, input, input_temp, output, pow2);
                std::swap(input, input_temp);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, input, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(input);
            cudaFree(input_temp);
            cudaFree(output);
        }
    }
}
