#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void prescan(int n, float* g_odata, float* g_idata) {
            extern __shared__ float temp[];
            // allocated on invocation 
            int thid = threadIdx.x;
            int offset = 1;
            temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
            temp[2*thid+1] = g_idata[2*thid+1];

            // build sum in place up the tree
            for (int d = n >> 1; d > 0; d >>= 1) {
                __syncthreads();    
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }
            if (thid == 0) { temp[n - 1] = 0; } // clear the last element
            
            // traverse down tree & build scan
            for (int d = 1; d < n; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    float t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();
            g_odata[2 * thid] = temp[2 * thid];
            // write results to device memory
            g_odata[2*thid+1] = temp[2*thid+1];
        }

        __device__ void kernUpSweep(int n, int pN, int* idata, int offset, int d) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= pN)
                return;

            if (idx < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                temp[bi] += temp[ai];
            }
        }

        __device__ void kernDownSweep() {

        }

        __global__ void kernExScan(int n, int pN) {

        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            return;

            int* dev_idata;

            int depth = ilog2ceil(n);
            // remember numbers are read from right to left
            int pN = 1 << depth;    // n rounded to the next power of 2 = n after padding

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idata failed!");

            // for most gpus there are 1024 threads per block
            int threadsPerBlock = 1024;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // ceiling of n / threadsPerBlock
            dim3 blockDim(threadsPerBlock, 0, 0);
            dim3 gridDim(blocksPerGrid, 0, 0);


            timer().startGpuTimer();
            // TODO
            // kernScan << <gridDim, blockDim >> > ();
            // upsweep
            int offset = 1;
            for (int d = n >> 1; d > 0; d >>= 1) {
                kernUpSweep<<<gridDim, blockDim>>>();
                offset *= 2;


                kernExScan << <gridDim, blockDim >> > (pN, dev_temp, dev_odata, dev_idata, dev_ping, dev_pong, offset, pingpong);
                checkCUDAError("kernExScan failed!");

                vector<int> temp_test(pN);
                cudaMemcpy(temp_test.data(), dev_ping, sizeof(int) * pN, cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_temp to temp_test failed!");
                printArray(pN, temp_test.data(), false);

                pingpong = 1 - pingpong;
                /*cudaMemcpy(dev_temp, dev_ping, pN * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(dev_ping, dev_pong, pN * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(dev_pong, dev_temp, pN * sizeof(int), cudaMemcpyDeviceToDevice);*/
                int* temp = dev_ping;
                dev_ping = dev_pong;
                dev_pong = temp;
            }
            timer().endGpuTimer();

            cudaFree(dev_idata);
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
        using namespace StreamCompaction::Common;
        int compact(int n, int *odata, const int *idata) {
            return -1;

            int* dev_idata;
            int* dev_odata;
            bool* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idata failed!");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMalloc((void**)&dev_bools, n * sizeof(bool));
            checkCUDAError("cudaMalloc dev_mask failed!");

            cudaMalloc((void**)&dev_indices, n * sizeof(bool));
            checkCUDAError("cudaMalloc dev_indices failed!");

            // for most gpus there are 1024 threads per block
            int threadsPerBlock = 1024;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // ceiling of n / threadsPerBlock
            dim3 blockDim(threadsPerBlock, 0, 0);
            dim3 gridDim(blocksPerGrid, 0, 0);


            timer().startGpuTimer();
            // TODO
            int k = ilog2ceil(n);
            // step 1: compute dev_bools = determine which elements should be purged
            // kernMapToBoolean<<<gridDim, blockDim>>>(n, dev_bools, dev_idata);
            // step 2: exclusive scan on dev_bools
            // kernScan<<<gridDim, blockDim>>>(n, dev_indices, dev_bools);
            // step 3: reduce the array based on bools
            // kernScatter<<<gridDim, blockDim>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return -1;
        }
    }
}
