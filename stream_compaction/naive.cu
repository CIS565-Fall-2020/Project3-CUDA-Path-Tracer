#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <stack>

#define blockSize 128

//#define SHARED_MEMORY 1

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScanParallel(int d, int n, int* odata, int* idata) 
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            if (index == 0) 
            {
                odata[index] = 0;
            }

            if (index >= n)
                return;

#ifdef SHARED_MEMORY    
            __shared__ int temp[2 * blockSize];
            int pout = 0;
            int pin = 1;

            temp[pout * blockSize + threadIdx.x] = index > 0 ? idata[index] : 0;
            __syncthreads();

            pout = 1 - pout;
            pin = 1 - pout;

            if (threadIdx.x < powf(2.0f, d - 1)) 
                temp[pout * blockSize + threadIdx.x] = temp[pin * blockSize + threadIdx.x];
            else 
                temp[pout * blockSize + threadIdx.x] = temp[pin * blockSize + threadIdx.x] + temp[pin * blockSize + threadIdx.x - (int)powf(2.0f, d - 1)];

            __syncthreads();
            odata[index] = temp[pout * blockSize + threadIdx.x];
#else
            if (index < powf(2.0f, d - 1))
            {
                odata[index] = idata[index];
            }
            else
            {
                int forwardIndex = index - powf(2.0f, d - 1);
                odata[index] = idata[index] + idata[forwardIndex];
            }
#endif
        }

        // Initialize Extra Memory
        __global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < N) {
                intBuffer[index] = value;
            }
        }

        __global__ void kernBlockFinalValue(int n, int* finalValues, int* idata) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index == 0) 
            {
                finalValues[index] = 0;
                return;
            }
                

            if (index * blockSize - 1 >= n)
                return;

            finalValues[index] = idata[index * blockSize - 1];
        }

        __global__ void kernAddFinalSum(int n, int* finalSum, int* odata, int* idata) 
        {
            __shared__ int finalSumEntry;
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (index >= n)
                return;

            if (threadIdx.x == 0) 
            {
                finalSumEntry = finalSum[(int)floorf(index / blockSize)];
            }
            __syncthreads();

            odata[index] = idata[index] + finalSumEntry;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int* dev_idata; // Result Array
            int* dev_odata;
            int* dev_finalData;

#ifdef SHARED_MEMORY
            std::stack<int*> multiIndexingStack;
            std::stack<int> indexingLengthStack;
            int curN = pow(2, ilog2ceil(n));

            dim3 scanBlocksPerGrid((curN + blockSize - 1) / blockSize);
            cudaMalloc((void**)&dev_idata, curN * sizeof(int) + sizeof(int));
            kernResetIntBuffer << <scanBlocksPerGrid, blockSize >> > (curN, dev_idata, 0);
            checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
            cudaMemcpy(dev_idata + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            while (true)
            {
                if (curN != pow(2, ilog2ceil(n)))
                {
                    dev_idata = dev_finalData;
                }

                cudaMalloc((void**)&dev_odata, curN * sizeof(int));
                cudaMalloc((void**)&dev_finalData, (int)ceilf(curN / blockSize) * sizeof(int));
                int sumN = ilog2ceil(blockSize);
                dim3 scanBlocksPerGrid((curN + blockSize - 1) / blockSize);
                for (int i = 1; i <= sumN + 1; i++)
                {
                    kernScanParallel<<<scanBlocksPerGrid, blockSize >> > (i, n, dev_odata, dev_idata);
                    int* dev_temp;

                    dev_temp = dev_idata;
                    dev_idata = dev_odata;
                    dev_odata = dev_temp;
                }

                multiIndexingStack.push(dev_odata);
                indexingLengthStack.push(curN);

                if (curN <= blockSize)
                    break;

                dim3 finalValueBlockPerGrid = ((int)ceilf(curN / blockSize) + blockSize - 1) / blockSize;
                kernBlockFinalValue<<<finalValueBlockPerGrid, blockSize>>>(curN, dev_finalData, dev_odata);
                curN = (int)ceilf(curN / blockSize);
            }


            while(true) 
            {
                dev_finalData = multiIndexingStack.top();
                multiIndexingStack.pop();

                if (multiIndexingStack.empty()) 
                {
                    //cudaFree(dev_finalData);
                    break;
                }
                    

                dev_odata = multiIndexingStack.top();

                indexingLengthStack.pop();
                curN = indexingLengthStack.top();

                dim3 scanBlocksPerGrid((curN + blockSize - 1) / blockSize);
                kernAddFinalSum<<<scanBlocksPerGrid, blockSize>>>(curN, dev_finalData, dev_odata, dev_odata);
                //cudaFree(dev_finalData);
            }

             timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
#else
            int sumN = ilog2ceil(n);
            dim3 scanBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            cudaMalloc((void**)&dev_idata, n * sizeof(int) + sizeof(int));
            cudaMemcpy(dev_idata + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            for (int i = 1; i <= sumN + 1; i++) 
            {
                kernScanParallel<<<scanBlocksPerGrid, blockSize>>>(i, n, dev_odata, dev_idata);

                int* dev_temp;

                dev_temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = dev_temp;
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
#endif
            
        }
    }
}
