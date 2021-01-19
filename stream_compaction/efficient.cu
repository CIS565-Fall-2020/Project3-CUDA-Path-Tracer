#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        // Up-Sweep
        __global__ void kernUpSweep(int n, int d, int* sweepArray) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            int power = 1 << (d + 1);
            int lastPower = 1 << d;
            int arrayIndex = index * power;
            if (arrayIndex >= n)
                return;

            sweepArray[arrayIndex + power - 1] += sweepArray[arrayIndex + lastPower - 1];
        }

        // Down-Sweep
        __global__ void kernDownSweep(int n, int d, int* sweepArray) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int power = 1 << (d + 1);
            int lastPower = 1 << d;
            int arrayIndex = index * power;

            if (arrayIndex >= n)
                return;

            int t = sweepArray[arrayIndex + lastPower - 1];
            sweepArray[arrayIndex + lastPower - 1] = sweepArray[arrayIndex + power - 1];
            sweepArray[arrayIndex + power - 1] += t;
        }

        // Initialize Extra Memory
        __global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < N) {
                intBuffer[index] = value;
            }
        }
        
        // Non-zero Entry Scan
        __global__ void kernNonZeroScan(int n, int roundN, int* zeroArray, int* iArray) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= roundN)
                return;
            else if (index >= n) 
            {
                zeroArray[index] = 0;
                return;
            }

            if (iArray[index] == 0)
                zeroArray[index] = 0;
            else
                zeroArray[index] = 1;
        }

        __global__ void kernCompact(int n, int* zeroArray, int* idxArray, int* oArray, int* iArray) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= n)
                return;

            if (zeroArray[index] == 1) 
            {
                oArray[idxArray[index]] = iArray[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int scanN = ilog2ceil(n) - 1;

            int roundCount = pow(2, scanN + 1);

            int* result;
            cudaMalloc((void**)&result, roundCount * sizeof(int));
            dim3 roundInitilize = (roundCount + blockSize - 1) / blockSize;
            kernResetIntBuffer<<<roundInitilize, blockSize>>>(roundCount, result, 0);

            cudaMemcpy(result, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int d = 0; d <= scanN; d++) 
            {
                dim3 upSweepBlockPerGrid = (roundCount / powf(2, d + 1) + blockSize - 1) / blockSize;
                kernUpSweep<<<upSweepBlockPerGrid, blockSize>>>(roundCount, d, result);
            }
            
            cudaMemset(result + roundCount - 1, 0, sizeof(int));

            for (int d = scanN; d >= 0; d--)
            {
                dim3 upSweepBlockPerGrid = (roundCount / powf(2, d + 1) + blockSize - 1) / blockSize;
                kernDownSweep<<<upSweepBlockPerGrid, blockSize>>>(roundCount, d, result);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, result, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(result);
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
            timer().startGpuTimer();
            // TODO
            int scanN = ilog2ceil(n) - 1;
            int roundCount = pow(2, scanN + 1);

            int* roundZeroArray;
            cudaMalloc((void**)&roundZeroArray, roundCount * sizeof(int));

            int* cudaIData;
            cudaMalloc((void**)&cudaIData, n * sizeof(int));
            cudaMemcpy(cudaIData, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 zeroScanBlockPerGrid = (roundCount + blockSize - 1) / blockSize;

            StreamCompaction::Common::kernMapToBoolean<<<zeroScanBlockPerGrid, blockSize>>>(n, roundCount, roundZeroArray, cudaIData);

            int* roundIdxArray;
            cudaMalloc((void**)&roundIdxArray, roundCount * sizeof(int));
            cudaMemcpy(roundIdxArray, roundZeroArray, roundCount * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d <= scanN; d++)
            {
                dim3 upSweepBlockPerGrid = (roundCount / powf(2, d + 1) + blockSize - 1) / blockSize;
                kernUpSweep << <upSweepBlockPerGrid, blockSize >> > (roundCount, d, roundIdxArray);
            }

            int* a = new int[1]();
            a[0] = 0;
            cudaMemcpy(roundIdxArray + roundCount - 1, a, sizeof(int), cudaMemcpyHostToDevice);

            delete[]a;

            for (int d = scanN; d >= 0; d--)
            {
                dim3 upSweepBlockPerGrid = (roundCount / powf(2, d + 1) + blockSize - 1) / blockSize;
                kernDownSweep << <upSweepBlockPerGrid, blockSize >> > (roundCount, d, roundIdxArray);
            }

            int compactCountTemp = -1;
            cudaMemcpy(&compactCountTemp, roundIdxArray + roundCount - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int* result;
            cudaMalloc((void**)&result, n * sizeof(int));
            dim3 compactBlockPerGrid = (n + blockSize - 1) / blockSize;

            StreamCompaction::Common::kernScatter<<<compactBlockPerGrid, blockSize>>>(n, result, cudaIData, roundZeroArray, roundIdxArray);

            cudaMemcpy(odata, result, n * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            cudaFree(roundZeroArray);
            cudaFree(cudaIData);
            cudaFree(roundIdxArray);
            cudaFree(result);
            return compactCountTemp;
        }
    }

    namespace Radix 
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        // Scan each bit
        __global__ void kernScanBit(int n, int* iArray, int* bitArray, int* eArray, int bit, int* bitExist)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
                return;

            if ((iArray[index] >> bit) & 1)
            {
                if (bitExist[0] == 0)
                    bitExist[0] = 1;

                bitArray[index] = 1;
                eArray[index] = 0;
            }
            else
            {
                bitArray[index] = 0;
                eArray[index] = 1;
            }

        }

        __global__ void kernComputeTArray(int n, int totalFalses, int* tArray, int* fArray) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
                return;

            tArray[index] = index - fArray[index] + totalFalses;
        }

        __global__ void kernComputeIndex(int n, int* dArray, int* tArray, int* fArray, int* bArray) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
                return;

            if (bArray[index] == 1)
                dArray[index] = tArray[index];
            else
                dArray[index] = fArray[index];
        }

        __global__ void kernReorganizeArray(int n, int* dArray, int* iArray, int* oArray) 
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
                return;

            oArray[dArray[index]] = iArray[index];

        }

        void radixSort(int n, int* odata, const int* idata) 
        {
            int scanN = ilog2ceil(n) - 1;
            int roundCount = pow(2, scanN + 1);

            int* dev_iArray;
            int* dev_bArray;
            int* dev_eArray;
            int* dev_fArray;
            int* dev_tArray;
            int* dev_dArray;
            int* dev_oArray;

            cudaMalloc((void**)&dev_iArray, n * sizeof(int));
            cudaMemcpy(dev_iArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc((void**)&dev_bArray, n * sizeof(int));
            cudaMalloc((void**)&dev_eArray, n * sizeof(int));
            cudaMalloc((void**)&dev_fArray, roundCount * sizeof(int));
            cudaMalloc((void**)&dev_tArray, n * sizeof(int));
            cudaMalloc((void**)&dev_dArray, n * sizeof(int));
            cudaMalloc((void**)&dev_oArray, n * sizeof(int));

            int* isBitExist;
            cudaMalloc((void**)&isBitExist, sizeof(int));

            int curBit = 0;

            dim3 bitCheckBlocksPerGrid = (n + blockSize - 1) / blockSize;
            dim3 scanBlocksPerGrid = (roundCount + blockSize - 1) / blockSize;

            while (true) 
            {
                // Get bit condition for each entry
                cudaMemset(isBitExist, 0, sizeof(int));
                kernScanBit<<<bitCheckBlocksPerGrid, blockSize>>>(n, dev_iArray, dev_bArray, dev_eArray, curBit, isBitExist);
                int h_isBitExist = 0;
                cudaMemcpy(&h_isBitExist, isBitExist, sizeof(int), cudaMemcpyDeviceToHost);
                if (h_isBitExist == 0)
                    break;
                
                // Scan the e array
               
                dim3 roundInitilize = (roundCount + blockSize - 1) / blockSize;
                StreamCompaction::Efficient::kernResetIntBuffer<<<scanBlocksPerGrid, blockSize>>>(roundCount, dev_fArray, 0);

                cudaMemcpy(dev_fArray, dev_eArray, n * sizeof(int), cudaMemcpyDeviceToDevice);
                int eArrayFinal = 0;
                cudaMemcpy(&eArrayFinal, dev_fArray + roundCount - 1, sizeof(int), cudaMemcpyDeviceToHost);

                for (int d = 0; d <= scanN; d++)
                {
                    dim3 upSweepBlockPerGrid = (roundCount / powf(2, d + 1) + blockSize - 1) / blockSize;
                    StreamCompaction::Efficient::kernUpSweep<<<upSweepBlockPerGrid, blockSize>>>(roundCount, d, dev_fArray);
                }

                cudaMemset(dev_fArray + roundCount - 1, 0, sizeof(int));

                for (int d = scanN; d >= 0; d--)
                {
                    dim3 upSweepBlockPerGrid = (roundCount / powf(2, d + 1) + blockSize - 1) / blockSize;
                    StreamCompaction::Efficient::kernDownSweep<<<upSweepBlockPerGrid, blockSize>>>(roundCount, d, dev_fArray);
                }

               
                int fArrayFinal = 0;
                cudaMemcpy(&fArrayFinal, dev_fArray + roundCount - 1, sizeof(int), cudaMemcpyDeviceToHost);

                int totalFalses = eArrayFinal + fArrayFinal;

                kernComputeTArray<<<bitCheckBlocksPerGrid, blockSize>>>(n, totalFalses, dev_tArray, dev_fArray);
                kernComputeIndex<<<bitCheckBlocksPerGrid, blockSize>>>(n, dev_dArray, dev_tArray, dev_fArray, dev_bArray);
                kernReorganizeArray<<<bitCheckBlocksPerGrid, blockSize>>>(n, dev_dArray, dev_iArray, dev_oArray);

                cudaMemcpy(dev_iArray, dev_oArray, n * sizeof(int), cudaMemcpyDeviceToDevice);

                curBit++;
            }
            
            cudaMemcpy(odata, dev_iArray, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_iArray);
            cudaFree(dev_eArray);
            cudaFree(dev_bArray);
            cudaFree(dev_fArray);
            cudaFree(dev_tArray);
            cudaFree(dev_dArray);
            cudaFree(dev_oArray);
        }

        
    }
}

