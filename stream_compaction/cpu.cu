#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int sum = 0;
            for (int i = 0; i < n; i++) 
            {
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int oCount = 0;
            for (int i = 0; i < n; i++) 
            {
                if (idata[i] != 0) 
                {
                    odata[oCount] = idata[i];
                    oCount++;
                }
            }
            timer().endCpuTimer();
            return oCount;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* mapArray = new int[n]();
            int* scannedArray = new int[n]();
            int sumIndex = 0;
            for (int i = 0; i < n; i++) 
            {
                if (idata[i] != 0) 
                {
                    mapArray[i] = 1;
                    scannedArray[i] = sumIndex;
                    sumIndex++;
                }
                else 
                {
                    mapArray[i] = 0;
                    scannedArray[i] = sumIndex;
                }
            }
            int oCount = 0;

            for (int i = 0; i < n; i++) 
            {
                if (mapArray[i] != 0) 
                {
                    odata[scannedArray[i]] = idata[i];
                    oCount++;
                }   
            }

            timer().endCpuTimer();
            return oCount;
        }
    }
}
