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
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = idata[i - 1] + odata[i - 1];
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
            int index = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[index] = idata[i];
                    index++;
                }
            }
            timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            //timer().startCpuTimer();
            // create a new array mapping the input array to zero's and one's
            int* zerosAndOnes = new int[n];
            for (int i = 0; i < n; i++) {
                idata[i] == 0 ? zerosAndOnes[i] = 0 : zerosAndOnes[i] = 1;
            }
            
            // scan new array
            int* scannedArray = new int[n];
            scan(n, scannedArray, zerosAndOnes);

            //scatter
            for (int i = 0; i < n; i++) {
                if (zerosAndOnes[i] == 1) {
                    odata[scannedArray[i]] = idata[i];
                }
            }
            
            //timer().endCpuTimer();
            return scannedArray[n-1];
        }
    }
}
