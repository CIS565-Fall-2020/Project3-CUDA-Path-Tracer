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
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
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
            int counter = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[counter] = idata[i];
                    counter++;
                }
            }
            timer().endCpuTimer();
            return counter;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // Construct criteria vector
            int* criteriaVec = new int[n];
            for (int i = 0; i < n; i++) {
                criteriaVec[i] = idata[i] != 0 ? 1 : 0;
            }

            // Construct scan vector from criteria vector
            int* scanVec = new int[n];
            scanVec[0] = 0;
            for (int i = 1; i < n; i++) {
                scanVec[i] = scanVec[i - 1] + criteriaVec[i - 1];
            }

            // Scatter
            int counter = 0;
            for (int i = 0; i < n; i++) {
                if (criteriaVec[i] == 1) {
                    odata[scanVec[i]] = idata[i];
                    counter++;
                }
            }

            delete[] criteriaVec;
            delete[] scanVec;
            timer().endCpuTimer();
            return counter;
        }
    }
}
