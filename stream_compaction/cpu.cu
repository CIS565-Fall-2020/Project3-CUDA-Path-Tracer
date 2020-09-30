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
            for (int i = 0; i < n; ++i) {
                int prefix_idx = i - 1;
                if (prefix_idx < 0) {
                    odata[i] = 0;
                }
                else {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            } 
            timer().endCpuTimer();
        }

        /**
        * CPU scan (prefix sum) as a helper method.
        * For performance analysis, this is supposed to be a simple for loop.
        */
        void scanImplementation(int n, int* odata, const int* idata) {
            for (int i = 0; i < n; ++i) {
                int prefix_idx = i - 1;
                if (prefix_idx < 0) {
                    odata[i] = 0;
                }
                else {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; ++i) {
                int elem = idata[i];
                if (elem != 0) {
                    odata[count] = elem;
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* temp = new int[n]; // create temp array
            // fill temp array with 0s and 1s
            for (int i = 0; i < n; ++i) {
                int elem = idata[i];
                if (elem != 0) {
                    temp[i] = 1;
                }
                else {
                    temp[i] = 0;
                }
            }
            // run scan
            int* scanned = new int[n] {0};
            StreamCompaction::CPU::scanImplementation(n, scanned, temp);
            // scatter
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (temp[i] == 1) {
                    odata[scanned[i]] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            delete[] scanned;
            delete[] temp;
            return count;
        }
    }
}
