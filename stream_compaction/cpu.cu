#include <cstdio>
#include <vector>
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
            // timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++)
                odata[i] = odata[i - 1] + idata[i - 1];
            // timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            std::vector<int> o = std::vector<int>();
            for (int i = 0; i < n; i++)
                if (idata[i])
                    o.push_back(idata[i]);
            odata = o.data();
            timer().endCpuTimer();
            return o.size();
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // step 1: compute bit mask
            std::vector<int> mask(n);
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    mask.at(i) = 0;
                }
                else {
                    mask.at(i) = 1;
                }
            }

            // step 2: exclusive scan 
            scan(n, odata, mask.data());

            timer().endCpuTimer();
            return -1;

            // step 3: scatter
            int m = odata[n - 1];
            std::vector<int> ovec(m);
            m = 0;
            for (int i = 0; i < n; i++) {
                if (mask[i]) {
                    ovec[odata[i]] = idata[i];
                    m++;
                }
            }

            odata = ovec.data();

            timer().endCpuTimer();
            return m;
        }
    }
}
