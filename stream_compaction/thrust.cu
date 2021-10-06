#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::host_vector<int> hv_in(n);
            thrust::device_vector<int> dv_in = hv_in;

            for (int i = 0; i < n; i++) {
                dv_in[i] = idata[i];
            }
            thrust::device_vector<int> dv_out(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();

            for (int i = 0; i < n; i++) {
                odata[i] = dv_out[i];
            }
        }
    }
}
