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
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin

            //thrust::exclusive_scan(idata, idata + n, odata);
            
            int* dev_idata, * dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);


            thrust::device_ptr<int> dev_thrust_odata(dev_odata);
            thrust::device_ptr<int> dev_thrust_idata(dev_idata);

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);
            
            timer().endGpuTimer();

            cudaMemcpy(odata, thrust::raw_pointer_cast(dev_thrust_odata), n * sizeof(int), cudaMemcpyDeviceToHost);
            /*
            for (int i = 0; i < n; i++) {
                //printf("%d  ", odata[i]);
            }
            */
        }
    }
}
