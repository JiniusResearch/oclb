/**********************************************************************
Copyright �2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef CONSTANT_BANDWIDTH_H_
#define CONSTANT_BANDWIDTH_H_

#define GROUP_SIZE 64
#define NUM_READS 256
#define WAVEFRONT 64
#define NUM_KERNELS 4

//Header Files
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.124.2"

using namespace appsdk;

/**
 * ConstantBandwidth
 * Class implements OpenCL Constant Buffer Bandwidth sample
 */

class ConstantBandwidth
{
        cl_double           setupTime;      /**< Time for setting up OpenCL */
        cl_double     totalKernelTime;      /**< Time for kernel execution */
        cl_double    totalProgramTime;      /**< Time for program execution */
        cl_uint                length;      /**< Length of the input data */
        cl_float               *input;      /**< Input array */
        cl_float              *output;      /**< Output array */

        cl_float  *verificationOutput;      /**< host output array */
        cl_context            context;      /**< CL context */
        cl_device_id         *devices;      /**< CL device list */
        cl_mem         constantBuffer;      /**< CL memory buffer */
        cl_mem           outputBuffer;      /**< CL memory buffer */


        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program            program;      /**< CL program */
        cl_kernel kernel[NUM_KERNELS];      /**< CL kernel */
        size_t          globalThreads;
        size_t           localThreads;

        size_t    kernelWorkGroupSize;      /**< Group Size returned by kernel */
        int
        iterations;      /**< Number of iterations for kernel execution */
        int
        vectorSize;      /**< Number of vector elements, default = 1 */
        bool dynamiArgFlag;
        bool vec3;
        SDKDeviceInfo deviceInfo;  /**< Structure to store device information*/

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        ConstantBandwidth()
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            input = NULL;
            output = NULL;
            verificationOutput = NULL;
            setupTime = 0;
            totalKernelTime = 0;
            iterations = 500;
            length = 1024 * 1024;
            vectorSize = 4;
            globalThreads = length;
            localThreads = GROUP_SIZE;
            dynamiArgFlag = false;
            vec3 = false;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupConstantBandwidth();

        /**
         * OpenCL related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build CL kernel program executable
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupCL();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runCLKernels();

        /**
         * Run bandwidth kernel and write values to output
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int bandwidth(cl_kernel &kernel);

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int initialize();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int verifyResults();
};


#endif
