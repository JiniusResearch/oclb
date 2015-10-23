#ifndef COMPUTE_BENCH_H_
#define COMPUTE_BENCH_H_


#define GROUP_SIZE 128
#define NUM_READS 2
#define NUM_KERNELS 1
#define NUM_INPUT 4096 * 128
#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.124.3"

//Header Files
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

using namespace appsdk;

/**
 * ComputeBench
 * Class implements OpenCL Global Buffer Bandwidth sample
 */

class ComputeBench {
    cl_double setupTime; /**< Time for setting up OpenCL */
    cl_uint length; /**< Length of the input data */
    cl_uint readRange;
    
    cl_float *input; /**< Input array */
    
    
    cl_float *verificationOutput; /**< Output Array for Verification */
    cl_context context; /**< CL context */
    cl_device_id *devices; /**< CL device list */

    cl_mem inputBuffer; /**< input buffer */
    cl_mem outputKadd;
    
    cl_mem constValue;

    

    cl_command_queue commandQueue; /**< CL command queue */
    cl_program program; /**< CL program */
    cl_kernel kernel[NUM_KERNELS]; /**< CL kernel */
    size_t globalThreads;
    size_t localThreads;


    size_t kernelWorkGroupSize; /**< Group Size returned by kernel */
    cl_ulong availableLocalMemory;
    cl_ulong neededLocalMemory;
    int
    iterations; /**< Number of iterations for kernel execution */
    int vectorSize; /**< float, float2, float4 */
    bool writeFlag;
    
    bool uncachedRead;
    bool vec3;
    
    double KaddGbps; /**< Record GBPS for every type of bandwidth test */
    
    double KaddTime; /**< Record time for every type of bandwidth test */
    
    SDKDeviceInfo deviceInfo; /**< Structure to store device information*/

    SDKTimer *sampleTimer; /**< SDKTimer object */

public:

    CLCommandArgs *sampleArgs; /**< CLCommand argument class */

    /**
     * Constructor
     * Initialize member variables
     */
    ComputeBench()
    {
        sampleArgs = new CLCommandArgs();
        sampleTimer = new SDKTimer();
        sampleArgs->sampleVerStr = SAMPLE_VERSION;
        input = NULL;
        outputKadd = NULL;
        verificationOutput = NULL;

        setupTime = 0;
        iterations = 100;
        length = NUM_INPUT;
        vectorSize =
                0; // Query the device later and select the preferred vector-size
        globalThreads = length;
        localThreads = GROUP_SIZE;
        writeFlag = false;

        uncachedRead = false;
        vec3 = false;
    }

    /**
     * Allocate and initialize host memory array with random values
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int setupComputeBench();

    /**
     * Override from SDKSample, Generate binary image of given kernel
     * and exit application
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int genBinaryImage();

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

    int runCLKernels_SVM();

    int bandwidth(cl_kernel &kernel,
            cl_mem outputBuffer,
            cl_float *outputSVMBuffer,
            double *timeTaken,
            double *gbps,
            bool useSVM);

    /**
     * Override from SDKSample. Print sample stats.
     */
    void printStats();

    /**
     * Override from SDKSample. Initialize
     * command line parser, add custom options
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int initialize();

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
    int verifyResults(bool useSVM);

private:

    /**
     * A common function to map cl_mem object to host
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    template<typename T>
    int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
            cl_map_flags flags = CL_MAP_READ);

    /**
     * A common function to unmap cl_mem object from host
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);

    int createSVMBuffers(cl_uint bufferSize, cl_uint inputExtraBufferSize);

    bool is64Bit();
};
#endif
