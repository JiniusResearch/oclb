#include "ComputeBench.hpp"


//Separator
std::string sep = "-----------------------------------------";

template<typename T>
int ComputeBench::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
        size_t sizeInBytes, cl_map_flags flags)
{
    cl_int status;
    hostPointer = (T*) clEnqueueMapBuffer(commandQueue,
            deviceBuffer,
            CL_TRUE,
            flags,
            0,
            sizeInBytes,
            0,
            NULL,
            NULL,
            &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed");

    return SDK_SUCCESS;
}

int
ComputeBench::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
{
    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue,
            deviceBuffer,
            hostPointer,
            0,
            NULL,
            NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

    return SDK_SUCCESS;
}

int
ComputeBench::setupComputeBench()
{
    return SDK_SUCCESS;
}

int
ComputeBench::genBinaryImage()
{
    return 0;
}

int
ComputeBench::setupCL(void)
{
    cl_int status = 0;
    cl_device_type dType;

    if (sampleArgs->deviceType.compare("cpu") == 0) {
        dType = CL_DEVICE_TYPE_CPU;
    } else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if (sampleArgs->isThereGPU() == false) {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId, sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    /*
     * If we could find our platform, use it. Otherwise use just available platform.
     */
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platform,
        0
    };

    context = clCreateContextFromType(cps,
            dType,
            NULL,
            NULL,
            &status);
    CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
            sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    std::string deviceStr(deviceInfo.deviceVersion);
    size_t vStart = deviceStr.find(" ", 0);
    size_t vEnd = deviceStr.find(" ", vStart + 1);
    std::string vStrVal = deviceStr.substr(vStart + 1, vEnd - vStart - 1);


    // OpenCL 1.1 has inbuilt support for vec3 data types
    if (vec3 == true) {
        OPENCL_EXPECTED_ERROR("Device doesn't support built-in 3 component vectors!");
    }
    // The block is to move the declaration of prop closer to its use
    /* Note: Using deprecated clCreateCommandQueue as CL_QUEUE_PROFILING_ENABLE flag not currently working 
     ***with clCreateCommandQueueWithProperties*/
    cl_command_queue_properties prop = 0;
    prop |= CL_QUEUE_PROFILING_ENABLE;

    commandQueue = clCreateCommandQueue(context,
            devices[sampleArgs->deviceId],
            prop,
            &status);
    CHECK_OPENCL_ERROR(status, "clCreateCommandQueue failed.");

    if (sampleArgs->isLoadBinaryEnabled()) {
        // Always assuming kernel was dumped for vector-width 1
        if (vectorSize != 0) {
            std::cout <<
                    "Ignoring specified vector-width. Assuming kernel was dumped for vector-width 1"
                    << std::endl;
        }
        vectorSize = 1;
    } else {
        // If vector-size is not specified in the command-line, choose the preferred size for the device
        if (vectorSize == 0) {
            vectorSize = deviceInfo.preferredFloatVecWidth;
        } else if (vectorSize == 3) {
            //Make vectorSize as 4 if -v option is 3.
            //This memory alignment is required as per OpenCL for type3 vectors
            vec3 = true;
            vectorSize = 4;
        } else if ((1 != vectorSize) && (2 != vectorSize) && (4 != vectorSize) &&
                (8 != vectorSize) && (16 != vectorSize)) {
            std::cout << "The vectorsize can only be one of 1,2,3(4),4,8,16!" << std::endl;
            return SDK_FAILURE;
        }
    }


    outputKadd = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof (cl_float) * vectorSize * length, 0, &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (outputKadd)");

    // create a CL program using the kernel source
    char buildOption[512];
    if (vectorSize == 1) {
        sprintf(buildOption, "-D DATATYPE=uint -D DATATYPE2=uint4 ");
        //sprintf(buildOption, "-D DATATYPE=float -D DATATYPE2=float4 ");
    } else {
        sprintf(buildOption, "-D DATATYPE=uint%d -D DATATYPE2=uint%d ", (vec3 == true) ? 3 : vectorSize, (vec3 == true) ? 3 : vectorSize);
        //sprintf(buildOption, "-D DATATYPE=float%d -D DATATYPE2=float%d ", (vec3 == true) ? 3 : vectorSize, (vec3 == true) ? 3 : vectorSize);
    }

    strcat(buildOption, "-D IDXTYPE=uint ");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("ComputeBench.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string(buildOption);
    if (sampleArgs->isLoadBinaryEnabled()) {
        buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
    }

    if (sampleArgs->isComplierFlagsSpecified()) {
        buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

    // Global memory bandwidth from read-single access
    kernel[0] = clCreateKernel(program, "Kadd", &status);
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed.(Kadd)");

    return SDK_SUCCESS;
}

int
ComputeBench::bandwidth(cl_kernel &kernel,
        cl_mem outputBuffer,
        double *timeTaken,
        double *gbps
        )
{
    cl_int status;

    // Check group size against kernelWorkGroupSize
    status = clGetKernelWorkGroupInfo(kernel,
            devices[sampleArgs->deviceId],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof (size_t),
            &kernelWorkGroupSize,
            0);
    CHECK_OPENCL_ERROR(status, "clGetKernelWorkGroupInfo failed.");

    if (localThreads > kernelWorkGroupSize) {
        localThreads = kernelWorkGroupSize;
    }

    //Set appropriate arguments to the kernel
    int argIndex = 0;
    {
        status = clSetKernelArg(kernel,
                argIndex++,
                sizeof (cl_mem),
                (void *) &outputBuffer);
        CHECK_OPENCL_ERROR(status, "clSetKernelArg failed.(outputBuffer)");
    }

    double sec = 0;
    int iter = iterations;

    // Run the kernel for a number of iterations
    for (int i = 0; i < iter; i++) {
        // Enqueue a kernel run call
        cl_event ndrEvt;
        status = clEnqueueNDRangeKernel(commandQueue,
                kernel,
                1,
                NULL,
                &globalThreads,
                &localThreads,
                0,
                NULL,
                &ndrEvt);
        CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");

        // wait for the kernel call to finish execution
        status = clWaitForEvents(1, &ndrEvt);
        CHECK_OPENCL_ERROR(status, "clWaitForEvents failed.");

        // Calculate performance
        cl_ulong startTime;
        cl_ulong endTime;

        // Get kernel profiling info
        status = clGetEventProfilingInfo(ndrEvt,
                CL_PROFILING_COMMAND_START,
                sizeof (cl_ulong),
                &startTime,
                0);
        CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(startTime)");

        status = clGetEventProfilingInfo(ndrEvt,
                CL_PROFILING_COMMAND_END,
                sizeof (cl_ulong),
                &endTime,
                0);
        CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(endTime)");

        // Cumulate time for each iteration
        sec += 1e-9 * (endTime - startTime);

        status = clReleaseEvent(ndrEvt);
        CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(endTime)");

        status = clFinish(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFinish failed");
    }

    // Copy bytes
    int bytesPerThread = FORLOOP;
    double bytes = (double) (iter * bytesPerThread);
    double perf = (bytes / sec) * 1e-9;
    perf *= globalThreads * vectorSize;

    *gbps = perf;
    *timeTaken = sec / iter;

    return SDK_SUCCESS;
}

int
ComputeBench::runCLKernels(void)
{
    std::cout << "Executing kernel for " << iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // Measure bandwidth of uncached linear reads from global buffer
    int status = bandwidth(kernel[0], outputKadd, &KaddTime, &KaddGbps);
    if (status != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
ComputeBench::initialize()
{
    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize()) {
        return SDK_FAILURE;
    }

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "num_iterators memory allocation failed");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

    Option* num_arguments = new Option;
    CHECK_ALLOCATION(num_arguments, "num_arguments memory allocation failed");

    num_arguments->_sVersion = "c";
    num_arguments->_lVersion = "components";
    num_arguments->_description =
            "Number of vector components to be used. Can be either 1,2,3(4),4,8,16";
    num_arguments->_type = CA_ARG_INT;
    num_arguments->_value = &vectorSize;

    sampleArgs->AddOption(num_arguments);
    delete num_arguments;

    return SDK_SUCCESS;
}

int
ComputeBench::setup()
{
    if (iterations < 1) {
        std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int status = setupCL();
    if (status != SDK_SUCCESS) {
        return status;
    }

    if (setupComputeBench() != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    setupTime = (cl_double) sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}

int
ComputeBench::run()
{
    bool useSVM = false;

    // Arguments are set and execution call is enqueued on command buffer
    if (runCLKernels() != SDK_SUCCESS) {
        return SDK_FAILURE;
    }
    if (sampleArgs->verify && verifyResults() != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    printStats();

    return SDK_SUCCESS;
}

int
ComputeBench::verifyResults()
{
    if (sampleArgs->verify) {
        int vecElements = (vec3 == true) ? 3 : vectorSize;
        int sizeElement = vectorSize * sizeof (cl_float);
        //int readLength = length + (NUM_READS * 1024 / sizeElement) + EXTRA_ELEMENTS;
        int status, passStatus;

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        std::cout << "\nVerifying results for KAdd : " << std::endl;

        // Map cl_mem outputKadd to host for reading
        status = mapBuffer(outputKadd, outputKaddHost, (length * sizeElement), CL_MAP_READ);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputKadd)");

        passStatus = 1;
        uint* devBuffer = (uint *) outputKaddHost;

        for (int i = 0; i < length; i++) {
            
            for (int j = 0; j < vecElements; j++) {
                //uint answer = i+j;
                uint answer = i;
                for (int ii = 0; ii < 1000; ii++) {
                    //answer ^= ii;
                    //answer = answer << (ii ) | answer >> (32 - ii );
                    //answer += ii;
                    
                    answer += ii;
                    answer = answer ^ ii;
                    answer++;
                }
//                std::cout << " gid:" << i << " vec:" << j << " answer:" << answer << " result:" << devBuffer[j] << std::endl;
                if (devBuffer[j] != answer)
                    passStatus = 0;
            }
            if (passStatus != 1)
                break;

            devBuffer += vectorSize;
        }

        status = unmapBuffer(outputKadd, outputKaddHost);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(outputKadd)");

        if (passStatus == 1) {
            std::cout << "Passed!\n" << std::endl;
        } else {
            std::cout << "Failed!\n" << std::endl;
            return SDK_FAILURE;
        }

    }

    return SDK_SUCCESS;
}

void
ComputeBench::printStats()
{
    std::string strArray[3];
    std::string stats[3];
    sampleArgs->timing = true;

    int sizeInBytesPerIter = (int) (NUM_READS * vectorSize * sizeof (
            cl_float) * globalThreads);
    std::cout << std::endl << std::setw(18) << std::left
            << "Vector width used " << ": " << ((vec3) ? 3 : vectorSize) << std::endl;
    std::cout << std::setw(18) << std::left
            << "Setup Time " << ": " << setupTime << " secs" << std::endl << std::endl;

    std::cout << "\n1.  Add" << std::endl;
    strArray[0] = "Times";
    stats[0] = toString(sizeInBytesPerIter, std::dec);
    strArray[1] = "Avg. Kernel Time (sec)";
    stats[1] = toString(KaddTime, std::dec);
    strArray[2] = "Avg Throughput ( GRPS )";
    stats[2] = toString(KaddGbps, std::dec);
    printStatistics(strArray, stats, 3);
}

int
ComputeBench::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseMemObject(outputKadd);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputKadd)");

    for (int i = 0; i < NUM_KERNELS; i++) {
        status = clReleaseKernel(kernel[i]);
        CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");
    }

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed. (context)");

    FREE(devices);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    cl_int status = 0;
    ComputeBench clComputeBench;

    if (clComputeBench.initialize() != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    if (clComputeBench.sampleArgs->parseCommandLine(argc,
            argv) != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    if (clComputeBench.sampleArgs->isDumpBinaryEnabled()) {
        return clComputeBench.genBinaryImage();
    }

    status = clComputeBench.setup();
    if (status != SDK_SUCCESS) {
        if (status == SDK_EXPECTED_FAILURE) {
            return SDK_SUCCESS;
        }

        return SDK_FAILURE;
    }

    if (clComputeBench.run() != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    if (clComputeBench.cleanup() != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}
