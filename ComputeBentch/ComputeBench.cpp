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
    cl_uint sizeElement = vectorSize * sizeof (cl_float);
    cl_uint readLength = length + (NUM_READS * 1024 / sizeElement);

    /*
     * Map cl_mem inputBuffer to host for writing
     * Note the usage of CL_MAP_WRITE_INVALIDATE_REGION flag
     * This flag indicates the runtime that whole buffer is mapped for writing and
     * there is no need of device->host transfer. Hence map call will be faster
     */
    int status = mapBuffer(inputBuffer, input, (readLength * sizeElement), CL_MAP_WRITE_INVALIDATE_REGION);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");

    // random initialisation of input
    fillRandom<cl_float>(input,
            readLength * vectorSize,
            1,
            0,
            (cl_float) (readLength - 1));

    /* Unmaps cl_mem inputBuffer from host
     * host->device transfer happens if device exists in different address-space
     */
    status = unmapBuffer(inputBuffer, input);
    CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(inputBuffer)");


    return SDK_SUCCESS;
}

int
ComputeBench::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("ComputeBench.cl");

    // Always using vector-width of 1 to dump kernels
    if (vectorSize != 0) {
        std::cout <<
                "Ignoring specified vector-width. Always using vector-width of 1 to dump kernels"
                << std::endl;
    }
    vectorSize = 1;

    // Pass vectorSize as DATATYPE to kernel
    char buildOption[128];

    sprintf(buildOption, "-D DATATYPE=float -D DATATYPE2=float4 ");

    binaryData.flagsStr = std::string(buildOption);
    if (sampleArgs->isComplierFlagsSpecified()) {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    CHECK_ERROR(status, SDK_SUCCESS, "OpenCL Generate Binary Image Failed");
    return status;
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
    int retValue = getPlatform(platform, sampleArgs->platformId,
            sampleArgs->isPlatformEnabled());
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

    // inputBufferExtra does the highest single allocation of all
    // Check if this is allocatable, else reduce 'length'
    cl_ulong maxAllocation = sizeof (cl_float) * vectorSize * ((length * NUM_READS) + NUM_READS);
    while (maxAllocation > deviceInfo.maxMemAllocSize) {
        length /= 2;
        maxAllocation = sizeof (cl_float) * vectorSize * ((length * NUM_READS) + NUM_READS);
    }

    globalThreads = length;

    cl_uint sizeElement = vectorSize * sizeof (cl_float);
    cl_uint readLength = length + (NUM_READS * 1024 / sizeElement);

    readRange = readLength;
    cl_uint size = readLength * vectorSize * sizeof (cl_float);

    // Create input buffer
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, 0, &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputBuffer)");


    outputKadd = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof (cl_float) * vectorSize * length, 0, &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (outputKadd)");

    constValue = clCreateBuffer(context, CL_MEM_READ_ONLY, vectorSize * sizeof (cl_float), 0, &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed.(constValue)");

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
        cl_float *outputSVMBuffer,
        double *timeTaken,
        double *gbps,
        bool useSVM = false)
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
    if (!writeFlag) {
        {
            status = clSetKernelArg(kernel, argIndex++, sizeof (cl_mem), (void *) &inputBuffer);
            CHECK_OPENCL_ERROR(status, "clSetKernelArg failed.(inputBuffer)");
        }
    } else {
        // Pass a single constant value to kernel of type - float<vectorSize>
        cl_float *temp;

        // Map cl_mem constValue to host for writing
        status = mapBuffer(constValue, temp,
                (vectorSize * sizeof (cl_float)),
                CL_MAP_WRITE_INVALIDATE_REGION);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(constValue)");

        memset(temp, 0, vectorSize * sizeof (cl_float));

        /* Unmaps cl_mem constValue from host
         * host->device transfer happens if device exists in different address-space
         */
        status = unmapBuffer(constValue, temp);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(constValue)");

        status = clSetKernelArg(kernel,
                argIndex++,
                sizeof (cl_mem),
                (void *) &constValue);
        CHECK_OPENCL_ERROR(status, "clSetKernelArg failed.(constValue)");
    }

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
    int bytesPerThread = 100;
//    if (vec3 == true) {
//        bytesPerThread = NUM_READS * 3 * sizeof (cl_float);
//    } else {
//        bytesPerThread = NUM_READS * vectorSize * sizeof (cl_float);
//    }
    double bytes = (double) (iter * bytesPerThread);
    double perf = (bytes / sec) * 1e-9;
    perf *= globalThreads;

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
    int status = bandwidth(kernel[0], outputKadd, NULL, &KaddTime, &KaddGbps);
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
    if (sampleArgs->verify && verifyResults(useSVM) != SDK_SUCCESS) {
        return SDK_FAILURE;
    }

    printStats();

    return SDK_SUCCESS;
}

int
ComputeBench::verifyResults(bool useSVM = false)
{
    //    if (sampleArgs->verify) {
    //        int vecElements = (vec3 == true) ? 3 : vectorSize;
    //        int sizeElement = vectorSize * sizeof (cl_float);
    //        int readLength = length + (NUM_READS * 1024 / sizeElement) + EXTRA_ELEMENTS;
    //        int status, passStatus;
    //
    //        // Verify result for single access
    //        verificationOutput = (cl_float*) malloc(length * vectorSize * sizeof (cl_float));
    //        CHECK_ALLOCATION(verificationOutput,
    //                "verificationOutput memory allocation failed");
    //        {
    //            /*
    //             * Map cl_mem inputBuffer to host for reading
    //             * device->host transfer happens if device exists in different address-space
    //             */
    //            status = mapBuffer(inputBuffer, input,
    //                    (readLength * sizeElement),
    //                    CL_MAP_READ);
    //            CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(inputBuffer)");
    //        }
    //        ///////////////////////////////////////////////////////////////////////////////////////////////////
    //        std::cout << "\nVerifying results for Read-Linear(uncached) : ";
    //        memset(verificationOutput, 0, length * vectorSize * sizeof (cl_float));
    //
    //        // Verify result for Linear access
    //        for (int i = 0; i < (int) length; i++) {
    //            int readPos = i;
    //            for (int j = 0; j < NUM_READS; j++) {
    //                readPos += OFFSET;
    //                for (int k = 0; k < vecElements; k++) {
    //                    verificationOutput[i * vectorSize + k] += input[readPos * vectorSize + k];
    //                }
    //            }
    //        }
    //
    //        // Map cl_mem outputKadd to host for reading
    //        status = mapBuffer(outputKadd, outputReadLU, (length * sizeElement), CL_MAP_READ);
    //        CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(outputKadd)");
    //
    //        passStatus = 0;
    //        float * devBuffer = (float *) outputReadLU;
    //        float * refBuffer = (float *) verificationOutput;
    //        for (int i = 0; i < (int) (length * vectorSize); i += vectorSize) {
    //            for (int j = 0; j < vecElements; j++) {
    //
    //                float fErr = devBuffer[j] - refBuffer[j];
    //                if (fErr < (float) 0)
    //                    fErr = -fErr;
    //
    //                if (fErr > (float) 1e-5) {
    //                    passStatus = 1;
    //                    break;
    //                }
    //            }
    //            if (passStatus != 0)
    //                break;
    //
    //            devBuffer += vectorSize;
    //            refBuffer += vectorSize;
    //        }
    //
    //        status = unmapBuffer(outputKadd, outputReadLU);
    //        CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(outputKadd)");
    //
    //        if (passStatus == 0) {
    //            std::cout << "Passed!\n" << std::endl;
    //        } else {
    //            std::cout << "Failed!\n" << std::endl;
    //            return SDK_FAILURE;
    //        }
    //
    //    }

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
    strArray[0] = "Size (Bytes)";
    stats[0] = toString(sizeInBytesPerIter, std::dec);
    strArray[1] = "Avg. Kernel Time (sec)";
    stats[1] = toString(KaddTime, std::dec);
    strArray[2] = "Avg Bandwidth (GBPS)";
    stats[2] = toString(KaddGbps, std::dec);
    printStatistics(strArray, stats, 3);
}

int
ComputeBench::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseMemObject(inputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer)");

    status = clReleaseMemObject(outputKadd);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputKadd)");

    if (constValue) {
        status = clReleaseMemObject(constValue);
        CHECK_OPENCL_ERROR(status, "clReleaseMemOnject failed.");
    }

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

    FREE(verificationOutput);
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
