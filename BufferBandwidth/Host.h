/**********************************************************************
Copyright �2014 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef _HOST_H_
#define _HOST_H_

#include <stddef.h>

#ifdef _WIN32
#define MEM_MULTICORE
#define MAXWORKERS 32
#else
#define MAXWORKERS 1
#endif

extern int nWorkers;

bool readVerifyMemCPU( void *, unsigned char, size_t );
bool readVerifyMemCPU_MT( void *, unsigned char, size_t );
void memset_MT( void *, unsigned char, size_t );
void memcpy_MT( void *, void *, size_t );
void writeMemCPU( void *, unsigned char, size_t );
bool readVerifyMemSSE( void *, unsigned char, size_t );
void writeMemSSE ( void *, unsigned char, size_t );

bool readmem2DPitch( void *, unsigned char, size_t, int );
void memset2DPitch( void *, unsigned char, size_t, size_t, size_t );

void runon( unsigned int );
void stridePagesCPU( void *, size_t );
void assessHostMemPerf( void *, void *, size_t );
void empty_MT();
bool myMasterFunc();
void waitForThreads();

void benchThreads();
void launchThreads();
void shutdownThreads();

#endif // _HOST_H_