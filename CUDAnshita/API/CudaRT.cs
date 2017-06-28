using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using size_t = Int64;

	/// <summary>
	/// NVIDIA CUDA Runtime API
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/">http://docs.nvidia.com/cuda/cuda-runtime-api/</a>
	/// </remarks>
	public class CudaRT {
		const string DLL_PATH = "cudart64_80.dll";
		const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
		const CharSet CHAR_SET = CharSet.Ansi;

		// ----- Device Management

		/*
		static extern cudaError cudaChooseDevice(ref int device, const cudaDeviceProp* prop );
		static extern cudaError cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
		cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId );
		cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache** pCacheConfig);
		cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit);
		cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
		cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device);
		​cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig** pConfig);
		cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
		cudaError_t cudaDeviceReset(void );
		​cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
		cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
		​cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
		cudaError_t cudaDeviceSynchronize(void );
		cudaError_t cudaGetDevice(int* device);
		*/

		/// <summary>
		/// Returns the number of compute-capable devices.
		/// </summary>
		/// <param name="count">Returns the number of devices with compute capability greater or equal to 2.0</param>
		/// <returns></returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaGetDeviceCount(ref int count);

		/*
		cudaError_t cudaGetDeviceFlags(unsigned int* flags);
		cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
		cudaError_t cudaIpcCloseMemHandle(void* devPtr);
		cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event );
		cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr );
		​cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle );
		cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags );
		​cudaError_t cudaSetDevice(int device);
		cudaError_t cudaSetDeviceFlags(unsigned int flags);
		cudaError_t cudaSetValidDevices(int* device_arr, int len);
		*/

		// ----- Thread Management [DEPRECATED]

		/*
		cudaError_t cudaThreadExit(void );
		cudaError_t cudaThreadGetCacheConfig(cudaFuncCache** pCacheConfig);
		cudaError_t cudaThreadGetLimit(size_t* pValue, cudaLimit limit);
		cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);
		cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value);
		cudaError_t cudaThreadSynchronize(void );
		*/

		// ----- Error Handling

		/*
		const char* cudaGetErrorName (cudaError_t error );
		const char* cudaGetErrorString (cudaError_t error );
		​cudaError_t cudaGetLastError(void );
		cudaError_t cudaPeekAtLastError(void );
		*/

		// ----- Stream Management

		/*
		cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
		cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length = 0, unsigned int flags = cudaMemAttachSingle);
		cudaError_t cudaStreamCreate(cudaStream_t* pStream);
		cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
		cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
		cudaError_t cudaStreamDestroy(cudaStream_t stream);
		cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags);
		cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority);
		cudaError_t cudaStreamQuery(cudaStream_t stream);
		cudaError_t cudaStreamSynchronize(cudaStream_t stream);
		cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags );
		*/

		// ----- Event Management

		/*
		​cudaError_t cudaEventCreate(cudaEvent_t* event );
		​cudaError_t cudaEventCreateWithFlags ( cudaEvent_t* event, unsigned int flags );
		cudaError_t cudaEventDestroy(cudaEvent_t event );
		cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end );
		cudaError_t cudaEventQuery(cudaEvent_t event );
		cudaError_t cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 );
		cudaError_t cudaEventSynchronize ( cudaEvent_t event );
		*/

		// ----- Execution Control

		/*
		cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func );
		cudaError_t cudaFuncSetCacheConfig( const void* func, cudaFuncCache cacheConfig );
		cudaError_t cudaFuncSetSharedMemConfig( const void* func, cudaSharedMemConfig config );
		​cudaError_t cudaLaunchKernel( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream );
		cudaError_t cudaSetDoubleForDevice(double* d);
		cudaError_t cudaSetDoubleForHost(double* d);
		*/

		// ----- Occupancy

		/*
		cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize );
		cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags );
		*/

		// ----- Execution Control [DEPRECATED]

		/*
		cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
		cudaError_t cudaLaunch( const void* func );
		cudaError_t cudaSetupArgument( const void* arg, size_t size, size_t offset );
		*/

		// ----- Memory Management

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaMalloc(ref IntPtr devPtr, size_t size);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaFree(IntPtr devPtr);

		// ----- Unified Addressing
		// ----- Peer Device Memory Access
		// ----- OpenGL Interoperability
		// ----- OpenGL Interoperability [DEPRECATED]
		// ----- Direct3D 9 Interoperability
		// ----- Direct3D 9 Interoperability [DEPRECATED]
		// ----- Direct3D 10 Interoperability
		// ----- Direct3D 10 Interoperability [DEPRECATED]
		// ----- Direct3D 11 Interoperability
		// ----- Direct3D 11 Interoperability [DEPRECATED]
		// ----- VDPAU Interoperability
		// ----- EGL Interoperability
		// ----- Graphics Interoperability
		// ----- Texture Reference Management
		// ----- Surface Reference Management
		// ----- Texture Object Management
		// ----- Surface Object Management

		// ----- Version Management

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaDriverGetVersion(ref int driverVersion);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaRuntimeGetVersion(ref int runtimeVersion);

		// ----- Profiler Control

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaProfilerStart();

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaProfilerStop();
	}
}
