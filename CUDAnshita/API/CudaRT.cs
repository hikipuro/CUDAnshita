using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CUDAnshita {
	using cudaError_t = cudaError;
	using cudaEvent_t = IntPtr;
	using cudaIpcEventHandle_t = cudaIpcEventHandle;
	using cudaIpcMemHandle_t = cudaIpcMemHandle;
	using cudaStream_t = IntPtr;
	using cudaStreamCallback_t = IntPtr;
	using cudaArray_t = IntPtr;
	using cudaArray_const_t = IntPtr;
	using cudaMipmappedArray_t = IntPtr;
	using cudaMipmappedArray_const_t = IntPtr;
	using cudaTextureObject_t = IntPtr;
	using cudaSurfaceObject_t = IntPtr;
	using size_t = Int64;

	/// <summary>
	/// NVIDIA CUDA Runtime API
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/">http://docs.nvidia.com/cuda/cuda-runtime-api/</a>
	/// </remarks>
	public class CudaRT {
		public class API {
			const string DLL_PATH = "cudart64_80.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			// ----- Device Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaChooseDevice(ref int device, ref cudaDeviceProp prop);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetAttribute(ref int value, cudaDeviceAttr attr, int device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetByPCIBusId(ref int device, string pciBusId);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetCacheConfig(ref cudaFuncCache pCacheConfig);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetLimit(ref size_t pValue, cudaLimit limit);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetP2PAttribute(ref int value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetPCIBusId(StringBuilder pciBusId, int len, int device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetSharedMemConfig(ref cudaSharedMemConfig pConfig);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetStreamPriorityRange(ref int leastPriority, ref int greatestPriority);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceReset();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSynchronize();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDevice(ref int device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDeviceCount(ref int count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDeviceFlags(ref uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDeviceProperties(ref cudaDeviceProp prop, int device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcCloseMemHandle(IntPtr devPtr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcGetEventHandle(ref cudaIpcEventHandle_t handle, cudaEvent_t cudaEvent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcGetMemHandle(ref cudaIpcMemHandle_t handle, IntPtr devPtr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcOpenEventHandle(ref cudaEvent_t cudaEvent, cudaIpcEventHandle_t handle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcOpenMemHandle(ref IntPtr devPtr, cudaIpcMemHandle_t handle, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDevice(int device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDeviceFlags(uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetValidDevices(int[] device_arr, int len);

			// ----- Thread Management [DEPRECATED]

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadExit();

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadGetCacheConfig(ref cudaFuncCache pCacheConfig);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadGetLimit(ref size_t pValue, cudaLimit limit);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadSynchronize();

			// ----- Error Handling

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr cudaGetErrorName(cudaError_t error);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr cudaGetErrorString(cudaError_t error);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetLastError();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaPeekAtLastError();

			// ----- Stream Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, IntPtr userData, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, IntPtr devPtr, size_t length = 0, uint flags = Defines.cudaMemAttachSingle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamCreate(ref cudaStream_t pStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamCreateWithFlags(ref cudaStream_t pStream, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamCreateWithPriority(ref cudaStream_t pStream, uint flags, int priority);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamDestroy(cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, ref uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, ref int priority);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamQuery(cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamSynchronize(cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t cudaEvent, uint flags);

			// ----- Event Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventCreate(ref cudaEvent_t cudaEvent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventCreateWithFlags(ref cudaEvent_t cudaEvent, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventDestroy(cudaEvent_t cudaEvent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventElapsedTime(ref float ms, cudaEvent_t start, cudaEvent_t end);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventQuery(cudaEvent_t cudaEvent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventRecord(cudaEvent_t cudaEvent, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventSynchronize(cudaEvent_t cudaEvent);

			// ----- Execution Control

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFuncGetAttributes(ref cudaFuncAttributes attr, IntPtr func);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFuncSetCacheConfig(IntPtr func, cudaFuncCache cacheConfig);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFuncSetSharedMemConfig(IntPtr func, cudaSharedMemConfig config);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaLaunchKernel(IntPtr func, dim3 gridDim, dim3 blockDim, ref IntPtr args, size_t sharedMem, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDoubleForDevice(ref double d);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDoubleForHost(ref double d);

			// ----- Occupancy

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(ref int numBlocks, IntPtr func, int blockSize, size_t dynamicSMemSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref int numBlocks, IntPtr func, int blockSize, size_t dynamicSMemSize, uint flags);

			// ----- Execution Control [DEPRECATED]

			[Obsolete("This function is deprecated as of CUDA 7.0")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);

			[Obsolete("This function is deprecated as of CUDA 7.0")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaLaunch(IntPtr func);

			[Obsolete("This function is deprecated as of CUDA 7.0")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetupArgument(IntPtr arg, size_t size, size_t offset);

			// ----- Memory Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaArrayGetInfo(ref cudaChannelFormatDesc desc, ref cudaExtent extent, ref uint flags, cudaArray_t array);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFree(IntPtr devPtr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFreeArray(cudaArray_t array);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFreeHost(IntPtr ptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetMipmappedArrayLevel(ref cudaArray_t levelArray, cudaMipmappedArray_const_t mipmappedArray, uint level);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSymbolAddress(ref IntPtr devPtr, IntPtr symbol);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSymbolSize(ref size_t size, IntPtr symbol);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostAlloc(ref IntPtr pHost, size_t size, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostGetDevicePointer(ref IntPtr pDevice, IntPtr pHost, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostGetFlags(ref uint pFlags, IntPtr pHost);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostRegister(IntPtr ptr, size_t size, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostUnregister(IntPtr ptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMalloc(ref IntPtr devPtr, size_t size);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMalloc3D(ref cudaPitchedPtr pitchedDevPtr, cudaExtent extent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMalloc3DArray(ref cudaArray_t array, ref cudaChannelFormatDesc desc, cudaExtent extent, uint flags = 0);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocArray(ref cudaArray_t array, ref cudaChannelFormatDesc desc, size_t width, size_t height = 0, uint flags = 0);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocHost(ref IntPtr ptr, size_t size);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocManaged(ref IntPtr devPtr, size_t size, uint flags = Defines.cudaMemAttachGlobal);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocMipmappedArray(ref cudaMipmappedArray_t mipmappedArray, ref cudaChannelFormatDesc desc, cudaExtent extent, uint numLevels, uint flags = 0);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocPitch(ref IntPtr devPtr, ref size_t pitch, size_t width, size_t height);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemAdvise(IntPtr devPtr, size_t count, cudaMemoryAdvise advice, int device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemGetInfo(ref size_t free, ref size_t total);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemPrefetchAsync(IntPtr devPtr, size_t count, int dstDevice, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemRangeGetAttribute(IntPtr data, size_t dataSize, cudaMemRangeAttribute attribute, IntPtr devPtr, size_t count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemRangeGetAttributes(ref IntPtr data, ref size_t dataSizes, ref cudaMemRangeAttribute[] attributes, size_t numAttributes, IntPtr devPtr, size_t count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2D(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DAsync(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DFromArray(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DFromArrayAsync(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3D(ref cudaMemcpy3DParms p);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3DAsync(ref cudaMemcpy3DParms p, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3DPeer(ref cudaMemcpy3DPeerParms p);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3DPeerAsync(ref cudaMemcpy3DPeerParms p, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyAsync(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromArray(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromArrayAsync(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromSymbol(IntPtr dst, IntPtr symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToHost);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromSymbolAsync(IntPtr dst, IntPtr symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToSymbol(IntPtr symbol, IntPtr src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyHostToDevice);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToSymbolAsync(IntPtr symbol, IntPtr src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset(IntPtr devPtr, int value, size_t count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset2D(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset2DAsync(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemsetAsync(IntPtr devPtr, int value, size_t count, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaPitchedPtr make_cudaPitchedPtr(IntPtr d, size_t p, size_t xsz, size_t ysz);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaPos make_cudaPos(size_t x, size_t y, size_t z);

			// ----- Unified Addressing

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaPointerGetAttributes(ref cudaPointerAttributes attributes, IntPtr ptr);

			// ----- Peer Device Memory Access

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceCanAccessPeer(ref int canAccessPeer, int device, int peerDevice);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, uint flags);

			// ----- OpenGL Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLGetDevices(ref uint pCudaDeviceCount, ref int pCudaDevices, uint cudaDeviceCount, cudaGLDeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsGLRegisterBuffer(ref cudaGraphicsResource[] resource, GLuint buffer, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsGLRegisterImage(ref cudaGraphicsResource[] resource, GLuint image, GLenum target, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaWGLGetDevice(ref int device, HGPUNV hGpu);

			// ----- OpenGL Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLMapBufferObject(ref IntPtr devPtr, GLuint bufObj);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLMapBufferObjectAsync(ref IntPtr devPtr, GLuint bufObj, cudaStream_t stream);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLRegisterBufferObject(GLuint bufObj);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj, uint flags);

			//[Obsolete]
			//​[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLSetGLDevice(int device);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLUnmapBufferObject(GLuint bufObj);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj, cudaStream_t stream);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj);

			// ----- Direct3D 9 Interoperability

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaD3D9GetDevice(ref int device, string pszAdapterName);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9GetDevices(ref uint pCudaDeviceCount, ref int pCudaDevices, uint cudaDeviceCount, IDirect3DDevice9* pD3D9Device, cudaD3D9DeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9GetDirect3DDevice(ref IDirect3DDevice9[] ppD3D9Device);

			//​[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9SetDirect3DDevice(ref IDirect3DDevice9 pD3D9Device, int device = -1);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsD3D9RegisterResource(ref cudaGraphicsResource[] resource, ref IDirect3DResource9 pD3DResource, uint flags);

			// ----- Direct3D 9 Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9MapResources(int count, ref IDirect3DResource9[] ppResources);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9RegisterResource(ref IDirect3DResource9 pResource, uint flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedArray(ref cudaArray[] ppArray, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedPitch(ref size_t pPitch, ref size_t pPitchSlice, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedPointer(ref IntPtr pPointer, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedSize(ref size_t pSize, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetSurfaceDimensions(ref size_t pWidth, ref size_t pHeight, ref size_t pDepth, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceSetMapFlags(ref IDirect3DResource9 pResource, uint flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9UnmapResources(int count, ref IDirect3DResource9[] ppResources);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9UnregisterResource(ref IDirect3DResource9 pResource);

			// ----- Direct3D 10 Interoperability

			//​[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10GetDevice(ref int device, ref IDXGIAdapter pAdapter);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10GetDevices(ref uint pCudaDeviceCount, ref int pCudaDevices, uint cudaDeviceCount, ref ID3D10Device pD3D10Device, cudaD3D10DeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsD3D10RegisterResource(ref cudaGraphicsResource[] resource, ref ID3D10Resource pD3DResource, uint flags);

			// ----- Direct3D 10 Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10GetDirect3DDevice(ref ID3D10Device[] ppD3D10Device);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10MapResources(int count, ref ID3D10Resource[] ppResources);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10RegisterResource(ref ID3D10Resource pResource, uint flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedArray(ref cudaArray[] ppArray, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedPitch(ref size_t pPitch, ref size_t pPitchSlice, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedPointer(ref IntPtr pPointer, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedSize(ref size_t pSize, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetSurfaceDimensions(ref size_t pWidth, ref size_t pHeight, ref size_t pDepth, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceSetMapFlags(ref ID3D10Resource pResource, uint flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10SetDirect3DDevice(ref ID3D10Device pD3D10Device, int device = -1);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10UnmapResources(int count, ref ID3D10Resource[] ppResources);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10UnregisterResource(ref ID3D10Resource pResource);

			// ----- Direct3D 11 Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D11GetDevice(ref int device, ref IDXGIAdapter pAdapter);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D11GetDevices(ref uint pCudaDeviceCount, ref int pCudaDevices, uint cudaDeviceCount, ref ID3D11Device pD3D11Device, cudaD3D11DeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsD3D11RegisterResource(ref cudaGraphicsResource[] resource, ref ID3D11Resource pD3DResource, uint flags);

			// ----- Direct3D 11 Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D11GetDirect3DDevice(ref ID3D11Device[] ppD3D11Device);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D11SetDirect3DDevice(ref ID3D11Device pD3D11Device, int device = -1);

			// ----- VDPAU Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsVDPAURegisterOutputSurface(ref cudaGraphicsResource[] resource, VdpOutputSurface vdpSurface, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsVDPAURegisterVideoSurface(ref cudaGraphicsResource[] resource, VdpVideoSurface vdpSurface, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaVDPAUGetDevice(ref int device, VdpDevice vdpDevice, ref VdpGetProcAddress vdpGetProcAddress);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, ref VdpGetProcAddress vdpGetProcAddress);

			// ----- EGL Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamConsumerAcquireFrame(ref cudaEglStreamConnection conn, ref cudaGraphicsResource_t pCudaResource, ref cudaStream_t pStream, uint timeout);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamConsumerConnect(ref cudaEglStreamConnection conn, EGLStreamKHR eglStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamConsumerConnectWithFlags(ref cudaEglStreamConnection conn, EGLStreamKHR eglStream, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamConsumerReleaseFrame(ref cudaEglStreamConnection conn, cudaGraphicsResource_t pCudaResource, ref cudaStream_t pStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamProducerConnect(ref cudaEglStreamConnection conn, EGLStreamKHR eglStream, EGLint width, EGLint height);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamProducerDisconnect(ref cudaEglStreamConnection conn);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamProducerPresentFrame(ref cudaEglStreamConnection conn, cudaEglFrame eglframe, ref cudaStream_t pStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaEGLStreamProducerReturnFrame(ref cudaEglStreamConnection conn, ref cudaEglFrame eglframe, ref cudaStream_t pStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsEGLRegisterImage(ref cudaGraphicsResource[] pCudaResource, EGLImageKHR image, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsResourceGetMappedEglFrame(ref cudaEglFrame eglFrame, cudaGraphicsResource_t resource, uint index, uint mipLevel);

			// ----- Graphics Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsMapResources(int count, ref cudaGraphicsResource_t resources, cudaStream_t stream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(ref cudaMipmappedArray_t mipmappedArray, cudaGraphicsResource_t resource);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsResourceGetMappedPointer(ref IntPtr devPtr, ref size_t size, cudaGraphicsResource_t resource);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsSubResourceGetMappedArray(ref cudaArray_t array, cudaGraphicsResource_t resource, uint arrayIndex, uint mipLevel);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsUnmapResources(int count, ref cudaGraphicsResource_t resources, cudaStream_t stream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);

			// ----- Texture Reference Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTexture(ref size_t offset, ref textureReference texref, IntPtr devPtr, ref cudaChannelFormatDesc desc, size_t size);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTexture2D(ref size_t offset, ref textureReference texref, IntPtr devPtr, ref cudaChannelFormatDesc desc, size_t width, size_t height, size_t pitch);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTextureToArray(ref textureReference texref, cudaArray_const_t array, ref cudaChannelFormatDesc desc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTextureToMipmappedArray(ref textureReference texref, cudaMipmappedArray_const_t mipmappedArray, ref cudaChannelFormatDesc desc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetChannelDesc(ref cudaChannelFormatDesc desc, cudaArray_const_t array);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureAlignmentOffset(ref size_t offset, ref textureReference texref);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureReference(ref textureReference texref, IntPtr symbol);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaUnbindTexture(textureReference texref);

			// ----- Surface Reference Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindSurfaceToArray(ref surfaceReference surfref, cudaArray_const_t array, ref cudaChannelFormatDesc desc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSurfaceReference(ref surfaceReference surfref, IntPtr symbol);

			// ----- Texture Object Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaCreateTextureObject(ref cudaTextureObject_t pTexObject, ref cudaResourceDesc pResDesc, ref cudaTextureDesc pTexDesc, ref cudaResourceViewDesc pResViewDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureObjectResourceDesc(ref cudaResourceDesc pResDesc, cudaTextureObject_t texObject);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureObjectResourceViewDesc(ref cudaResourceViewDesc pResViewDesc, cudaTextureObject_t texObject);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureObjectTextureDesc(ref cudaTextureDesc pTexDesc, cudaTextureObject_t texObject);

			// ----- Surface Object Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaCreateSurfaceObject(ref cudaSurfaceObject_t pSurfObject, ref cudaResourceDesc pResDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSurfaceObjectResourceDesc(ref cudaResourceDesc pResDesc, cudaSurfaceObject_t surfObject);

			// ----- Version Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDriverGetVersion(ref int driverVersion);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaRuntimeGetVersion(ref int runtimeVersion);

			// ----- Profiler Control

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaProfilerStart();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaProfilerStop();
		}

		public static int ChooseDevice(cudaDeviceProp prop) {
			int device = 0;
			CheckStatus(API.cudaChooseDevice(ref device, ref prop));
			return device;
		}

		public static int DeviceGetAttribute(cudaDeviceAttr attr, int device) {
			int value = 0;
			CheckStatus(API.cudaDeviceGetAttribute(ref value, attr, device));
			return value;
		}

		public static int DeviceGetByPCIBusId(string pciBusId) {
			int device = 0;
			CheckStatus(API.cudaDeviceGetByPCIBusId(ref device, pciBusId));
			return device;
		}

		public static cudaFuncCache DeviceGetCacheConfig() {
			cudaFuncCache pCacheConfig = cudaFuncCache.cudaFuncCachePreferEqual;
			CheckStatus(API.cudaDeviceGetCacheConfig(ref pCacheConfig));
			return pCacheConfig;
		}

		public static size_t DeviceGetLimit(cudaLimit limit) {
			size_t pValue = 0;
			CheckStatus(API.cudaDeviceGetLimit(ref pValue, limit));
			return pValue;
		}

		public static int DeviceGetP2PAttribute(cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
			int value = 0;
			CheckStatus(API.cudaDeviceGetP2PAttribute(ref value, attr, srcDevice, dstDevice));
			return value;
		}

		public static string DeviceGetPCIBusId(int device) {
			StringBuilder pciBusId = new StringBuilder(32);
			CheckStatus(API.cudaDeviceGetPCIBusId(pciBusId, 32, device));
			return pciBusId.ToString();
		}

		public static cudaSharedMemConfig DeviceGetSharedMemConfig() {
			cudaSharedMemConfig pConfig = new cudaSharedMemConfig();
			CheckStatus(API.cudaDeviceGetSharedMemConfig(ref pConfig));
			return pConfig;
		}

		public static int[] DeviceGetStreamPriorityRange() {
			int leastPriority = 0;
			int greatestPriority = 0;
			CheckStatus(API.cudaDeviceGetStreamPriorityRange(ref leastPriority, ref greatestPriority));
			return new int[] { leastPriority, greatestPriority };
		}

		public static void DeviceReset() {
			CheckStatus(API.cudaDeviceReset());
		}

		public static void DeviceSetCacheConfig(cudaFuncCache cacheConfig) {
			CheckStatus(API.cudaDeviceSetCacheConfig(cacheConfig));
		}

		public static void DeviceSetLimit(cudaLimit limit, size_t value) {
			CheckStatus(API.cudaDeviceSetLimit(limit, value));
		}

		public static void DeviceSetSharedMemConfig(cudaSharedMemConfig config) {
			CheckStatus(API.cudaDeviceSetSharedMemConfig(config));
		}

		public static void DeviceSynchronize() {
			CheckStatus(API.cudaDeviceSynchronize());
		}

		public static int GetDevice() {
			int device = 0;
			CheckStatus(API.cudaGetDevice(ref device));
			return device;
		}

		public static int GetDeviceCount() {
			int count = 0;
			CheckStatus(API.cudaGetDeviceCount(ref count));
			return count;
		}

		public static uint GetDeviceFlags() {
			uint flags = 0;
			CheckStatus(API.cudaGetDeviceFlags(ref flags));
			return flags;
		}

		public static cudaDeviceProp GetDeviceProperties(int device) {
			cudaDeviceProp prop = new cudaDeviceProp();
			CheckStatus(API.cudaGetDeviceProperties(ref prop, device));
			return prop;
		}

		public static void IpcCloseMemHandle(IntPtr devPtr) {
			CheckStatus(API.cudaIpcCloseMemHandle(devPtr));
		}

		public static cudaIpcEventHandle_t IpcGetEventHandle(cudaEvent_t cudaEvent) {
			cudaIpcEventHandle_t handle = new cudaIpcEventHandle_t();
			CheckStatus(API.cudaIpcGetEventHandle(ref handle, cudaEvent));
			return handle;
		}

		public static cudaIpcMemHandle_t IpcGetMemHandle(IntPtr devPtr) {
			cudaIpcMemHandle_t handle = new cudaIpcMemHandle_t();
			CheckStatus(API.cudaIpcGetMemHandle(ref handle, devPtr));
			return handle;
		}

		public static cudaEvent_t IpcOpenEventHandle(cudaIpcEventHandle_t handle) {
			cudaEvent_t cudaEvent = IntPtr.Zero;
			CheckStatus(API.cudaIpcOpenEventHandle(ref cudaEvent, handle));
			return cudaEvent;
		}

		public static IntPtr IpcOpenMemHandle(cudaIpcMemHandle_t handle, uint flags) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaIpcOpenMemHandle(ref devPtr, handle, flags));
			return devPtr;
		}

		public static void SetDevice(int device) {
			CheckStatus(API.cudaSetDevice(device));
		}

		public static void SetDeviceFlags(uint flags) {
			CheckStatus(API.cudaSetDeviceFlags(flags));
		}

		public static void SetValidDevices(int[] devices) {
			CheckStatus(API.cudaSetValidDevices(devices, devices.Length));
		}

		[Obsolete]
		public static void ThreadExit() {
			CheckStatus(API.cudaThreadExit());
		}

		[Obsolete]
		public static cudaFuncCache ThreadGetCacheConfig() {
			cudaFuncCache pCacheConfig = new cudaFuncCache();
			CheckStatus(API.cudaThreadGetCacheConfig(ref pCacheConfig));
			return pCacheConfig;
		}

		[Obsolete]
		public static size_t ThreadGetLimit(cudaLimit limit) {
			size_t pValue = 0;
			CheckStatus(API.cudaThreadGetLimit(ref pValue, limit));
			return pValue;
		}

		[Obsolete]
		public static void ThreadSetCacheConfig(cudaFuncCache cacheConfig) {
			CheckStatus(API.cudaThreadSetCacheConfig(cacheConfig));
		}

		[Obsolete]
		public static void ThreadSetLimit(cudaLimit limit, size_t value) {
			CheckStatus(API.cudaThreadSetLimit(limit, value));
		}

		[Obsolete]
		public static void ThreadSynchronize() {
			CheckStatus(API.cudaThreadSynchronize());
		}

		public static string GetErrorName(cudaError_t error) {
			IntPtr ptr = API.cudaGetErrorName(error);
			return Marshal.PtrToStringAnsi(ptr);
		}

		public static string GetErrorString(cudaError_t error) {
			IntPtr ptr = API.cudaGetErrorString(error);
			return Marshal.PtrToStringAnsi(ptr);
		}

		public static cudaError_t GetLastError() {
			return API.cudaGetLastError();
		}

		public static cudaError_t PeekAtLastError() {
			return API.cudaPeekAtLastError();
		}

		public static void StreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, IntPtr userData, uint flags) {
			CheckStatus(API.cudaStreamAddCallback(stream, callback, userData, flags));
		}

		public static void StreamAttachMemAsync(cudaStream_t stream, IntPtr devPtr, size_t length = 0, uint flags = Defines.cudaMemAttachSingle) {
			CheckStatus(API.cudaStreamAttachMemAsync(stream, devPtr, length, flags));
		}

		public static cudaStream_t StreamCreate() {
			cudaStream_t pStream = IntPtr.Zero;
			CheckStatus(API.cudaStreamCreate(ref pStream));
			return pStream;
		}

		public static cudaStream_t StreamCreateWithFlags(uint flags) {
			cudaStream_t pStream = IntPtr.Zero;
			CheckStatus(API.cudaStreamCreateWithFlags(ref pStream, flags));
			return pStream;
		}

		public static cudaStream_t StreamCreateWithPriority(uint flags, int priority) {
			cudaStream_t pStream = IntPtr.Zero;
			CheckStatus(API.cudaStreamCreateWithPriority(ref pStream, flags, priority));
			return pStream;
		}

		public static void StreamDestroy(cudaStream_t stream) {
			CheckStatus(API.cudaStreamDestroy(stream));
		}

		public static uint StreamGetFlags(cudaStream_t hStream) {
			uint flags = 0;
			CheckStatus(API.cudaStreamGetFlags(hStream, ref flags));
			return flags;
		}

		public static int StreamGetPriority(cudaStream_t hStream) {
			int priority = 0;
			CheckStatus(API.cudaStreamGetPriority(hStream, ref priority));
			return priority;
		}

		public static cudaError StreamQuery(cudaStream_t stream) {
			return API.cudaStreamQuery(stream);
		}

		public static void StreamSynchronize(cudaStream_t stream) {
			CheckStatus(API.cudaStreamSynchronize(stream));
		}

		public static void StreamWaitEvent(cudaStream_t stream, cudaEvent_t cudaEvent, uint flags) {
			CheckStatus(API.cudaStreamWaitEvent(stream, cudaEvent, flags));
		}

		public static cudaEvent_t EventCreate() {
			cudaEvent_t cudaEvent = IntPtr.Zero;
			CheckStatus(API.cudaEventCreate(ref cudaEvent));
			return cudaEvent;
		}

		public static cudaEvent_t EventCreateWithFlags(uint flags) {
			cudaEvent_t cudaEvent = IntPtr.Zero;
			CheckStatus(API.cudaEventCreateWithFlags(ref cudaEvent, flags));
			return cudaEvent;
		}

		public static void EventDestroy(cudaEvent_t cudaEvent) {
			CheckStatus(API.cudaEventDestroy(cudaEvent));
		}

		public static float EventElapsedTime(cudaEvent_t start, cudaEvent_t end) {
			float ms = 0;
			CheckStatus(API.cudaEventElapsedTime(ref ms, start, end));
			return ms;
		}

		public static cudaError EventQuery(cudaEvent_t cudaEvent) {
			return API.cudaEventQuery(cudaEvent);
		}

		public static void EventRecord(cudaEvent_t cudaEvent, cudaStream_t stream) {
			CheckStatus(API.cudaEventRecord(cudaEvent, stream));
		}

		public static void EventSynchronize(cudaEvent_t cudaEvent) {
			CheckStatus(API.cudaEventSynchronize(cudaEvent));
		}

		public static cudaFuncAttributes FuncGetAttributes(IntPtr func) {
			cudaFuncAttributes attr = new cudaFuncAttributes();
			CheckStatus(API.cudaFuncGetAttributes(ref attr, func));
			return attr;
		}

		public static void FuncSetCacheConfig(IntPtr func, cudaFuncCache cacheConfig) {
			CheckStatus(API.cudaFuncSetCacheConfig(func, cacheConfig));
		}

		public static void FuncSetSharedMemConfig(IntPtr func, cudaSharedMemConfig config) {
			CheckStatus(API.cudaFuncSetSharedMemConfig(func, config));
		}

		public static void LaunchKernel(IntPtr func, dim3 gridDim, dim3 blockDim, IntPtr args, size_t sharedMem, cudaStream_t stream) {
			CheckStatus(API.cudaLaunchKernel(func, gridDim, blockDim, ref args, sharedMem, stream));
		}

		[Obsolete("This function is deprecated as of CUDA 7.5")]
		public static void SetDoubleForDevice(double d) {
			CheckStatus(API.cudaSetDoubleForDevice(ref d));
		}

		[Obsolete("This function is deprecated as of CUDA 7.5")]
		public static void SetDoubleForHost(double d) {
			CheckStatus(API.cudaSetDoubleForHost(ref d));
		}

		public static int OccupancyMaxActiveBlocksPerMultiprocessor(IntPtr func, int blockSize, size_t dynamicSMemSize) {
			int numBlocks = 0;
			CheckStatus(API.cudaOccupancyMaxActiveBlocksPerMultiprocessor(ref numBlocks, func, blockSize, dynamicSMemSize));
			return numBlocks;
		}

		public static int OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(IntPtr func, int blockSize, size_t dynamicSMemSize, uint flags) {
			int numBlocks = 0;
			CheckStatus(API.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref numBlocks, func, blockSize, dynamicSMemSize, flags));
			return numBlocks;
		}

		[Obsolete("This function is deprecated as of CUDA 7.0")]
		public static void ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
			CheckStatus(API.cudaConfigureCall(gridDim, blockDim, sharedMem, stream));
		}

		[Obsolete("This function is deprecated as of CUDA 7.0")]
		public static void Launch(IntPtr func) {
			CheckStatus(API.cudaLaunch(func));
		}

		[Obsolete("This function is deprecated as of CUDA 7.0")]
		public static void SetupArgument(IntPtr arg, size_t size, size_t offset) {
			CheckStatus(API.cudaSetupArgument(arg, size, offset));
		}

		public static void ArrayGetInfo(cudaArray_t array) {
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			cudaExtent extent = new cudaExtent();
			uint flags = 0;
			CheckStatus(API.cudaArrayGetInfo(ref desc, ref extent, ref flags, array));
		}

		public static void Free(IntPtr devPtr) {
			CheckStatus(API.cudaFree(devPtr));
		}

		public static void FreeArray(cudaArray_t array) {
			CheckStatus(API.cudaFreeArray(array));
		}

		public static void FreeHost(IntPtr ptr) {
			CheckStatus(API.cudaFreeHost(ptr));
		}

		public static void FreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
			CheckStatus(API.cudaFreeMipmappedArray(mipmappedArray));
		}

		public static cudaArray_t GetMipmappedArrayLevel(cudaMipmappedArray_const_t mipmappedArray, uint level) {
			cudaArray_t levelArray = IntPtr.Zero;
			CheckStatus(API.cudaGetMipmappedArrayLevel(ref levelArray, mipmappedArray, level));
			return levelArray;
		}

		public static IntPtr GetSymbolAddress(IntPtr symbol) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaGetSymbolAddress(ref devPtr, symbol));
			return devPtr;
		}

		public static size_t GetSymbolSize(IntPtr symbol) {
			size_t size = 0;
			CheckStatus(API.cudaGetSymbolSize(ref size, symbol));
			return size;
		}

		public static IntPtr HostAlloc(size_t size, uint flags) {
			IntPtr pHost = IntPtr.Zero;
			CheckStatus(API.cudaHostAlloc(ref pHost, size, flags));
			return pHost;
		}

		public static IntPtr HostGetDevicePointer(IntPtr pHost, uint flags) {
			IntPtr pDevice = IntPtr.Zero;
			CheckStatus(API.cudaHostGetDevicePointer(ref pDevice, pHost, flags));
			return pHost;
		}

		public static uint HostGetFlags(IntPtr pHost) {
			uint pFlags = 0;
			CheckStatus(API.cudaHostGetFlags(ref pFlags, pHost));
			return pFlags;
		}

		public static void HostRegister(IntPtr ptr, size_t size, uint flags) {
			CheckStatus(API.cudaHostRegister(ptr, size, flags));
		}

		public static void HostUnregister(IntPtr ptr) {
			CheckStatus(API.cudaHostUnregister(ptr));
		}

		public static IntPtr Malloc(size_t size) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaMalloc(ref devPtr, size));
			return devPtr;
		}

		public static cudaPitchedPtr Malloc3D(cudaExtent extent) {
			cudaPitchedPtr pitchedDevPtr = new cudaPitchedPtr();
			CheckStatus(API.cudaMalloc3D(ref pitchedDevPtr, extent));
			return pitchedDevPtr;
		}

		public static cudaArray_t Malloc3DArray(cudaExtent extent, uint flags = 0) {
			cudaArray_t array = IntPtr.Zero;
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaMalloc3DArray(ref array, ref desc, extent, flags));
			return array;
		}

		public static cudaArray_t MallocArray(size_t width, size_t height = 0, uint flags = 0) {
			cudaArray_t array = IntPtr.Zero;
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaMallocArray(ref array, ref desc, width, height, flags));
			return array;
		}

		public static IntPtr MallocHost(size_t size) {
			IntPtr ptr = IntPtr.Zero;
			CheckStatus(API.cudaMallocHost(ref ptr, size));
			return ptr;
		}

		public static IntPtr MallocManaged(size_t size, uint flags = Defines.cudaMemAttachGlobal) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaMallocManaged(ref devPtr, size, flags));
			return devPtr;
		}

		public static cudaMipmappedArray_t MallocMipmappedArray(cudaExtent extent, uint numLevels, uint flags = 0) {
			cudaMipmappedArray_t mipmappedArray = new cudaMipmappedArray_t();
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaMallocMipmappedArray(ref mipmappedArray, ref desc, extent, numLevels, flags));
			return mipmappedArray;
		}

		public static IntPtr MallocPitch(size_t width, size_t height) {
			IntPtr devPtr = IntPtr.Zero;
			size_t pitch = 0;
			CheckStatus(API.cudaMallocPitch(ref devPtr, ref pitch, width, height));
			return devPtr;
		}

		public static void MemAdvise(IntPtr devPtr, size_t count, cudaMemoryAdvise advice, int device) {
			CheckStatus(API.cudaMemAdvise(devPtr, count, advice, device));
		}

		public static size_t[] MemGetInfo() {
			size_t free = 0;
			size_t total = 0;
			CheckStatus(API.cudaMemGetInfo(ref free, ref total));
			return new size_t[] { free, total };
		}

		public static void MemPrefetchAsync(IntPtr devPtr, size_t count, int dstDevice, cudaStream_t stream) {
			CheckStatus(API.cudaMemPrefetchAsync(devPtr, count, dstDevice, stream));
		}

		public static void MemRangeGetAttribute(IntPtr data, size_t dataSize, cudaMemRangeAttribute attribute, IntPtr devPtr, size_t count) {
			CheckStatus(API.cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count));
		}

		//public static void MemRangeGetAttributes(IntPtr data, size_t dataSize, cudaMemRangeAttribute attribute, IntPtr devPtr, size_t count) {
		//	CheckStatus(API.cudaMemRangeGetAttributes(data, dataSize, attribute, devPtr, count));
		//}

		public static void Memcpy(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy(dst, src, count, kind));
		}

		public static void Memcpy2D(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
		}

		public static void Memcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice) {
			CheckStatus(API.cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind));
		}

		public static void Memcpy2DAsync(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream));
		}

		public static void Memcpy2DFromArray(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind));
		}

		public static void Memcpy2DFromArrayAsync(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream));
		}

		public static void Memcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind));
		}

		public static void Memcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream));
		}

		public static void Memcpy3D(cudaMemcpy3DParms p) {
			CheckStatus(API.cudaMemcpy3D(ref p));
		}

		public static void Memcpy3DAsync(cudaMemcpy3DParms p, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy3DAsync(ref p, stream));
		}

		public static void Memcpy3DPeer(cudaMemcpy3DPeerParms p) {
			CheckStatus(API.cudaMemcpy3DPeer(ref p));
		}

		public static void Memcpy3DPeerAsync(cudaMemcpy3DPeerParms p, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy3DPeerAsync(ref p, stream));
		}

		public static void MemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice) {
			CheckStatus(API.cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind));
		}

		public static void MemcpyAsync(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyAsync(dst, src, count, kind, stream));
		}

		public static void MemcpyFromArray(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind));
		}

		public static void MemcpyFromArrayAsync(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream));
		}

		public static void MemcpyFromSymbol(IntPtr dst, IntPtr symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToHost) {
			CheckStatus(API.cudaMemcpyFromSymbol(dst, symbol, count, offset, kind));
		}

		public static void MemcpyFromSymbolAsync(IntPtr dst, IntPtr symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream));
		}

		public static void MemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count) {
			CheckStatus(API.cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count));
		}

		public static void MemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream));
		}

		public static void MemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind));
		}

		public static void MemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream));
		}

		public static void MemcpyToSymbol(IntPtr symbol, IntPtr src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyHostToDevice) {
			CheckStatus(API.cudaMemcpyToSymbol(symbol, src, count, offset, kind));
		}

		public static void MemcpyToSymbolAsync(IntPtr symbol, IntPtr src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream));
		}

		public static void Memset(IntPtr devPtr, int value, size_t count) {
			CheckStatus(API.cudaMemset(devPtr, value, count));
		}

		public static void Memset2D(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height) {
			CheckStatus(API.cudaMemset2D(devPtr, pitch, value, width, height));
		}

		public static void Memset2DAsync(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
			CheckStatus(API.cudaMemset2DAsync(devPtr, pitch, value, width, height, stream));
		}

		public static void Memset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) {
			CheckStatus(API.cudaMemset3D(pitchedDevPtr, value, extent));
		}

		public static void Memset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) {
			CheckStatus(API.cudaMemset3DAsync(pitchedDevPtr, value, extent, stream));
		}

		public static void MemsetAsync(IntPtr devPtr, int value, size_t count, cudaStream_t stream) {
			CheckStatus(API.cudaMemsetAsync(devPtr, value, count, stream));
		}

		public static cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
			return API.make_cudaExtent(w, h, d);
		}

		public static cudaPitchedPtr make_cudaPitchedPtr(IntPtr d, size_t p, size_t xsz, size_t ysz) {
			return API.make_cudaPitchedPtr(d, p, xsz, ysz);
		}

		public static cudaPos make_cudaPos(size_t x, size_t y, size_t z) {
			return API.make_cudaPos(x, y, z);
		}

		public static cudaPointerAttributes PointerGetAttributes(IntPtr ptr) {
			cudaPointerAttributes attributes = new cudaPointerAttributes();
			CheckStatus(API.cudaPointerGetAttributes(ref attributes, ptr));
			return attributes;
		}

		public static int DeviceCanAccessPeer(int device, int peerDevice) {
			int canAccessPeer = 0;
			CheckStatus(API.cudaDeviceCanAccessPeer(ref canAccessPeer, device, peerDevice));
			return canAccessPeer;
		}

		public static void DeviceDisablePeerAccess(int peerDevice) {
			CheckStatus(API.cudaDeviceDisablePeerAccess(peerDevice));
		}

		public static void DeviceEnablePeerAccess(int peerDevice, uint flags) {
			CheckStatus(API.cudaDeviceEnablePeerAccess(peerDevice, flags));
		}

		public static void BindTexture(size_t offset, textureReference texref, IntPtr devPtr, cudaChannelFormatDesc desc, size_t size = uint.MaxValue) {
			CheckStatus(API.cudaBindTexture(ref offset, ref texref, devPtr, ref desc, size));
		}

		public static void BindTexture2D(size_t offset, textureReference texref, IntPtr devPtr, cudaChannelFormatDesc desc, size_t width, size_t height, size_t pitch) {
			CheckStatus(API.cudaBindTexture2D(ref offset, ref texref, devPtr, ref desc, width, height, pitch));
		}

		public static void BindTextureToArray(textureReference texref, cudaArray_const_t array, cudaChannelFormatDesc desc) {
			CheckStatus(API.cudaBindTextureToArray(ref texref, array, ref desc));
		}

		public static void BindTextureToMipmappedArray(textureReference texref, cudaMipmappedArray_const_t mipmappedArray, cudaChannelFormatDesc desc) {
			CheckStatus(API.cudaBindTextureToMipmappedArray(ref texref, mipmappedArray, ref desc));
		}

		public static cudaChannelFormatDesc CreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) {
			return API.cudaCreateChannelDesc(x, y, z, w, f);
		}

		public static cudaChannelFormatDesc GetChannelDesc(cudaArray_const_t array) {
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaGetChannelDesc(ref desc, array));
			return desc;
		}

		public static size_t GetTextureAlignmentOffset(textureReference texref) {
			size_t offset = 0;
			CheckStatus(API.cudaGetTextureAlignmentOffset(ref offset, ref texref));
			return offset;
		}

		public static textureReference GetTextureReference(IntPtr symbol) {
			textureReference texref = new textureReference();
			CheckStatus(API.cudaGetTextureReference(ref texref, symbol));
			return texref;
		}

		public static void UnbindTexture(textureReference texref) {
			CheckStatus(API.cudaUnbindTexture(texref));
		}

		public static void BindSurfaceToArray(surfaceReference surfref, cudaArray_const_t array, cudaChannelFormatDesc desc) {
			CheckStatus(API.cudaBindSurfaceToArray(ref surfref, array, ref desc));
		}

		public static surfaceReference GetSurfaceReference(IntPtr symbol) {
			surfaceReference surfref = new surfaceReference();
			CheckStatus(API.cudaGetSurfaceReference(ref surfref, symbol));
			return surfref;
		}

		public static cudaTextureObject_t CreateTextureObject(cudaResourceDesc pResDesc, cudaTextureDesc pTexDesc, cudaResourceViewDesc pResViewDesc) {
			cudaTextureObject_t pTexObject = IntPtr.Zero;
			CheckStatus(API.cudaCreateTextureObject(ref pTexObject, ref pResDesc, ref pTexDesc, ref pResViewDesc));
			return pTexObject;
		}

		public static void DestroyTextureObject(cudaTextureObject_t texObject) {
			CheckStatus(API.cudaDestroyTextureObject(texObject));
		}

		public static cudaResourceDesc GetTextureObjectResourceDesc(cudaTextureObject_t texObject) {
			cudaResourceDesc pResDesc = new cudaResourceDesc();
			CheckStatus(API.cudaGetTextureObjectResourceDesc(ref pResDesc, texObject));
			return pResDesc;
		}

		public static cudaResourceViewDesc GetTextureObjectResourceViewDesc(cudaTextureObject_t texObject) {
			cudaResourceViewDesc pResViewDesc = new cudaResourceViewDesc();
			CheckStatus(API.cudaGetTextureObjectResourceViewDesc(ref pResViewDesc, texObject));
			return pResViewDesc;
		}

		public static cudaTextureDesc GetTextureObjectTextureDesc(cudaTextureObject_t texObject) {
			cudaTextureDesc pTexDesc = new cudaTextureDesc();
			CheckStatus(API.cudaGetTextureObjectTextureDesc(ref pTexDesc, texObject));
			return pTexDesc;
		}

		public static cudaSurfaceObject_t CreateSurfaceObject(cudaResourceDesc pResDesc) {
			cudaSurfaceObject_t pSurfObject = IntPtr.Zero;
			CheckStatus(API.cudaCreateSurfaceObject(ref pSurfObject, ref pResDesc));
			return pSurfObject;
		}

		public static void DestroySurfaceObject(cudaSurfaceObject_t surfObject) {
			CheckStatus(API.cudaDestroySurfaceObject(surfObject));
		}

		public static cudaResourceDesc GetSurfaceObjectResourceDesc(cudaSurfaceObject_t surfObject) {
			cudaResourceDesc pResDesc = new cudaResourceDesc();
			CheckStatus(API.cudaGetSurfaceObjectResourceDesc(ref pResDesc, surfObject));
			return pResDesc;
		}

		public static int DriverGetVersion() {
			int driverVersion = 0;
			CheckStatus(API.cudaDriverGetVersion(ref driverVersion));
			return driverVersion;
		}

		public static int RuntimeGetVersion() {
			int runtimeVersion = 0;
			CheckStatus(API.cudaRuntimeGetVersion(ref runtimeVersion));
			return runtimeVersion;
		}

		public static void ProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode) {
			CheckStatus(API.cudaProfilerInitialize(configFile, outputFile, outputMode));
		}

		public static void ProfilerStart() {
			CheckStatus(API.cudaProfilerStart());
		}

		public static void ProfilerStop() {
			CheckStatus(API.cudaProfilerStop());
		}

		static void CheckStatus(cudaError status) {
			if (status != cudaError.cudaSuccess) {
				throw new CudaException(status.ToString());
			}
		}
	}
}
