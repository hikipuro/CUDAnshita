using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using CUdevice = Int32;
	using CUdeviceptr = IntPtr;
	using CUlinkState = IntPtr;

	using CUcontext = IntPtr;
	using CUmodule = IntPtr;
	using CUfunction = IntPtr;
	using CUarray = IntPtr;
	using CUmipmappedArray = IntPtr;
	using CUtexref = IntPtr;
	using CUsurfref = IntPtr;
	using CUevent = IntPtr;
	using CUstream = IntPtr;
	using CUgraphicsResource = IntPtr;
	using CUtexObject = UInt64;
	using CUsurfObject = UInt64;

	using size_t = Int64;
	//using CUresult = cudaError;

	/// <summary>
	/// NVIDIA CUDA Driver API
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cuda-driver-api/">http://docs.nvidia.com/cuda/cuda-driver-api/</a>
	/// </remarks>
	public class NvCuda {
		public class API {
			const string DLL_PATH = "nvcuda.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			// ----- Error Handling

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGetErrorName(CUresult error, ref IntPtr pStr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGetErrorString(CUresult error, ref IntPtr pStr);

			// ----- Initialization

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuInit(uint Flags);

			// ----- Version Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDriverGetVersion(ref int driverVersion);

			// ----- Device Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGet(ref CUdevice device, int ordinal);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetAttribute(ref int pi, CUdevice_attribute attrib, CUdevice dev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetCount(ref int count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetName(string name, int len, CUdevice dev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceTotalMem(ref size_t bytes, CUdevice dev);

			// ----- Device Management [DEPRECATED]

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceComputeCapability(ref int major, ref int minor, CUdevice dev);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetProperties(ref CUdevprop prop, CUdevice dev);

			// ----- Primary Context Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxGetState(CUdevice dev, ref uint flags, ref int active);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxRelease(CUdevice dev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxReset(CUdevice dev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxRetain(ref CUcontext pctx, CUdevice dev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, uint flags);

			// ----- Context Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxCreate(ref CUcontext pctx, uint flags, CUdevice dev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxDestroy(CUcontext ctx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetApiVersion(CUcontext ctx, ref uint version);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetCacheConfig(ref CUfunc_cache pconfig);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetCurrent(ref CUcontext pctx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetDevice(ref CUdevice device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetFlags(ref uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetLimit(ref size_t pvalue, CUlimit limit);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetSharedMemConfig(ref CUsharedconfig pConfig);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetStreamPriorityRange(ref int leastPriority, ref int greatestPriority);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxPopCurrent(ref CUcontext pctx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxPushCurrent(CUcontext ctx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetCacheConfig(CUfunc_cache config);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetCurrent(CUcontext ctx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetLimit(CUlimit limit, size_t value);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetSharedMemConfig(CUsharedconfig config);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSynchronize();

			// ----- Context Management [DEPRECATED]

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxAttach(ref CUcontext pctx, uint flags);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxDetach(CUcontext ctx);

			// ----- Module Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, IntPtr data, size_t size, string name, uint numOptions, ref CUjit_option options, IntPtr optionValues);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, string path, uint numOptions, ref CUjit_option options, IntPtr optionValues);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkComplete(CUlinkState state, IntPtr cubinOut, ref size_t sizeOut);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkCreate(uint numOptions, ref CUjit_option options, IntPtr optionValues, ref CUlinkState stateOut);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkDestroy(CUlinkState state);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetFunction(ref CUfunction hfunc, CUmodule hmod, string name);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetGlobal(ref CUdeviceptr dptr, ref size_t bytes, CUmodule hmod, string name);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetSurfRef(ref CUsurfref pSurfRef, CUmodule hmod, string name);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetTexRef(ref CUtexref pTexRef, CUmodule hmod, string name);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoad(ref CUmodule module, string fname);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoadData(ref CUmodule module, IntPtr image);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoadDataEx(ref CUmodule module, IntPtr image, uint numOptions, CUjit_option options, IntPtr optionValues);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoadFatBinary(ref CUmodule module, IntPtr fatCubin);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleUnload(CUmodule hmod);

			// ----- Memory Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArray3DCreate(ref CUarray pHandle, ref CUDA_ARRAY3D_DESCRIPTOR pAllocateArray);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArray3DGetDescriptor(ref CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor, CUarray hArray);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArrayCreate(ref CUarray pHandle, ref CUDA_ARRAY_DESCRIPTOR pAllocateArray);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArrayDestroy(CUarray hArray);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArrayGetDescriptor(ref CUDA_ARRAY_DESCRIPTOR pArrayDescriptor, CUarray hArray);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetByPCIBusId(ref CUdevice dev, string pciBusId);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetPCIBusId(string pciBusId, int len, CUdevice dev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcGetEventHandle(ref CUipcEventHandle pHandle, CUevent cuEvent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcGetMemHandle(ref CUipcMemHandle pHandle, CUdeviceptr dptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcOpenEventHandle(ref CUevent phEvent, CUipcEventHandle handle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcOpenMemHandle(ref CUdeviceptr pdptr, CUipcMemHandle handle, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAlloc(ref CUdeviceptr dptr, size_t bytesize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAllocHost(IntPtr pp, size_t bytesize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAllocManaged(ref CUdeviceptr dptr, size_t bytesize, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAllocPitch(ref CUdeviceptr dptr, ref size_t pPitch, size_t WidthInBytes, size_t Height, uint ElementSizeBytes);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemFree(CUdeviceptr dptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemFreeHost(IntPtr p);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemGetAddressRange(ref CUdeviceptr pbase, ref size_t psize, CUdeviceptr dptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemGetInfo(ref size_t free, ref size_t total);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostAlloc(IntPtr pp, size_t bytesize, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostGetDevicePointer(ref CUdeviceptr pdptr, IntPtr p, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostGetFlags(ref uint pFlags, IntPtr p);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostRegister(IntPtr p, size_t bytesize, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostUnregister(IntPtr p);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy2D(ref CUDA_MEMCPY2D pCopy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy2DAsync(ref CUDA_MEMCPY2D pCopy, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy2DUnaligned(ref CUDA_MEMCPY2D pCopy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3D(ref CUDA_MEMCPY3D pCopy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3DAsync(ref CUDA_MEMCPY3D pCopy, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3DPeer(ref CUDA_MEMCPY3D_PEER pCopy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3DPeerAsync(ref CUDA_MEMCPY3D_PEER pCopy, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoH(IntPtr dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoHAsync(IntPtr dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoHAsync(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, IntPtr srcHost, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, IntPtr srcHost, size_t ByteCount, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, IntPtr srcHost, size_t ByteCount, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD16(CUdeviceptr dstDevice, ushort us, size_t N);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD16Async(CUdeviceptr dstDevice, ushort us, size_t N, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, byte uc, size_t Width, size_t Height);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, byte uc, size_t Width, size_t Height, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD32(CUdeviceptr dstDevice, uint ui, size_t N);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD32Async(CUdeviceptr dstDevice, uint ui, size_t N, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD8(CUdeviceptr dstDevice, byte uc, size_t N);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD8Async(CUdeviceptr dstDevice, byte uc, size_t N, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMipmappedArrayCreate(ref CUmipmappedArray pHandle, ref CUDA_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc, uint numMipmapLevels);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMipmappedArrayGetLevel(ref CUarray pLevelArray, CUmipmappedArray hMipmappedArray, uint level);

			// ----- Unified Addressing

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemRangeGetAttribute(IntPtr data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemRangeGetAttributes(IntPtr data, ref size_t dataSizes, ref CUmem_range_attribute attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuPointerGetAttribute(IntPtr data, CUpointer_attribute attribute, CUdeviceptr ptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuPointerGetAttributes(uint numAttributes, ref CUpointer_attribute attributes, IntPtr data, CUdeviceptr ptr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuPointerSetAttribute(IntPtr value, CUpointer_attribute attribute, CUdeviceptr ptr);

			// ----- Stream Management

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, IntPtr userData, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamCreate(ref CUstream phStream, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamCreateWithPriority(ref CUstream phStream, uint flags, int priority);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamDestroy(CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamGetFlags(CUstream hStream, ref uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamGetPriority(CUstream hStream, ref int priority);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamQuery(CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamSynchronize(CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, uint Flags);

			// ----- Event Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventCreate(ref CUevent phEvent, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventDestroy(CUevent hEvent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventElapsedTime(ref float pMilliseconds, CUevent hStart, CUevent hEnd);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventQuery(CUevent hEvent);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventRecord(CUevent hEvent, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventSynchronize(CUevent hEvent);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamBatchMemOp(CUstream stream, uint count, ref CUstreamBatchMemOpParams paramArray, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, uint flags);

			// ----- Execution Control

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuFuncGetAttribute(ref int pi, CUfunction_attribute attrib, CUfunction hfunc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLaunchKernel(
				CUfunction f,
				uint gridDimX, uint gridDimY, uint gridDimZ,
				uint blockDimX, uint blockDimY, uint blockDimZ,
				uint sharedMemBytes, CUstream hStream,
				IntPtr kernelParams,
				IntPtr extra);

			// ----- Execution Control [DEPRECATED]

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuFuncSetSharedSize(CUfunction hfunc, uint bytes);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLaunch(CUfunction f);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetSize(CUfunction hfunc, uint numbytes);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetf(CUfunction hfunc, int offset, float value);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSeti(CUfunction hfunc, int offset, uint value);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetv(CUfunction hfunc, int offset, IntPtr ptr, uint numbytes);

			// ----- Occupancy

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(ref int numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref int numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuOccupancyMaxPotentialBlockSize(ref int minGridSize, ref int blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(ref int minGridSize, ref int blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, uint flags);

			// ----- Texture Reference Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetAddress(ref CUdeviceptr pdptr, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetAddressMode(ref CUaddress_mode pam, CUtexref hTexRef, int dim);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetArray(ref CUarray phArray, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetBorderColor(ref float pBorderColor, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetFilterMode(ref CUfilter_mode pfm, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetFlags(ref uint pFlags, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetFormat(ref CUarray_format pFormat, ref int pNumChannels, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetMaxAnisotropy(ref int pmaxAniso, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetMipmapFilterMode(ref CUfilter_mode pfm, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetMipmapLevelBias(ref float pbias, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetMipmapLevelClamp(ref float pminMipmapLevelClamp, ref float pmaxMipmapLevelClamp, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefGetMipmappedArray(ref CUmipmappedArray phMipmappedArray, CUtexref hTexRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetAddress(ref size_t ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetAddress2D(CUtexref hTexRef, ref CUDA_ARRAY_DESCRIPTOR desc, CUdeviceptr dptr, size_t Pitch);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetBorderColor(CUtexref hTexRef, ref float pBorderColor);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetFlags(CUtexref hTexRef, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, uint maxAniso);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, uint Flags);

			// ----- Texture Reference Management [DEPRECATED]

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefCreate(ref CUtexref pTexRef);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuTexRefDestroy(CUtexref hTexRef);

			// ----- Surface Reference Management

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuSurfRefGetArray(ref CUarray phArray, CUsurfref hSurfRef);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, uint Flags);

			// ----- Texture Object Management

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuTexObjectCreate(ref CUtexObject pTexObject, ref CUDA_RESOURCE_DESC pResDesc, ref CUDA_TEXTURE_DESC pTexDesc, ref CUDA_RESOURCE_VIEW_DESC pResViewDesc);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuTexObjectDestroy(CUtexObject texObject);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuTexObjectGetResourceDesc(ref CUDA_RESOURCE_DESC pResDesc, CUtexObject texObject);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuTexObjectGetResourceViewDesc(ref CUDA_RESOURCE_VIEW_DESC pResViewDesc, CUtexObject texObject);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuTexObjectGetTextureDesc(ref CUDA_TEXTURE_DESC pTexDesc, CUtexObject texObject);

			// ----- Surface Object Management

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuSurfObjectCreate(ref CUsurfObject pSurfObject, ref CUDA_RESOURCE_DESC pResDesc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuSurfObjectDestroy(CUsurfObject surfObject);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuSurfObjectGetResourceDesc(ref CUDA_RESOURCE_DESC pResDesc, CUsurfObject surfObject);

			// ----- Peer Context Memory Access

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxDisablePeerAccess(CUcontext peerContext);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxEnablePeerAccess(CUcontext peerContext, uint Flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceCanAccessPeer(ref int canAccessPeer, CUdevice dev, CUdevice peerDev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetP2PAttribute(ref int value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);

			// ----- Graphics Interoperability

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGraphicsMapResources(uint count, ref CUgraphicsResource resources, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGraphicsResourceGetMappedMipmappedArray(ref CUmipmappedArray pMipmappedArray, CUgraphicsResource resource);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGraphicsResourceGetMappedPointer(ref CUdeviceptr pDevPtr, ref size_t pSize, CUgraphicsResource resource);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGraphicsSubResourceGetMappedArray(ref CUarray pArray, CUgraphicsResource resource, uint arrayIndex, uint mipLevel);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGraphicsUnmapResources(uint count, ref CUgraphicsResource resources, CUstream hStream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource);

			// ----- Profiler Control

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuProfilerInitialize(string configFile, string outputFile, CUoutput_mode outputMode);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuProfilerStart();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuProfilerStop();

			// ----- OpenGL Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLGetDevices(ref uint pCudaDeviceCount, ref CUdevice pCudaDevices, uint cudaDeviceCount, CUGLDeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsGLRegisterBuffer(ref CUgraphicsResource pCudaResource, GLuint buffer, uint Flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsGLRegisterImage(ref CUgraphicsResource pCudaResource, GLuint image, GLenum target, uint Flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuWGLGetDevice(ref CUdevice pDevice, HGPUNV hGpu);

			// ----- OpenGL Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLCtxCreate(ref CUcontext pCtx, uint Flags, CUdevice device);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGLInit();

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLMapBufferObject(ref CUdeviceptr dptr, ref size_t size, GLuint buffer);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLMapBufferObjectAsync(ref CUdeviceptr dptr, ref size_t size, GLuint buffer, CUstream hStream);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLRegisterBufferObject(GLuint buffer);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLSetBufferObjectMapFlags(GLuint buffer, uint Flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLUnmapBufferObject(GLuint buffer);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGLUnregisterBufferObject(GLuint buffer);

			// ----- Direct3D 9 Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, uint Flags, ref IDirect3DDevice9 pD3DDevice);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9CtxCreateOnDevice(ref CUcontext pCtx, uint flags, ref IDirect3DDevice9 pD3DDevice, CUdevice cudaDevice);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuD3D9GetDevice(ref CUdevice pCudaDevice, string pszAdapterName);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9GetDevices(ref uint pCudaDeviceCount, ref CUdevice pCudaDevices, uint cudaDeviceCount, ref IDirect3DDevice9 pD3D9Device, CUd3d9DeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9GetDirect3DDevice(ref IDirect3DDevice9 ppD3DDevice);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsD3D9RegisterResource(ref CUgraphicsResource pCudaResource, ref IDirect3DResource9 pD3DResource, uint Flags);

			// ----- Direct3D 9 Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9MapResources(uint count, ref IDirect3DResource9 ppResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9RegisterResource(ref IDirect3DResource9 pResource, uint Flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9ResourceGetMappedArray(ref CUarray pArray, ref IDirect3DResource9 pResource, uint Face, uint Level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9ResourceGetMappedPitch(ref size_t pPitch, ref size_t pPitchSlice, ref IDirect3DResource9 pResource, uint Face, uint Level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, ref IDirect3DResource9 pResource, uint Face, uint Level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9ResourceGetMappedSize(ref size_t pSize, ref IDirect3DResource9 pResource, uint Face, uint Level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9ResourceGetSurfaceDimensions(ref size_t pWidth, ref size_t pHeight, ref size_t pDepth, ref IDirect3DResource9 pResource, uint Face, uint Level);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9ResourceSetMapFlags(ref IDirect3DResource9 pResource, uint Flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9UnmapResources(uint count, ref IDirect3DResource9 ppResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D9UnregisterResource(ref IDirect3DResource9 pResource);

			// ----- Direct3D 10 Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10GetDevice(ref CUdevice pCudaDevice, ref IDXGIAdapter pAdapter);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10GetDevices(ref uint pCudaDeviceCount, ref CUdevice pCudaDevices, uint cudaDeviceCount, ref ID3D10Device pD3D10Device, CUd3d10DeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsD3D10RegisterResource(ref CUgraphicsResource pCudaResource, ref ID3D10Resource pD3DResource, uint Flags);

			// ----- Direct3D 10 Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, uint Flags, ref ID3D10Device pD3DDevice);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10CtxCreateOnDevice(ref CUcontext pCtx, uint flags, ref ID3D10Device pD3DDevice, CUdevice cudaDevice);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10GetDirect3DDevice(ref ID3D10Device ppD3DDevice);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10MapResources(uint count, ref ID3D10Resource ppResources);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10RegisterResource(ref ID3D10Resource pResource, uint Flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10ResourceGetMappedArray(ref CUarray pArray, ref ID3D10Resource pResource, uint SubResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10ResourceGetMappedPitch(ref size_t pPitch, ref size_t pPitchSlice, ref ID3D10Resource pResource, uint SubResource)

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, ref ID3D10Resource pResource, uint SubResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10ResourceGetMappedSize(ref size_t pSize, ref ID3D10Resource pResource, uint SubResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10ResourceGetSurfaceDimensions(ref size_t pWidth, ref size_t pHeight, ref size_t pDepth, ref ID3D10Resource pResource, uint SubResource);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10ResourceSetMapFlags(ref ID3D10Resource pResource, uint Flags);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10UnmapResources(uint count, ref ID3D10Resource ppResources);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D10UnregisterResource(ref ID3D10Resource pResource);

			// ----- Direct3D 11 Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D11GetDevice(ref CUdevice pCudaDevice, ref IDXGIAdapter pAdapter);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D11GetDevices(ref uint pCudaDeviceCount, ref CUdevice pCudaDevices, uint cudaDeviceCount, ref ID3D11Device pD3D11Device, CUd3d11DeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsD3D11RegisterResource(ref CUgraphicsResource pCudaResource, ref ID3D11Resource pD3DResource, uint Flags);

			// ----- Direct3D 11 Interoperability [DEPRECATED]

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D11CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, uint Flags, ref ID3D11Device pD3DDevice);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D11CtxCreateOnDevice(ref CUcontext pCtx, uint flags, ref ID3D11Device pD3DDevice, CUdevice cudaDevice);

			//[Obsolete]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuD3D11GetDirect3DDevice(ref ID3D11Device ppD3DDevice);

			// ----- VDPAU Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsVDPAURegisterOutputSurface(ref CUgraphicsResource pCudaResource, VdpOutputSurface vdpSurface, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsVDPAURegisterVideoSurface(ref CUgraphicsResource pCudaResource, VdpVideoSurface vdpSurface, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuVDPAUCtxCreate(ref CUcontext pCtx, uint flags, CUdevice device, VdpDevice vdpDevice, ref VdpGetProcAddress vdpGetProcAddress);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuVDPAUGetDevice(ref CUdevice pDevice, VdpDevice vdpDevice, ref VdpGetProcAddress vdpGetProcAddress);

			// ----- EGL Interoperability

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamConsumerAcquireFrame(ref CUeglStreamConnection conn, ref CUgraphicsResource pCudaResource, ref CUstream pStream, uint timeout);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamConsumerConnect(ref CUeglStreamConnection conn, EGLStreamKHR stream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamConsumerConnectWithFlags(ref CUeglStreamConnection conn, EGLStreamKHR stream, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamConsumerDisconnect(ref CUeglStreamConnection conn);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamConsumerReleaseFrame(ref CUeglStreamConnection conn, CUgraphicsResource pCudaResource, ref CUstream pStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamProducerConnect(ref CUeglStreamConnection conn, EGLStreamKHR stream, EGLint width, EGLint height);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamProducerDisconnect(ref CUeglStreamConnection conn);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamProducerPresentFrame(ref CUeglStreamConnection conn, CUeglFrame eglframe, ref CUstream pStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuEGLStreamProducerReturnFrame(ref CUeglStreamConnection conn, ref CUeglFrame eglframe, ref CUstream pStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsEGLRegisterImage(ref CUgraphicsResource pCudaResource, EGLImageKHR image, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuGraphicsResourceGetMappedEglFrame(ref CUeglFrame eglFrame, CUgraphicsResource resource, uint index, uint mipLevel);
		}
	}
}
