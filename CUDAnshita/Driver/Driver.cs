using System;
using System.Runtime.InteropServices;
using System.Text;

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
	using CUstreamCallback = IntPtr;
	using CUgraphicsResource = IntPtr;
	using CUtexObject = UInt64;
	using CUsurfObject = UInt64;
	using CUoccupancyB2DSize = IntPtr;

	using cuuint32_t = UInt32;
	using cuuint64_t = UInt64;
	using size_t = Int64;

	//using CUresult = cudaError;

	/// <summary>
	/// NVIDIA CUDA Driver API
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cuda-driver-api/">http://docs.nvidia.com/cuda/cuda-driver-api/</a>
	/// </remarks>
	public class Driver {
		public class API {
			const string DLL_PATH = "nvcuda.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			// ----- Error Handling

			/// <summary>
			/// Gets the string representation of an error code enum name.
			/// </summary>
			/// <param name="error">Error code to convert to string.</param>
			/// <param name="pStr">Address of the string pointer.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGetErrorName(CUresult error, ref IntPtr pStr);

			/// <summary>
			/// Gets the string description of an error code.
			/// </summary>
			/// <param name="error">Error code to convert to string.</param>
			/// <param name="pStr">Address of the string pointer.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuGetErrorString(CUresult error, ref IntPtr pStr);

			// ----- Initialization

			/// <summary>
			/// Initialize the CUDA driver API.
			/// </summary>
			/// <param name="Flags">Initialization flag for CUDA.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
			/// CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuInit(uint Flags);

			// ----- Version Management

			/// <summary>
			/// Returns the latest CUDA version supported by driver.
			/// </summary>
			/// <param name="driverVersion">Returns the CUDA driver version.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDriverGetVersion(ref int driverVersion);

			// ----- Device Management

			/// <summary>
			/// Returns a handle to a compute device.
			/// </summary>
			/// <param name="device">Returned device handle.</param>
			/// <param name="ordinal">Device number to get handle for.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
			/// CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGet(ref CUdevice device, int ordinal);

			/// <summary>
			/// Returns information about the device.
			/// </summary>
			/// <param name="pi">Returned device attribute value.</param>
			/// <param name="attrib">Device attribute to query.</param>
			/// <param name="dev">Device handle.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
			/// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetAttribute(ref int pi, CUdevice_attribute attrib, CUdevice dev);

			/// <summary>
			/// Returns the number of compute-capable devices.
			/// </summary>
			/// <param name="count">Returned number of compute-capable devices.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetCount(ref int count);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev);

			/// <summary>
			/// Returns an identifer string for the device.
			/// </summary>
			/// <param name="name">Returned identifier string for the device.</param>
			/// <param name="len">Maximum length of string to store in name.</param>
			/// <param name="dev">Device to get identifier string for.</param>
			/// <returns>CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetName(StringBuilder name, int len, CUdevice dev);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev);

			/// <summary>
			/// Returns the total amount of memory on the device.
			/// </summary>
			/// <param name="bytes"></param>
			/// <param name="dev"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceTotalMem(ref size_t bytes, CUdevice dev);

			// ----- Device Management [DEPRECATED]

			/// <summary>
			/// Returns the compute capability of the device.
			/// </summary>
			/// <param name="major"></param>
			/// <param name="minor"></param>
			/// <param name="dev"></param>
			/// <returns></returns>
			[Obsolete("This function was deprecated as of CUDA 5.0 and its functionality superceded by cuDeviceGetAttribute().")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceComputeCapability(ref int major, ref int minor, CUdevice dev);

			/// <summary>
			/// Returns properties for a selected device.
			/// </summary>
			/// <param name="prop"></param>
			/// <param name="dev"></param>
			/// <returns></returns>
			[Obsolete("This function was deprecated as of CUDA 5.0 and replaced by cuDeviceGetAttribute().")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetProperties(ref CUdevprop prop, CUdevice dev);

			// ----- Primary Context Management

			/// <summary>
			/// Get the state of the primary context.
			/// </summary>
			/// <param name="dev"></param>
			/// <param name="flags"></param>
			/// <param name="active"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxGetState(CUdevice dev, ref uint flags, ref int active);

			/// <summary>
			/// Release the primary context on the GPU.
			/// </summary>
			/// <param name="dev"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxRelease(CUdevice dev);

			/// <summary>
			/// Destroy all allocations and reset all state on the primary context.
			/// </summary>
			/// <param name="dev"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxReset(CUdevice dev);

			/// <summary>
			/// Retain the primary context on the GPU.
			/// </summary>
			/// <param name="pctx"></param>
			/// <param name="dev"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxRetain(ref CUcontext pctx, CUdevice dev);

			/// <summary>
			/// Set flags for the primary context.
			/// </summary>
			/// <param name="dev"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, uint flags);

			// ----- Context Management

			/// <summary>
			/// Create a CUDA context.
			/// </summary>
			/// <param name="pctx"></param>
			/// <param name="flags"></param>
			/// <param name="dev"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxCreate(ref CUcontext pctx, uint flags, CUdevice dev);

			/// <summary>
			/// Destroy a CUDA context.
			/// </summary>
			/// <param name="ctx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxDestroy(CUcontext ctx);

			/// <summary>
			/// Gets the context's API version.
			/// </summary>
			/// <param name="ctx"></param>
			/// <param name="version"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetApiVersion(CUcontext ctx, ref uint version);

			/// <summary>
			/// Returns the preferred cache configuration for the current context.
			/// </summary>
			/// <param name="pconfig"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetCacheConfig(ref CUfunc_cache pconfig);

			/// <summary>
			/// Returns the CUDA context bound to the calling CPU thread.
			/// </summary>
			/// <param name="pctx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetCurrent(ref CUcontext pctx);

			/// <summary>
			/// Returns the device ID for the current context.
			/// </summary>
			/// <param name="device"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetDevice(ref CUdevice device);

			/// <summary>
			/// Returns the flags for the current context.
			/// </summary>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetFlags(ref uint flags);

			/// <summary>
			/// Returns resource limits.
			/// </summary>
			/// <param name="pvalue"></param>
			/// <param name="limit"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetLimit(ref size_t pvalue, CUlimit limit);

			/// <summary>
			/// Returns the current shared memory configuration for the current context.
			/// </summary>
			/// <param name="pConfig"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetSharedMemConfig(ref CUsharedconfig pConfig);

			/// <summary>
			/// Returns numerical values that correspond to the least and greatest stream priorities.
			/// </summary>
			/// <param name="leastPriority"></param>
			/// <param name="greatestPriority"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxGetStreamPriorityRange(ref int leastPriority, ref int greatestPriority);

			/// <summary>
			/// Pops the current CUDA context from the current CPU thread.
			/// </summary>
			/// <param name="pctx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxPopCurrent(ref CUcontext pctx);

			/// <summary>
			/// Pushes a context on the current CPU thread.
			/// </summary>
			/// <param name="ctx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxPushCurrent(CUcontext ctx);

			/// <summary>
			/// Sets the preferred cache configuration for the current context.
			/// </summary>
			/// <param name="config"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetCacheConfig(CUfunc_cache config);

			/// <summary>
			/// Binds the specified CUDA context to the calling CPU thread.
			/// </summary>
			/// <param name="ctx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetCurrent(CUcontext ctx);

			/// <summary>
			/// Set resource limits.
			/// </summary>
			/// <param name="limit"></param>
			/// <param name="value"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetLimit(CUlimit limit, size_t value);

			/// <summary>
			/// Sets the shared memory configuration for the current context.
			/// </summary>
			/// <param name="config"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSetSharedMemConfig(CUsharedconfig config);

			/// <summary>
			/// Block for a context's tasks to complete.
			/// </summary>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxSynchronize();

			// ----- Context Management [DEPRECATED]

			/// <summary>
			/// Increment a context's usage-count.
			/// </summary>
			/// <param name="pctx"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[Obsolete("this function is deprecated and should not be used")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxAttach(ref CUcontext pctx, uint flags);

			/// <summary>
			/// Decrement a context's usage-count.
			/// </summary>
			/// <param name="ctx"></param>
			/// <returns></returns>
			[Obsolete("this function is deprecated and should not be used")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuCtxDetach(CUcontext ctx);

			// ----- Module Management

			/// <summary>
			/// Add an input to a pending linker invocation.
			/// </summary>
			/// <param name="state"></param>
			/// <param name="type"></param>
			/// <param name="data"></param>
			/// <param name="size"></param>
			/// <param name="name"></param>
			/// <param name="numOptions"></param>
			/// <param name="options"></param>
			/// <param name="optionValues"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, IntPtr data, size_t size, string name, uint numOptions, ref CUjit_option options, ref IntPtr optionValues);

			/// <summary>
			/// Add a file input to a pending linker invocation.
			/// </summary>
			/// <param name="state"></param>
			/// <param name="type"></param>
			/// <param name="path"></param>
			/// <param name="numOptions"></param>
			/// <param name="options"></param>
			/// <param name="optionValues"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, string path, uint numOptions, ref CUjit_option options, ref IntPtr optionValues);

			/// <summary>
			/// Complete a pending linker invocation.
			/// </summary>
			/// <param name="state"></param>
			/// <param name="cubinOut"></param>
			/// <param name="sizeOut"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkComplete(CUlinkState state, IntPtr cubinOut, ref size_t sizeOut);

			/// <summary>
			/// Creates a pending JIT linker invocation.
			/// </summary>
			/// <param name="numOptions"></param>
			/// <param name="options"></param>
			/// <param name="optionValues"></param>
			/// <param name="stateOut"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkCreate(uint numOptions, ref CUjit_option options, IntPtr optionValues, ref CUlinkState stateOut);

			/// <summary>
			/// Destroys state for a JIT linker invocation.
			/// </summary>
			/// <param name="state"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLinkDestroy(CUlinkState state);

			/// <summary>
			/// Returns a function handle.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="hmod"></param>
			/// <param name="name"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetFunction(ref CUfunction hfunc, CUmodule hmod, string name);

			/// <summary>
			/// Returns a global pointer from a module.
			/// </summary>
			/// <param name="dptr"></param>
			/// <param name="bytes"></param>
			/// <param name="hmod"></param>
			/// <param name="name"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetGlobal(ref CUdeviceptr dptr, ref size_t bytes, CUmodule hmod, string name);

			/// <summary>
			/// Returns a handle to a surface reference.
			/// </summary>
			/// <param name="pSurfRef"></param>
			/// <param name="hmod"></param>
			/// <param name="name"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetSurfRef(ref CUsurfref pSurfRef, CUmodule hmod, string name);

			/// <summary>
			/// Returns a handle to a texture reference.
			/// </summary>
			/// <param name="pTexRef"></param>
			/// <param name="hmod"></param>
			/// <param name="name"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleGetTexRef(ref CUtexref pTexRef, CUmodule hmod, string name);

			/// <summary>
			/// Loads a compute module.
			/// </summary>
			/// <param name="module"></param>
			/// <param name="fname"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoad(ref CUmodule module, string fname);

			/// <summary>
			/// Load a module's data.
			/// </summary>
			/// <param name="module"></param>
			/// <param name="image"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoadData(ref CUmodule module, IntPtr image);

			/// <summary>
			/// Load a module's data with options.
			/// </summary>
			/// <param name="module"></param>
			/// <param name="image"></param>
			/// <param name="numOptions"></param>
			/// <param name="options"></param>
			/// <param name="optionValues"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoadDataEx(ref CUmodule module, IntPtr image, uint numOptions, CUjit_option options, IntPtr optionValues);

			/// <summary>
			/// Load a module's data.
			/// </summary>
			/// <param name="module"></param>
			/// <param name="fatCubin"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleLoadFatBinary(ref CUmodule module, IntPtr fatCubin);

			/// <summary>
			/// Unloads a module.
			/// </summary>
			/// <param name="hmod"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuModuleUnload(CUmodule hmod);

			// ----- Memory Management

			/// <summary>
			/// Creates a 3D CUDA array.
			/// </summary>
			/// <param name="pHandle"></param>
			/// <param name="pAllocateArray"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArray3DCreate(ref CUarray pHandle, ref CUDA_ARRAY3D_DESCRIPTOR pAllocateArray);

			/// <summary>
			/// Get a 3D CUDA array descriptor.
			/// </summary>
			/// <param name="pArrayDescriptor"></param>
			/// <param name="hArray"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArray3DGetDescriptor(ref CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor, CUarray hArray);

			/// <summary>
			/// Creates a 1D or 2D CUDA array.
			/// </summary>
			/// <param name="pHandle"></param>
			/// <param name="pAllocateArray"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArrayCreate(ref CUarray pHandle, ref CUDA_ARRAY_DESCRIPTOR pAllocateArray);

			/// <summary>
			/// Destroys a CUDA array.
			/// </summary>
			/// <param name="hArray"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArrayDestroy(CUarray hArray);

			/// <summary>
			/// Get a 1D or 2D CUDA array descriptor.
			/// </summary>
			/// <param name="pArrayDescriptor"></param>
			/// <param name="hArray"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuArrayGetDescriptor(ref CUDA_ARRAY_DESCRIPTOR pArrayDescriptor, CUarray hArray);

			/// <summary>
			/// Returns a handle to a compute device.
			/// </summary>
			/// <param name="dev"></param>
			/// <param name="pciBusId"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetByPCIBusId(ref CUdevice dev, string pciBusId);

			/// <summary>
			/// Returns a PCI Bus Id string for the device.
			/// </summary>
			/// <param name="pciBusId"></param>
			/// <param name="len"></param>
			/// <param name="dev"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuDeviceGetPCIBusId(StringBuilder pciBusId, int len, CUdevice dev);

			/// <summary>
			/// Close memory mapped with cuIpcOpenMemHandle.
			/// </summary>
			/// <param name="dptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);

			/// <summary>
			/// Gets an interprocess handle for a previously allocated event.
			/// </summary>
			/// <param name="pHandle"></param>
			/// <param name="cuEvent"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcGetEventHandle(ref CUipcEventHandle pHandle, CUevent cuEvent);

			/// <summary>
			/// Gets an interprocess memory handle for an existing device memory allocation.
			/// </summary>
			/// <param name="pHandle"></param>
			/// <param name="dptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcGetMemHandle(ref CUipcMemHandle pHandle, CUdeviceptr dptr);

			/// <summary>
			/// Opens an interprocess event handle for use in the current process.
			/// </summary>
			/// <param name="phEvent"></param>
			/// <param name="handle"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcOpenEventHandle(ref CUevent phEvent, CUipcEventHandle handle);

			/// <summary>
			/// Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
			/// </summary>
			/// <param name="pdptr"></param>
			/// <param name="handle"></param>
			/// <param name="Flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuIpcOpenMemHandle(ref CUdeviceptr pdptr, CUipcMemHandle handle, uint Flags);

			/// <summary>
			/// Allocates device memory.
			/// </summary>
			/// <param name="dptr"></param>
			/// <param name="bytesize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAlloc(ref CUdeviceptr dptr, size_t bytesize);

			/// <summary>
			/// Allocates page-locked host memory.
			/// </summary>
			/// <param name="pp"></param>
			/// <param name="bytesize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAllocHost(ref IntPtr pp, size_t bytesize);

			/// <summary>
			/// Allocates memory that will be automatically managed by the Unified Memory system.
			/// </summary>
			/// <param name="dptr"></param>
			/// <param name="bytesize"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAllocManaged(ref CUdeviceptr dptr, size_t bytesize, uint flags);

			/// <summary>
			/// Allocates pitched device memory.
			/// </summary>
			/// <param name="dptr"></param>
			/// <param name="pPitch"></param>
			/// <param name="WidthInBytes"></param>
			/// <param name="Height"></param>
			/// <param name="ElementSizeBytes"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAllocPitch(ref CUdeviceptr dptr, ref size_t pPitch, size_t WidthInBytes, size_t Height, uint ElementSizeBytes);

			/// <summary>
			/// Frees device memory.
			/// </summary>
			/// <param name="dptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemFree(CUdeviceptr dptr);

			/// <summary>
			/// Frees page-locked host memory.
			/// </summary>
			/// <param name="p"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemFreeHost(IntPtr p);

			/// <summary>
			/// Get information on memory allocations.
			/// </summary>
			/// <param name="pbase"></param>
			/// <param name="psize"></param>
			/// <param name="dptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemGetAddressRange(ref CUdeviceptr pbase, ref size_t psize, CUdeviceptr dptr);

			/// <summary>
			/// Gets free and total memory.
			/// </summary>
			/// <param name="free"></param>
			/// <param name="total"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemGetInfo(ref size_t free, ref size_t total);

			/// <summary>
			/// Allocates page-locked host memory.
			/// </summary>
			/// <param name="pp"></param>
			/// <param name="bytesize"></param>
			/// <param name="Flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostAlloc(ref IntPtr pp, size_t bytesize, uint Flags);

			/// <summary>
			/// Passes back device pointer of mapped pinned memory.
			/// </summary>
			/// <param name="pdptr"></param>
			/// <param name="p"></param>
			/// <param name="Flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostGetDevicePointer(ref CUdeviceptr pdptr, IntPtr p, uint Flags);

			/// <summary>
			/// Passes back flags that were used for a pinned allocation.
			/// </summary>
			/// <param name="pFlags"></param>
			/// <param name="p"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostGetFlags(ref uint pFlags, IntPtr p);

			/// <summary>
			/// Registers an existing host memory range for use by CUDA.
			/// </summary>
			/// <param name="p"></param>
			/// <param name="bytesize"></param>
			/// <param name="Flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostRegister(IntPtr p, size_t bytesize, uint Flags);

			/// <summary>
			/// Unregisters a memory range that was registered with cuMemHostRegister.
			/// </summary>
			/// <param name="p"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemHostUnregister(IntPtr p);

			/// <summary>
			/// Copies memory.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);

			/// <summary>
			/// Copies memory for 2D arrays.
			/// </summary>
			/// <param name="pCopy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy2D(ref CUDA_MEMCPY2D pCopy);

			/// <summary>
			/// Copies memory for 2D arrays.
			/// </summary>
			/// <param name="pCopy"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy2DAsync(ref CUDA_MEMCPY2D pCopy, CUstream hStream);

			/// <summary>
			/// Copies memory for 2D arrays.
			/// </summary>
			/// <param name="pCopy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy2DUnaligned(ref CUDA_MEMCPY2D pCopy);

			/// <summary>
			/// Copies memory for 3D arrays.
			/// </summary>
			/// <param name="pCopy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3D(ref CUDA_MEMCPY3D pCopy);

			/// <summary>
			/// Copies memory for 3D arrays.
			/// </summary>
			/// <param name="pCopy"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3DAsync(ref CUDA_MEMCPY3D pCopy, CUstream hStream);

			/// <summary>
			/// Copies memory between contexts.
			/// </summary>
			/// <param name="pCopy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3DPeer(ref CUDA_MEMCPY3D_PEER pCopy);

			/// <summary>
			/// Copies memory between contexts asynchronously.
			/// </summary>
			/// <param name="pCopy"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpy3DPeerAsync(ref CUDA_MEMCPY3D_PEER pCopy, CUstream hStream);

			/// <summary>
			/// Copies memory asynchronously.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="ByteCount"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);

			/// <summary>
			/// Copies memory from Array to Array.
			/// </summary>
			/// <param name="dstArray"></param>
			/// <param name="dstOffset"></param>
			/// <param name="srcArray"></param>
			/// <param name="srcOffset"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);

			/// <summary>
			/// Copies memory from Array to Device.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="srcArray"></param>
			/// <param name="srcOffset"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);

			/// <summary>
			/// Copies memory from Array to Host.
			/// </summary>
			/// <param name="dstHost"></param>
			/// <param name="srcArray"></param>
			/// <param name="srcOffset"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoH(IntPtr dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);

			/// <summary>
			/// Copies memory from Array to Host.
			/// </summary>
			/// <param name="dstHost"></param>
			/// <param name="srcArray"></param>
			/// <param name="srcOffset"></param>
			/// <param name="ByteCount"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyAtoHAsync(IntPtr dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);

			/// <summary>
			/// Copies memory from Device to Array.
			/// </summary>
			/// <param name="dstArray"></param>
			/// <param name="dstOffset"></param>
			/// <param name="srcDevice"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);

			/// <summary>
			/// Copies memory from Device to Device.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="srcDevice"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);

			/// <summary>
			/// Copies memory from Device to Device.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="srcDevice"></param>
			/// <param name="ByteCount"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

			/// <summary>
			/// Copies memory from Device to Host.
			/// </summary>
			/// <param name="dstHost"></param>
			/// <param name="srcDevice"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount);

			/// <summary>
			/// Copies memory from Device to Host.
			/// </summary>
			/// <param name="dstHost"></param>
			/// <param name="srcDevice"></param>
			/// <param name="ByteCount"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyDtoHAsync(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

			/// <summary>
			/// Copies memory from Host to Array.
			/// </summary>
			/// <param name="dstArray"></param>
			/// <param name="dstOffset"></param>
			/// <param name="srcHost"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, IntPtr srcHost, size_t ByteCount);

			/// <summary>
			/// Copies memory from Host to Array.
			/// </summary>
			/// <param name="dstArray"></param>
			/// <param name="dstOffset"></param>
			/// <param name="srcHost"></param>
			/// <param name="ByteCount"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, IntPtr srcHost, size_t ByteCount, CUstream hStream);

			/// <summary>
			/// Copies memory from Host to Device.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="srcHost"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, size_t ByteCount);

			/// <summary>
			/// Copies memory from Host to Device.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="srcHost"></param>
			/// <param name="ByteCount"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, IntPtr srcHost, size_t ByteCount, CUstream hStream);

			/// <summary>
			/// Copies device memory between two contexts.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstContext"></param>
			/// <param name="srcDevice"></param>
			/// <param name="srcContext"></param>
			/// <param name="ByteCount"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);

			/// <summary>
			/// Copies device memory between two contexts asynchronously.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstContext"></param>
			/// <param name="srcDevice"></param>
			/// <param name="srcContext"></param>
			/// <param name="ByteCount"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);

			/// <summary>
			/// Initializes device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="us"></param>
			/// <param name="N"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD16(CUdeviceptr dstDevice, ushort us, size_t N);

			/// <summary>
			/// Sets device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="us"></param>
			/// <param name="N"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD16Async(CUdeviceptr dstDevice, ushort us, size_t N, CUstream hStream);

			/// <summary>
			/// Initializes device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstPitch"></param>
			/// <param name="us"></param>
			/// <param name="Width"></param>
			/// <param name="Height"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height);

			/// <summary>
			/// Sets device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstPitch"></param>
			/// <param name="us"></param>
			/// <param name="Width"></param>
			/// <param name="Height"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height, CUstream hStream);

			/// <summary>
			/// Initializes device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstPitch"></param>
			/// <param name="ui"></param>
			/// <param name="Width"></param>
			/// <param name="Height"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height);

			/// <summary>
			/// Sets device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstPitch"></param>
			/// <param name="ui"></param>
			/// <param name="Width"></param>
			/// <param name="Height"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height, CUstream hStream);

			/// <summary>
			/// Initializes device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstPitch"></param>
			/// <param name="uc"></param>
			/// <param name="Width"></param>
			/// <param name="Height"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, byte uc, size_t Width, size_t Height);

			/// <summary>
			/// Sets device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="dstPitch"></param>
			/// <param name="uc"></param>
			/// <param name="Width"></param>
			/// <param name="Height"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, byte uc, size_t Width, size_t Height, CUstream hStream);

			/// <summary>
			/// Initializes device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="ui"></param>
			/// <param name="N"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD32(CUdeviceptr dstDevice, uint ui, size_t N);

			/// <summary>
			/// Sets device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="ui"></param>
			/// <param name="N"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD32Async(CUdeviceptr dstDevice, uint ui, size_t N, CUstream hStream);

			/// <summary>
			/// Initializes device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="uc"></param>
			/// <param name="N"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD8(CUdeviceptr dstDevice, byte uc, size_t N);

			/// <summary>
			/// Sets device memory.
			/// </summary>
			/// <param name="dstDevice"></param>
			/// <param name="uc"></param>
			/// <param name="N"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemsetD8Async(CUdeviceptr dstDevice, byte uc, size_t N, CUstream hStream);

			/// <summary>
			/// Creates a CUDA mipmapped array.
			/// </summary>
			/// <param name="pHandle"></param>
			/// <param name="pMipmappedArrayDesc"></param>
			/// <param name="numMipmapLevels"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMipmappedArrayCreate(ref CUmipmappedArray pHandle, ref CUDA_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc, uint numMipmapLevels);

			/// <summary>
			/// Destroys a CUDA mipmapped array.
			/// </summary>
			/// <param name="hMipmappedArray"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);

			/// <summary>
			/// Gets a mipmap level of a CUDA mipmapped array.
			/// </summary>
			/// <param name="pLevelArray"></param>
			/// <param name="hMipmappedArray"></param>
			/// <param name="level"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMipmappedArrayGetLevel(ref CUarray pLevelArray, CUmipmappedArray hMipmappedArray, uint level);

			// ----- Unified Addressing

			/// <summary>
			/// Advise about the usage of a given memory range.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="count"></param>
			/// <param name="advice"></param>
			/// <param name="device"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);

			/// <summary>
			/// Prefetches memory to the specified destination device.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="count"></param>
			/// <param name="dstDevice"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);

			/// <summary>
			/// Query an attribute of a given memory range.
			/// </summary>
			/// <param name="data"></param>
			/// <param name="dataSize"></param>
			/// <param name="attribute"></param>
			/// <param name="devPtr"></param>
			/// <param name="count"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemRangeGetAttribute(IntPtr data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);

			/// <summary>
			/// Query attributes of a given memory range.
			/// </summary>
			/// <param name="data"></param>
			/// <param name="dataSizes"></param>
			/// <param name="attributes"></param>
			/// <param name="numAttributes"></param>
			/// <param name="devPtr"></param>
			/// <param name="count"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuMemRangeGetAttributes(ref IntPtr data, ref size_t dataSizes, ref CUmem_range_attribute attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);

			/// <summary>
			/// Returns information about a pointer.
			/// </summary>
			/// <param name="data"></param>
			/// <param name="attribute"></param>
			/// <param name="ptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuPointerGetAttribute(ref IntPtr data, CUpointer_attribute attribute, CUdeviceptr ptr);

			/// <summary>
			/// Returns information about a pointer.
			/// </summary>
			/// <param name="numAttributes"></param>
			/// <param name="attributes"></param>
			/// <param name="data"></param>
			/// <param name="ptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuPointerGetAttributes(uint numAttributes, ref CUpointer_attribute attributes, IntPtr data, CUdeviceptr ptr);

			/// <summary>
			/// Set attributes on a previously allocated memory region.
			/// </summary>
			/// <param name="value"></param>
			/// <param name="attribute"></param>
			/// <param name="ptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuPointerSetAttribute(IntPtr value, CUpointer_attribute attribute, CUdeviceptr ptr);

			// ----- Stream Management

			/// <summary>
			/// Add a callback to a compute stream.
			/// </summary>
			/// <param name="hStream"></param>
			/// <param name="callback"></param>
			/// <param name="userData"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, IntPtr userData, uint flags);

			/// <summary>
			/// Attach memory to a stream asynchronously.
			/// </summary>
			/// <param name="hStream"></param>
			/// <param name="dptr"></param>
			/// <param name="length"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);

			/// <summary>
			/// Create a stream.
			/// </summary>
			/// <param name="phStream"></param>
			/// <param name="Flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamCreate(ref CUstream phStream, uint Flags);

			/// <summary>
			/// Create a stream with the given priority.
			/// </summary>
			/// <param name="phStream"></param>
			/// <param name="flags"></param>
			/// <param name="priority"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamCreateWithPriority(ref CUstream phStream, uint flags, int priority);

			/// <summary>
			/// Destroys a stream.
			/// </summary>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamDestroy(CUstream hStream);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus, cuuint64_t* id);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx);

			/// <summary>
			/// Query the flags of a given stream.
			/// </summary>
			/// <param name="hStream"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamGetFlags(CUstream hStream, ref uint flags);

			/// <summary>
			/// Query the priority of a given stream.
			/// </summary>
			/// <param name="hStream"></param>
			/// <param name="priority"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamGetPriority(CUstream hStream, ref int priority);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus);

			/// <summary>
			/// Determine status of a compute stream.
			/// </summary>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamQuery(CUstream hStream);

			/// <summary>
			/// Wait until a stream's tasks are completed.
			/// </summary>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamSynchronize(CUstream hStream);

			/// <summary>
			/// Make a compute stream wait on an event.
			/// </summary>
			/// <param name="hStream"></param>
			/// <param name="hEvent"></param>
			/// <param name="Flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, uint Flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode);

			// ----- Event Management

			/// <summary>
			/// Creates an event.
			/// </summary>
			/// <param name="phEvent"></param>
			/// <param name="Flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventCreate(ref CUevent phEvent, uint Flags);

			/// <summary>
			/// Destroys an event.
			/// </summary>
			/// <param name="hEvent"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventDestroy(CUevent hEvent);

			/// <summary>
			/// Computes the elapsed time between two events.
			/// </summary>
			/// <param name="pMilliseconds"></param>
			/// <param name="hStart"></param>
			/// <param name="hEnd"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventElapsedTime(ref float pMilliseconds, CUevent hStart, CUevent hEnd);

			/// <summary>
			/// Queries an event's status.
			/// </summary>
			/// <param name="hEvent"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventQuery(CUevent hEvent);

			/// <summary>
			/// Records an event.
			/// </summary>
			/// <param name="hEvent"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventRecord(CUevent hEvent, CUstream hStream);

			/// <summary>
			/// Waits for an event to complete.
			/// </summary>
			/// <param name="hEvent"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuEventSynchronize(CUevent hEvent);

			// ----- External Resource Interoperability

			// ----- Stream memory operations

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamBatchMemOp(CUstream stream, uint count, ref CUstreamBatchMemOpParams paramArray, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, uint flags);

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

			/// <summary>
			/// Sets the block-dimensions for the function.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="x"></param>
			/// <param name="y"></param>
			/// <param name="z"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);

			/// <summary>
			/// Sets the dynamic shared-memory size for the function.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="bytes"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuFuncSetSharedSize(CUfunction hfunc, uint bytes);

			/// <summary>
			/// Launches a CUDA function.
			/// </summary>
			/// <param name="f"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLaunch(CUfunction f);

			/// <summary>
			/// Launches a CUDA function.
			/// </summary>
			/// <param name="f"></param>
			/// <param name="grid_width"></param>
			/// <param name="grid_height"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);

			/// <summary>
			/// Launches a CUDA function.
			/// </summary>
			/// <param name="f"></param>
			/// <param name="grid_width"></param>
			/// <param name="grid_height"></param>
			/// <param name="hStream"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);

			/// <summary>
			/// Sets the parameter size for the function.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="numbytes"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetSize(CUfunction hfunc, uint numbytes);

			/// <summary>
			/// Adds a texture-reference to the function's argument list.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="texunit"></param>
			/// <param name="hTexRef"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);

			/// <summary>
			/// Adds a floating-point parameter to the function's argument list.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="offset"></param>
			/// <param name="value"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetf(CUfunction hfunc, int offset, float value);

			/// <summary>
			/// Adds an integer parameter to the function's argument list.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="offset"></param>
			/// <param name="value"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSeti(CUfunction hfunc, int offset, uint value);

			/// <summary>
			/// Adds arbitrary data to the function's argument list.
			/// </summary>
			/// <param name="hfunc"></param>
			/// <param name="offset"></param>
			/// <param name="ptr"></param>
			/// <param name="numbytes"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuParamSetv(CUfunction hfunc, int offset, IntPtr ptr, uint numbytes);

			// ----- Graph Management

			// ----- Occupancy

			/// <summary>
			/// Returns occupancy of a function.
			/// </summary>
			/// <param name="numBlocks"></param>
			/// <param name="func"></param>
			/// <param name="blockSize"></param>
			/// <param name="dynamicSMemSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(ref int numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);

			/// <summary>
			/// Returns occupancy of a function.
			/// </summary>
			/// <param name="numBlocks"></param>
			/// <param name="func"></param>
			/// <param name="blockSize"></param>
			/// <param name="dynamicSMemSize"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref int numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, uint flags);

			/// <summary>
			/// Suggest a launch configuration with reasonable occupancy.
			/// </summary>
			/// <param name="minGridSize"></param>
			/// <param name="blockSize"></param>
			/// <param name="func"></param>
			/// <param name="blockSizeToDynamicSMemSize"></param>
			/// <param name="dynamicSMemSize"></param>
			/// <param name="blockSizeLimit"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuOccupancyMaxPotentialBlockSize(ref int minGridSize, ref int blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);

			/// <summary>
			/// Suggest a launch configuration with reasonable occupancy.
			/// </summary>
			/// <param name="minGridSize"></param>
			/// <param name="blockSize"></param>
			/// <param name="func"></param>
			/// <param name="blockSizeToDynamicSMemSize"></param>
			/// <param name="dynamicSMemSize"></param>
			/// <param name="blockSizeLimit"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(ref int minGridSize, ref int blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, uint flags);

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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUresult cuProfilerInitialize(string configFile, string outputFile, CUoutput_mode outputMode);

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

		/// <summary>
		/// Gets the string representation of an error code enum name.
		/// </summary>
		/// <param name="error">Error code to convert to string.</param>
		/// <returns></returns>
		public static string GetErrorName(CUresult error) {
			IntPtr ptr = IntPtr.Zero;
			CheckStatus(API.cuGetErrorName(error, ref ptr));
			return Marshal.PtrToStringAnsi(ptr);
		}

		/// <summary>
		/// Gets the string description of an error code.
		/// </summary>
		/// <param name="error">Error code to convert to string.</param>
		/// <returns></returns>
		public static string GetErrorString(CUresult error) {
			IntPtr ptr = IntPtr.Zero;
			CheckStatus(API.cuGetErrorString(error, ref ptr));
			return Marshal.PtrToStringAnsi(ptr);
		}

		/// <summary>
		/// Initialize the CUDA driver API.
		/// </summary>
		/// <param name="Flags"></param>
		public static void Init(uint Flags) {
			CheckStatus(API.cuInit(Flags));
		}

		/// <summary>
		/// Returns the latest CUDA version supported by driver.
		/// </summary>
		/// <returns></returns>
		public static int DriverGetVersion() {
			int driverVersion = 0;
			CheckStatus(API.cuDriverGetVersion(ref driverVersion));
			return driverVersion;
		}

		/// <summary>
		/// Returns a handle to a compute device.
		/// </summary>
		/// <param name="ordinal"></param>
		/// <returns></returns>
		public static CUdevice DeviceGet(int ordinal) {
			CUdevice device = 0;
			CheckStatus(API.cuDeviceGet(ref device, ordinal));
			return device;
		}

		/// <summary>
		/// Returns information about the device.
		/// </summary>
		/// <param name="attrib"></param>
		/// <param name="dev"></param>
		/// <returns></returns>
		public static int DeviceGetAttribute(CUdevice_attribute attrib, CUdevice dev) {
			int pi = 0;
			CheckStatus(API.cuDeviceGetAttribute(ref pi, attrib, dev));
			return pi;
		}

		/// <summary>
		/// Returns the number of compute-capable devices.
		/// </summary>
		/// <returns></returns>
		public static int DeviceGetCount() {
			int count = 0;
			CheckStatus(API.cuDeviceGetCount(ref count));
			return count;
		}

		/// <summary>
		/// Returns an identifer string for the device.
		/// </summary>
		/// <param name="dev"></param>
		/// <returns></returns>
		public static string DeviceGetName(CUdevice dev) {
			StringBuilder name = new StringBuilder(256);
			CheckStatus(API.cuDeviceGetName(name, 256, dev));
			return name.ToString();
		}

		/// <summary>
		/// Returns the total amount of memory on the device.
		/// </summary>
		/// <param name="dev"></param>
		/// <returns></returns>
		public static size_t DeviceTotalMem(CUdevice dev) {
			size_t bytes = 0;
			CheckStatus(API.cuDeviceTotalMem(ref bytes, dev));
			return bytes;
		}

		/// <summary>
		/// Returns the compute capability of the device.
		/// </summary>
		/// <param name="dev"></param>
		/// <returns></returns>
		[Obsolete("This function was deprecated as of CUDA 5.0 and its functionality superceded by cuDeviceGetAttribute().")]
		public static string DeviceComputeCapability(CUdevice dev) {
			int major = 0;
			int minor = 0;
			CheckStatus(API.cuDeviceComputeCapability(ref major, ref minor, dev));
			return string.Format("{0}.{1}", major, minor);
		}

		/// <summary>
		/// Returns properties for a selected device.
		/// </summary>
		/// <param name="dev"></param>
		/// <returns></returns>
		[Obsolete("This function was deprecated as of CUDA 5.0 and replaced by cuDeviceGetAttribute().")]
		public static CUdevprop DeviceGetProperties(CUdevice dev) {
			CUdevprop prop = new CUdevprop();
			CheckStatus(API.cuDeviceGetProperties(ref prop, dev));
			return prop;
		}

		/// <summary>
		/// Get the state of the primary context.
		/// </summary>
		/// <param name="dev"></param>
		/// <returns></returns>
		public static uint[] DevicePrimaryCtxGetState(CUdevice dev) {
			uint flags = 0;
			int active = 0;
			CheckStatus(API.cuDevicePrimaryCtxGetState(dev, ref flags, ref active));
			return new uint[] { flags, (uint)active };
		}

		/// <summary>
		/// Release the primary context on the GPU.
		/// </summary>
		/// <param name="dev"></param>
		public static void DevicePrimaryCtxRelease(CUdevice dev) {
			CheckStatus(API.cuDevicePrimaryCtxRelease(dev));
		}

		/// <summary>
		/// Destroy all allocations and reset all state on the primary context.
		/// </summary>
		/// <param name="dev"></param>
		public static void DevicePrimaryCtxReset(CUdevice dev) {
			CheckStatus(API.cuDevicePrimaryCtxReset(dev));
		}

		/// <summary>
		/// Retain the primary context on the GPU.
		/// </summary>
		/// <param name="dev"></param>
		/// <returns></returns>
		public static CUcontext DevicePrimaryCtxRetain(CUdevice dev) {
			CUcontext pctx = IntPtr.Zero;
			CheckStatus(API.cuDevicePrimaryCtxRetain(ref pctx, dev));
			return pctx;
		}

		/// <summary>
		/// Set flags for the primary context.
		/// </summary>
		/// <param name="dev"></param>
		/// <param name="flags"></param>
		public static void DevicePrimaryCtxSetFlags(CUdevice dev, uint flags) {
			CheckStatus(API.cuDevicePrimaryCtxSetFlags(dev, flags));
		}

		/// <summary>
		/// Create a CUDA context.
		/// </summary>
		/// <param name="flags"></param>
		/// <param name="dev"></param>
		/// <returns></returns>
		public static CUcontext CtxCreate(uint flags, CUdevice dev) {
			CUcontext pctx = IntPtr.Zero;
			CheckStatus(API.cuCtxCreate(ref pctx, flags, dev));
			return pctx;
		}

		/// <summary>
		/// Destroy a CUDA context.
		/// </summary>
		/// <param name="ctx"></param>
		public static void CtxDestroy(CUcontext ctx) {
			CheckStatus(API.cuCtxDestroy(ctx));
		}

		/// <summary>
		/// Gets the context's API version.
		/// </summary>
		/// <param name="ctx"></param>
		/// <returns></returns>
		public static uint CtxGetApiVersion(CUcontext ctx) {
			uint version = 0;
			CheckStatus(API.cuCtxGetApiVersion(ctx, ref version));
			return version;
		}

		/// <summary>
		/// Returns the preferred cache configuration for the current context.
		/// </summary>
		/// <returns></returns>
		public static CUfunc_cache CtxGetCacheConfig() {
			CUfunc_cache pconfig = CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE;
			CheckStatus(API.cuCtxGetCacheConfig(ref pconfig));
			return pconfig;
		}

		/// <summary>
		/// Returns the CUDA context bound to the calling CPU thread.
		/// </summary>
		/// <returns></returns>
		public static CUcontext CtxGetCurrent() {
			CUcontext pctx = IntPtr.Zero;
			CheckStatus(API.cuCtxGetCurrent(ref pctx));
			return pctx;
		}

		/// <summary>
		/// Returns the device ID for the current context.
		/// </summary>
		/// <returns></returns>
		public static CUdevice CtxGetDevice() {
			CUdevice device = 0;
			CheckStatus(API.cuCtxGetDevice(ref device));
			return device;
		}

		/// <summary>
		/// Returns the flags for the current context.
		/// </summary>
		/// <returns></returns>
		public static uint CtxGetFlags() {
			uint flags = 0;
			CheckStatus(API.cuCtxGetFlags(ref flags));
			return flags;
		}

		/// <summary>
		/// Returns resource limits.
		/// </summary>
		/// <param name="limit"></param>
		/// <returns></returns>
		public static size_t CtxGetLimit(CUlimit limit) {
			size_t pvalue = 0;
			CheckStatus(API.cuCtxGetLimit(ref pvalue, limit));
			return pvalue;
		}

		/// <summary>
		/// Returns the current shared memory configuration for the current context.
		/// </summary>
		/// <param name="limit"></param>
		/// <returns></returns>
		public static CUsharedconfig CtxGetSharedMemConfig(CUlimit limit) {
			CUsharedconfig pConfig = CUsharedconfig.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
			CheckStatus(API.cuCtxGetSharedMemConfig(ref pConfig));
			return pConfig;
		}

		/// <summary>
		/// Returns numerical values that correspond to the least and greatest stream priorities.
		/// </summary>
		/// <returns></returns>
		public static int[] CtxGetStreamPriorityRange() {
			int leastPriority = 0;
			int greatestPriority = 0;
			CheckStatus(API.cuCtxGetStreamPriorityRange(ref leastPriority, ref greatestPriority));
			return new int[] { leastPriority, greatestPriority };
		}

		/// <summary>
		/// Pops the current CUDA context from the current CPU thread.
		/// </summary>
		/// <returns></returns>
		public static CUcontext CtxPopCurrent() {
			CUcontext pctx = IntPtr.Zero;
			CheckStatus(API.cuCtxPopCurrent(ref pctx));
			return pctx;
		}

		/// <summary>
		/// Pushes a context on the current CPU thread.
		/// </summary>
		/// <param name="ctx"></param>
		public static void CtxPushCurrent(CUcontext ctx) {
			CheckStatus(API.cuCtxPushCurrent(ctx));
		}

		/// <summary>
		/// Sets the preferred cache configuration for the current context.
		/// </summary>
		/// <param name="config"></param>
		public static void CtxSetCacheConfig(CUfunc_cache config) {
			CheckStatus(API.cuCtxSetCacheConfig(config));
		}

		/// <summary>
		/// Binds the specified CUDA context to the calling CPU thread.
		/// </summary>
		/// <param name="ctx"></param>
		public static void CtxSetCurrent(CUcontext ctx) {
			CheckStatus(API.cuCtxSetCurrent(ctx));
		}

		/// <summary>
		/// Set resource limits.
		/// </summary>
		/// <param name="limit"></param>
		/// <param name="value"></param>
		public static void CtxSetLimit(CUlimit limit, size_t value) {
			CheckStatus(API.cuCtxSetLimit(limit, value));
		}

		/// <summary>
		/// Sets the shared memory configuration for the current context.
		/// </summary>
		/// <param name="config"></param>
		public static void CtxSetSharedMemConfig(CUsharedconfig config) {
			CheckStatus(API.cuCtxSetSharedMemConfig(config));
		}

		/// <summary>
		/// Block for a context's tasks to complete.
		/// </summary>
		public static void CtxSynchronize() {
			CheckStatus(API.cuCtxSynchronize());
		}

		/// <summary>
		/// Increment a context's usage-count.
		/// </summary>
		/// <param name="flags"></param>
		/// <returns></returns>
		[Obsolete("this function is deprecated and should not be used")]
		public static CUcontext CtxAttach(uint flags) {
			CUcontext pctx = IntPtr.Zero;
			CheckStatus(API.cuCtxAttach(ref pctx, flags));
			return pctx;
		}

		/// <summary>
		/// Decrement a context's usage-count.
		/// </summary>
		/// <param name="ctx"></param>
		[Obsolete("this function is deprecated and should not be used")]
		public static void CtxDetach(CUcontext ctx) {
			CheckStatus(API.cuCtxDetach(ctx));
		}

		/// <summary>
		/// Add an input to a pending linker invocation.
		/// </summary>
		/// <param name="state"></param>
		/// <param name="type"></param>
		/// <param name="data"></param>
		/// <param name="size"></param>
		/// <param name="name"></param>
		/// <param name="numOptions"></param>
		/// <param name="options"></param>
		/// <param name="optionValues"></param>
		public static void LinkAddData(CUlinkState state, CUjitInputType type, IntPtr data, size_t size, string name, uint numOptions, CUjit_option options, IntPtr optionValues) {
			CheckStatus(API.cuLinkAddData(state, type, data, size, name, numOptions, ref options, ref optionValues));
		}

		/// <summary>
		/// Add a file input to a pending linker invocation.
		/// </summary>
		/// <param name="state"></param>
		/// <param name="type"></param>
		/// <param name="path"></param>
		/// <param name="numOptions"></param>
		/// <param name="options"></param>
		/// <param name="optionValues"></param>
		public static void LinkAddFile(CUlinkState state, CUjitInputType type, string path, uint numOptions, CUjit_option options, IntPtr optionValues) {
			CheckStatus(API.cuLinkAddFile(state, type, path, numOptions, ref options, ref optionValues));
		}

		/// <summary>
		/// Complete a pending linker invocation.
		/// </summary>
		/// <param name="state"></param>
		/// <param name="cubinOut"></param>
		/// <param name="sizeOut"></param>
		public static void LinkComplete(CUlinkState state, IntPtr cubinOut, size_t sizeOut) {
			CheckStatus(API.cuLinkComplete(state, cubinOut, ref sizeOut));
		}

		/// <summary>
		/// Creates a pending JIT linker invocation.
		/// </summary>
		/// <param name="numOptions"></param>
		/// <param name="options"></param>
		/// <param name="optionValues"></param>
		/// <param name="stateOut"></param>
		public static void LinkCreate(uint numOptions, CUjit_option options, IntPtr optionValues, CUlinkState stateOut) {
			CheckStatus(API.cuLinkCreate(numOptions, ref options, optionValues, ref stateOut));
		}

		/// <summary>
		/// Destroys state for a JIT linker invocation.
		/// </summary>
		/// <param name="state"></param>
		public static void LinkDestroy(CUlinkState state) {
			CheckStatus(API.cuLinkDestroy(state));
		}

		/// <summary>
		/// Returns a function handle.
		/// </summary>
		/// <param name="hmod"></param>
		/// <param name="name"></param>
		/// <returns></returns>
		public static CUfunction ModuleGetFunction(CUmodule hmod, string name) {
			CUfunction hfunc = IntPtr.Zero;
			CheckStatus(API.cuModuleGetFunction(ref hfunc, hmod, name));
			return hfunc;
		}

		/// <summary>
		/// Returns a global pointer from a module.
		/// </summary>
		/// <param name="hmod"></param>
		/// <param name="name"></param>
		/// <returns></returns>
		public static CUdeviceptr ModuleGetGlobal(CUmodule hmod, string name) {
			CUdeviceptr dptr = IntPtr.Zero;
			size_t bytes = 0;
			CheckStatus(API.cuModuleGetGlobal(ref dptr, ref bytes, hmod, name));
			return dptr;
		}

		/// <summary>
		/// Returns a handle to a surface reference.
		/// </summary>
		/// <param name="hmod"></param>
		/// <param name="name"></param>
		/// <returns></returns>
		public static CUsurfref ModuleGetSurfRef(CUmodule hmod, string name) {
			CUsurfref pSurfRef = IntPtr.Zero;
			CheckStatus(API.cuModuleGetSurfRef(ref pSurfRef, hmod, name));
			return pSurfRef;
		}

		/// <summary>
		/// Returns a handle to a texture reference.
		/// </summary>
		/// <param name="hmod"></param>
		/// <param name="name"></param>
		/// <returns></returns>
		public static CUtexref ModuleGetTexRef(CUmodule hmod, string name) {
			CUtexref pTexRef = IntPtr.Zero;
			CheckStatus(API.cuModuleGetTexRef(ref pTexRef, hmod, name));
			return pTexRef;
		}

		/// <summary>
		/// Loads a compute module.
		/// </summary>
		/// <param name="fname"></param>
		/// <returns></returns>
		public static CUmodule ModuleLoad(string fname) {
			CUmodule module = IntPtr.Zero;
			CheckStatus(API.cuModuleLoad(ref module, fname));
			return module;
		}

		/// <summary>
		/// Load a module's data.
		/// </summary>
		/// <param name="image"></param>
		/// <returns></returns>
		public static CUmodule ModuleLoadData(IntPtr image) {
			CUmodule module = IntPtr.Zero;
			CheckStatus(API.cuModuleLoadData(ref module, image));
			return module;
		}

		/// <summary>
		/// Load a module's data with options.
		/// </summary>
		/// <param name="image"></param>
		/// <param name="numOptions"></param>
		/// <param name="options"></param>
		/// <param name="optionValues"></param>
		/// <returns></returns>
		public static CUmodule ModuleLoadDataEx(IntPtr image, uint numOptions, CUjit_option options, IntPtr optionValues) {
			CUmodule module = IntPtr.Zero;
			CheckStatus(API.cuModuleLoadDataEx(ref module, image, numOptions, options, optionValues));
			return module;
		}

		/// <summary>
		/// Load a module's data.
		/// </summary>
		/// <param name="fatCubin"></param>
		/// <returns></returns>
		public static CUmodule ModuleLoadFatBinary(IntPtr fatCubin) {
			CUmodule module = IntPtr.Zero;
			CheckStatus(API.cuModuleLoadFatBinary(ref module, fatCubin));
			return module;
		}

		/// <summary>
		/// Unloads a module.
		/// </summary>
		/// <param name="hmod"></param>
		public static void ModuleUnload(CUmodule hmod) {
			CheckStatus(API.cuModuleUnload(hmod));
		}

		/// <summary>
		/// Creates a 3D CUDA array.
		/// </summary>
		/// <param name="pAllocateArray"></param>
		/// <returns></returns>
		public static CUarray Array3DCreate(CUDA_ARRAY3D_DESCRIPTOR pAllocateArray) {
			CUarray pHandle = IntPtr.Zero;
			CheckStatus(API.cuArray3DCreate(ref pHandle, ref pAllocateArray));
			return pHandle;
		}

		/// <summary>
		/// Get a 3D CUDA array descriptor.
		/// </summary>
		/// <param name="hArray"></param>
		/// <returns></returns>
		public static CUDA_ARRAY3D_DESCRIPTOR Array3DGetDescriptor(CUarray hArray) {
			CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor = new CUDA_ARRAY3D_DESCRIPTOR();
			CheckStatus(API.cuArray3DGetDescriptor(ref pArrayDescriptor, hArray));
			return pArrayDescriptor;
		}

		/// <summary>
		/// Creates a 1D or 2D CUDA array.
		/// </summary>
		/// <param name="pAllocateArray"></param>
		/// <returns></returns>
		public static CUarray ArrayCreate(CUDA_ARRAY_DESCRIPTOR pAllocateArray) {
			CUarray pHandle = IntPtr.Zero;
			CheckStatus(API.cuArrayCreate(ref pHandle, ref pAllocateArray));
			return pHandle;
		}

		/// <summary>
		/// Destroys a CUDA array.
		/// </summary>
		/// <param name="hArray"></param>
		public static void ArrayDestroy(CUarray hArray) {
			CheckStatus(API.cuArrayDestroy(hArray));
		}

		/// <summary>
		/// Get a 1D or 2D CUDA array descriptor.
		/// </summary>
		/// <param name="hArray"></param>
		/// <returns></returns>
		public static CUDA_ARRAY_DESCRIPTOR ArrayGetDescriptor(CUarray hArray) {
			CUDA_ARRAY_DESCRIPTOR pArrayDescriptor = new CUDA_ARRAY_DESCRIPTOR();
			CheckStatus(API.cuArrayGetDescriptor(ref pArrayDescriptor, hArray));
			return pArrayDescriptor;
		}

		/// <summary>
		/// Returns a handle to a compute device.
		/// </summary>
		/// <param name="pciBusId"></param>
		/// <returns></returns>
		public static CUdevice DeviceGetByPCIBusId(string pciBusId) {
			CUdevice dev = 0;
			CheckStatus(API.cuDeviceGetByPCIBusId(ref dev, pciBusId));
			return dev;
		}

		/// <summary>
		/// Returns a PCI Bus Id string for the device.
		/// </summary>
		/// <param name="dev"></param>
		/// <returns></returns>
		public static string DeviceGetPCIBusId(CUdevice dev) {
			StringBuilder pciBusId = new StringBuilder(256);
			CheckStatus(API.cuDeviceGetPCIBusId(pciBusId, 256, dev));
			return pciBusId.ToString();
		}

		/// <summary>
		/// Close memory mapped with cuIpcOpenMemHandle.
		/// </summary>
		/// <param name="dptr"></param>
		public static void IpcCloseMemHandle(CUdeviceptr dptr) {
			CheckStatus(API.cuIpcCloseMemHandle(dptr));
		}

		/// <summary>
		/// Gets an interprocess handle for a previously allocated event.
		/// </summary>
		/// <param name="cuEvent"></param>
		/// <returns></returns>
		public static CUipcEventHandle IpcGetEventHandle(CUevent cuEvent) {
			CUipcEventHandle pHandle = new CUipcEventHandle();
			CheckStatus(API.cuIpcGetEventHandle(ref pHandle, cuEvent));
			return pHandle;
		}

		/// <summary>
		/// Gets an interprocess memory handle for an existing device memory allocation.
		/// </summary>
		/// <param name="dptr"></param>
		/// <returns></returns>
		public static CUipcMemHandle IpcGetMemHandle(CUdeviceptr dptr) {
			CUipcMemHandle pHandle = new CUipcMemHandle();
			CheckStatus(API.cuIpcGetMemHandle(ref pHandle, dptr));
			return pHandle;
		}

		/// <summary>
		/// Opens an interprocess event handle for use in the current process.
		/// </summary>
		/// <param name="handle"></param>
		/// <returns></returns>
		public static CUevent IpcOpenEventHandle(CUipcEventHandle handle) {
			CUevent phEvent = IntPtr.Zero;
			CheckStatus(API.cuIpcOpenEventHandle(ref phEvent, handle));
			return phEvent;
		}

		/// <summary>
		/// Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="Flags"></param>
		/// <returns></returns>
		public static CUdeviceptr IpcOpenMemHandle(CUipcMemHandle handle, uint Flags) {
			CUdeviceptr pdptr = IntPtr.Zero;
			CheckStatus(API.cuIpcOpenMemHandle(ref pdptr, handle, Flags));
			return pdptr;
		}

		/// <summary>
		/// Allocates device memory.
		/// </summary>
		/// <param name="bytesize"></param>
		/// <returns></returns>
		public static CUdeviceptr MemAlloc(size_t bytesize) {
			CUdeviceptr dptr = IntPtr.Zero;
			CheckStatus(API.cuMemAlloc(ref dptr, bytesize));
			return dptr;
		}

		/// <summary>
		/// Allocates page-locked host memory.
		/// </summary>
		/// <param name="bytesize"></param>
		/// <returns></returns>
		public static IntPtr MemAllocHost(size_t bytesize) {
			IntPtr pp = IntPtr.Zero;
			CheckStatus(API.cuMemAllocHost(ref pp, bytesize));
			return pp;
		}

		/// <summary>
		/// Allocates memory that will be automatically managed by the Unified Memory system.
		/// </summary>
		/// <param name="bytesize"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static CUdeviceptr MemAllocManaged(size_t bytesize, uint flags) {
			CUdeviceptr dptr = IntPtr.Zero;
			CheckStatus(API.cuMemAllocManaged(ref dptr, bytesize, flags));
			return dptr;
		}

		/// <summary>
		/// Allocates pitched device memory.
		/// </summary>
		/// <param name="WidthInBytes"></param>
		/// <param name="Height"></param>
		/// <param name="ElementSizeBytes"></param>
		/// <returns></returns>
		public static CUdeviceptr MemAllocPitch(size_t WidthInBytes, size_t Height, uint ElementSizeBytes) {
			CUdeviceptr dptr = IntPtr.Zero;
			size_t pPitch = 0;
			CheckStatus(API.cuMemAllocPitch(ref dptr, ref pPitch, WidthInBytes, Height, ElementSizeBytes));
			return dptr;
		}

		/// <summary>
		/// Frees device memory.
		/// </summary>
		/// <param name="dptr"></param>
		public static void MemFree(CUdeviceptr dptr) {
			CheckStatus(API.cuMemFree(dptr));
		}

		/// <summary>
		/// Frees page-locked host memory.
		/// </summary>
		/// <param name="p"></param>
		public static void MemFreeHost(IntPtr p) {
			CheckStatus(API.cuMemFreeHost(p));
		}

		/// <summary>
		/// Get information on memory allocations.
		/// </summary>
		/// <param name="dptr"></param>
		/// <returns></returns>
		public static CUdeviceptr MemGetAddressRange(CUdeviceptr dptr) {
			CUdeviceptr pbase = IntPtr.Zero;
			size_t psize = 0;
			CheckStatus(API.cuMemGetAddressRange(ref pbase, ref psize, dptr));
			return pbase;
		}

		/// <summary>
		/// Gets free and total memory.
		/// </summary>
		/// <param name="p"></param>
		/// <returns></returns>
		public static size_t[] MemGetInfo(IntPtr p) {
			size_t free = 0;
			size_t total = 0;
			CheckStatus(API.cuMemGetInfo(ref free, ref total));
			return new size_t[] { free, total };
		}

		/// <summary>
		/// Allocates page-locked host memory.
		/// </summary>
		/// <param name="bytesize"></param>
		/// <param name="Flags"></param>
		/// <returns></returns>
		public static IntPtr MemHostAlloc(size_t bytesize, uint Flags) {
			IntPtr pp = IntPtr.Zero;
			CheckStatus(API.cuMemHostAlloc(ref pp, bytesize, Flags));
			return pp;
		}

		/// <summary>
		/// Passes back device pointer of mapped pinned memory.
		/// </summary>
		/// <param name="p"></param>
		/// <param name="Flags"></param>
		/// <returns></returns>
		public static CUdeviceptr MemHostGetDevicePointer(IntPtr p, uint Flags) {
			CUdeviceptr pdptr = IntPtr.Zero;
			CheckStatus(API.cuMemHostGetDevicePointer(ref pdptr, p, Flags));
			return pdptr;
		}

		/// <summary>
		/// Passes back flags that were used for a pinned allocation.
		/// </summary>
		/// <param name="p"></param>
		/// <returns></returns>
		public static uint MemHostGetFlags(IntPtr p) {
			uint pFlags = 0;
			CheckStatus(API.cuMemHostGetFlags(ref pFlags, p));
			return pFlags;
		}

		/// <summary>
		/// Registers an existing host memory range for use by CUDA.
		/// </summary>
		/// <param name="p"></param>
		/// <param name="bytesize"></param>
		/// <param name="Flags"></param>
		public static void MemHostRegister(IntPtr p, size_t bytesize, uint Flags) {
			CheckStatus(API.cuMemHostRegister(p, bytesize, Flags));
		}

		/// <summary>
		/// Unregisters a memory range that was registered with cuMemHostRegister.
		/// </summary>
		/// <param name="p"></param>
		public static void MemHostUnregister(IntPtr p) {
			CheckStatus(API.cuMemHostUnregister(p));
		}

		/// <summary>
		/// Copies memory.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		/// <param name="ByteCount"></param>
		public static void Memcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
			CheckStatus(API.cuMemcpy(dst, src, ByteCount));
		}

		/// <summary>
		/// Copies memory for 2D arrays.
		/// </summary>
		/// <param name="pCopy"></param>
		public static void Memcpy2D(CUDA_MEMCPY2D pCopy) {
			CheckStatus(API.cuMemcpy2D(ref pCopy));
		}

		/// <summary>
		/// Copies memory for 2D arrays.
		/// </summary>
		/// <param name="pCopy"></param>
		/// <param name="hStream"></param>
		public static void Memcpy2DAsync(CUDA_MEMCPY2D pCopy, CUstream hStream) {
			CheckStatus(API.cuMemcpy2DAsync(ref pCopy, hStream));
		}

		/// <summary>
		/// Copies memory for 2D arrays.
		/// </summary>
		/// <param name="pCopy"></param>
		public static void Memcpy2DUnaligned(CUDA_MEMCPY2D pCopy) {
			CheckStatus(API.cuMemcpy2DUnaligned(ref pCopy));
		}

		/// <summary>
		/// Copies memory for 3D arrays.
		/// </summary>
		/// <param name="pCopy"></param>
		public static void Memcpy3D(CUDA_MEMCPY3D pCopy) {
			CheckStatus(API.cuMemcpy3D(ref pCopy));
		}

		/// <summary>
		/// Copies memory for 3D arrays.
		/// </summary>
		/// <param name="pCopy"></param>
		/// <param name="hStream"></param>
		public static void Memcpy3DAsync(CUDA_MEMCPY3D pCopy, CUstream hStream) {
			CheckStatus(API.cuMemcpy3DAsync(ref pCopy, hStream));
		}

		/// <summary>
		/// Copies memory between contexts.
		/// </summary>
		/// <param name="pCopy"></param>
		public static void Memcpy3DPeer(CUDA_MEMCPY3D_PEER pCopy) {
			CheckStatus(API.cuMemcpy3DPeer(ref pCopy));
		}

		/// <summary>
		/// Copies memory between contexts asynchronously.
		/// </summary>
		/// <param name="pCopy"></param>
		/// <param name="hStream"></param>
		public static void Memcpy3DPeerAsync(CUDA_MEMCPY3D_PEER pCopy, CUstream hStream) {
			CheckStatus(API.cuMemcpy3DPeerAsync(ref pCopy, hStream));
		}

		/// <summary>
		/// Copies memory asynchronously.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		/// <param name="ByteCount"></param>
		/// <param name="hStream"></param>
		public static void MemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
			CheckStatus(API.cuMemcpyAsync(dst, src, ByteCount, hStream));
		}

		/// <summary>
		/// Copies memory from Array to Array.
		/// </summary>
		/// <param name="dstArray"></param>
		/// <param name="dstOffset"></param>
		/// <param name="srcArray"></param>
		/// <param name="srcOffset"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
			CheckStatus(API.cuMemcpyAtoA(dstArray, dstOffset, srcArray, srcOffset, ByteCount));
		}

		/// <summary>
		/// Copies memory from Array to Device.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="srcArray"></param>
		/// <param name="srcOffset"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
			CheckStatus(API.cuMemcpyAtoD(dstDevice, srcArray, srcOffset, ByteCount));
		}

		/// <summary>
		/// Copies memory from Array to Host.
		/// </summary>
		/// <param name="dstHost"></param>
		/// <param name="srcArray"></param>
		/// <param name="srcOffset"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyAtoH(IntPtr dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
			CheckStatus(API.cuMemcpyAtoH(dstHost, srcArray, srcOffset, ByteCount));
		}

		/// <summary>
		/// Copies memory from Array to Host.
		/// </summary>
		/// <param name="dstHost"></param>
		/// <param name="srcArray"></param>
		/// <param name="srcOffset"></param>
		/// <param name="ByteCount"></param>
		/// <param name="hStream"></param>
		public static void MemcpyAtoHAsync(IntPtr dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
			CheckStatus(API.cuMemcpyAtoHAsync(dstHost, srcArray, srcOffset, ByteCount, hStream));
		}

		/// <summary>
		/// Copies memory from Device to Array.
		/// </summary>
		/// <param name="dstArray"></param>
		/// <param name="dstOffset"></param>
		/// <param name="srcDevice"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) {
			CheckStatus(API.cuMemcpyDtoA(dstArray, dstOffset, srcDevice, ByteCount));
		}

		/// <summary>
		/// Copies memory from Device to Device.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="srcDevice"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
			CheckStatus(API.cuMemcpyDtoD(dstDevice, srcDevice, ByteCount));
		}

		/// <summary>
		/// Copies memory from Device to Device.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="srcDevice"></param>
		/// <param name="ByteCount"></param>
		/// <param name="hStream"></param>
		public static void MemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
			CheckStatus(API.cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream));
		}

		/// <summary>
		/// Copies memory from Device to Host.
		/// </summary>
		/// <param name="dstHost"></param>
		/// <param name="srcDevice"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
			CheckStatus(API.cuMemcpyDtoH(dstHost, srcDevice, ByteCount));
		}

		/// <summary>
		/// Copies memory from Device to Host.
		/// </summary>
		/// <param name="dstHost"></param>
		/// <param name="srcDevice"></param>
		/// <param name="ByteCount"></param>
		/// <param name="hStream"></param>
		public static void MemcpyDtoHAsync(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
			CheckStatus(API.cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream));
		}

		/// <summary>
		/// Copies memory from Host to Array.
		/// </summary>
		/// <param name="dstArray"></param>
		/// <param name="dstOffset"></param>
		/// <param name="srcHost"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyHtoA(CUarray dstArray, size_t dstOffset, IntPtr srcHost, size_t ByteCount) {
			CheckStatus(API.cuMemcpyHtoA(dstArray, dstOffset, srcHost, ByteCount));
		}

		/// <summary>
		/// Copies memory from Host to Array.
		/// </summary>
		/// <param name="dstArray"></param>
		/// <param name="dstOffset"></param>
		/// <param name="srcHost"></param>
		/// <param name="ByteCount"></param>
		/// <param name="hStream"></param>
		public static void MemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, IntPtr srcHost, size_t ByteCount, CUstream hStream) {
			CheckStatus(API.cuMemcpyHtoAAsync(dstArray, dstOffset, srcHost, ByteCount, hStream));
		}

		/// <summary>
		/// Copies memory from Host to Device.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="srcHost"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, size_t ByteCount) {
			CheckStatus(API.cuMemcpyHtoD(dstDevice, srcHost, ByteCount));
		}

		/// <summary>
		/// Copies memory from Host to Device.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="srcHost"></param>
		/// <param name="ByteCount"></param>
		/// <param name="hStream"></param>
		public static void MemcpyHtoDAsync(CUdeviceptr dstDevice, IntPtr srcHost, size_t ByteCount, CUstream hStream) {
			CheckStatus(API.cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream));
		}

		/// <summary>
		/// Copies device memory between two contexts.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstContext"></param>
		/// <param name="srcDevice"></param>
		/// <param name="srcContext"></param>
		/// <param name="ByteCount"></param>
		public static void MemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
			CheckStatus(API.cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount));
		}

		/// <summary>
		/// Copies device memory between two contexts asynchronously.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstContext"></param>
		/// <param name="srcDevice"></param>
		/// <param name="srcContext"></param>
		/// <param name="ByteCount"></param>
		/// <param name="hStream"></param>
		public static void MemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
			CheckStatus(API.cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream));
		}

		/// <summary>
		/// Initializes device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="us"></param>
		/// <param name="N"></param>
		public static void MemsetD16(CUdeviceptr dstDevice, ushort us, size_t N) {
			CheckStatus(API.cuMemsetD16(dstDevice, us, N));
		}

		/// <summary>
		/// Sets device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="us"></param>
		/// <param name="N"></param>
		/// <param name="hStream"></param>
		public static void MemsetD16Async(CUdeviceptr dstDevice, ushort us, size_t N, CUstream hStream) {
			CheckStatus(API.cuMemsetD16Async(dstDevice, us, N, hStream));
		}

		/// <summary>
		/// Initializes device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstPitch"></param>
		/// <param name="us"></param>
		/// <param name="Width"></param>
		/// <param name="Height"></param>
		public static void MemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height) {
			CheckStatus(API.cuMemsetD2D16(dstDevice, dstPitch, us, Width, Height));
		}

		/// <summary>
		/// Sets device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstPitch"></param>
		/// <param name="us"></param>
		/// <param name="Width"></param>
		/// <param name="Height"></param>
		/// <param name="hStream"></param>
		public static void MemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height, CUstream hStream) {
			CheckStatus(API.cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream));
		}

		/// <summary>
		/// Initializes device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstPitch"></param>
		/// <param name="ui"></param>
		/// <param name="Width"></param>
		/// <param name="Height"></param>
		public static void MemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height) {
			CheckStatus(API.cuMemsetD2D32(dstDevice, dstPitch, ui, Width, Height));
		}

		/// <summary>
		/// Sets device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstPitch"></param>
		/// <param name="ui"></param>
		/// <param name="Width"></param>
		/// <param name="Height"></param>
		/// <param name="hStream"></param>
		public static void MemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height, CUstream hStream) {
			CheckStatus(API.cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream));
		}

		/// <summary>
		/// Initializes device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstPitch"></param>
		/// <param name="uc"></param>
		/// <param name="Width"></param>
		/// <param name="Height"></param>
		public static void MemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, byte uc, size_t Width, size_t Height) {
			CheckStatus(API.cuMemsetD2D8(dstDevice, dstPitch, uc, Width, Height));
		}

		/// <summary>
		/// Sets device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="dstPitch"></param>
		/// <param name="uc"></param>
		/// <param name="Width"></param>
		/// <param name="Height"></param>
		/// <param name="hStream"></param>
		public static void MemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, byte uc, size_t Width, size_t Height, CUstream hStream) {
			CheckStatus(API.cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream));
		}

		/// <summary>
		/// Initializes device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="ui"></param>
		/// <param name="N"></param>
		public static void MemsetD32(CUdeviceptr dstDevice, uint ui, size_t N) {
			CheckStatus(API.cuMemsetD32(dstDevice, ui, N));
		}

		/// <summary>
		/// Sets device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="ui"></param>
		/// <param name="N"></param>
		/// <param name="hStream"></param>
		public static void MemsetD32Async(CUdeviceptr dstDevice, uint ui, size_t N, CUstream hStream) {
			CheckStatus(API.cuMemsetD32Async(dstDevice, ui, N, hStream));
		}

		/// <summary>
		/// Initializes device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="uc"></param>
		/// <param name="N"></param>
		public static void MemsetD8(CUdeviceptr dstDevice, byte uc, size_t N) {
			CheckStatus(API.cuMemsetD8(dstDevice, uc, N));
		}

		/// <summary>
		/// Sets device memory.
		/// </summary>
		/// <param name="dstDevice"></param>
		/// <param name="uc"></param>
		/// <param name="N"></param>
		/// <param name="hStream"></param>
		public static void MemsetD8Async(CUdeviceptr dstDevice, byte uc, size_t N, CUstream hStream) {
			CheckStatus(API.cuMemsetD8Async(dstDevice, uc, N, hStream));
		}

		/// <summary>
		/// Creates a CUDA mipmapped array.
		/// </summary>
		/// <param name="pMipmappedArrayDesc"></param>
		/// <param name="numMipmapLevels"></param>
		/// <returns></returns>
		public static CUmipmappedArray MipmappedArrayCreate(CUDA_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc, uint numMipmapLevels) {
			CUmipmappedArray pHandle = IntPtr.Zero;
			CheckStatus(API.cuMipmappedArrayCreate(ref pHandle, ref pMipmappedArrayDesc, numMipmapLevels));
			return pHandle;
		}

		/// <summary>
		/// Destroys a CUDA mipmapped array.
		/// </summary>
		/// <param name="hMipmappedArray"></param>
		public static void MipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
			CheckStatus(API.cuMipmappedArrayDestroy(hMipmappedArray));
		}

		/// <summary>
		/// Gets a mipmap level of a CUDA mipmapped array.
		/// </summary>
		/// <param name="hMipmappedArray"></param>
		/// <param name="level"></param>
		/// <returns></returns>
		public static CUarray MipmappedArrayGetLevel(CUmipmappedArray hMipmappedArray, uint level) {
			CUarray pLevelArray = IntPtr.Zero;
			CheckStatus(API.cuMipmappedArrayGetLevel(ref pLevelArray, hMipmappedArray, level));
			return pLevelArray;
		}

		/// <summary>
		/// Advise about the usage of a given memory range.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="count"></param>
		/// <param name="advice"></param>
		/// <param name="device"></param>
		public static void MemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) {
			CheckStatus(API.cuMemAdvise(devPtr, count, advice, device));
		}

		/// <summary>
		/// Prefetches memory to the specified destination device.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="count"></param>
		/// <param name="dstDevice"></param>
		/// <param name="hStream"></param>
		public static void MemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
			CheckStatus(API.cuMemPrefetchAsync(devPtr, count, dstDevice, hStream));
		}

		/// <summary>
		/// Query an attribute of a given memory range.
		/// </summary>
		/// <param name="data"></param>
		/// <param name="dataSize"></param>
		/// <param name="attribute"></param>
		/// <param name="devPtr"></param>
		/// <param name="count"></param>
		public static void MemRangeGetAttribute(IntPtr data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
			CheckStatus(API.cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count));
		}

		//public static void MemRangeGetAttributes(IntPtr data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
		//	CheckStatus(API.cuMemRangeGetAttributes(data, dataSize, attribute, devPtr, count));
		//}

		/// <summary>
		/// Returns information about a pointer.
		/// </summary>
		/// <param name="attribute"></param>
		/// <param name="ptr"></param>
		/// <returns></returns>
		public static IntPtr PointerGetAttribute(CUpointer_attribute attribute, CUdeviceptr ptr) {
			IntPtr data = IntPtr.Zero;
			CheckStatus(API.cuPointerGetAttribute(ref data, attribute, ptr));
			return data;
		}

		/// <summary>
		/// Returns information about a pointer.
		/// </summary>
		/// <param name="numAttributes"></param>
		/// <param name="attributes"></param>
		/// <param name="data"></param>
		/// <param name="ptr"></param>
		public static void PointerGetAttributes(uint numAttributes, CUpointer_attribute attributes, IntPtr data, CUdeviceptr ptr) {
			CheckStatus(API.cuPointerGetAttributes(numAttributes, ref attributes, data, ptr));
		}

		/// <summary>
		/// Set attributes on a previously allocated memory region.
		/// </summary>
		/// <param name="value"></param>
		/// <param name="attribute"></param>
		/// <param name="ptr"></param>
		public static void PointerSetAttribute(IntPtr value, CUpointer_attribute attribute, CUdeviceptr ptr) {
			CheckStatus(API.cuPointerSetAttribute(value, attribute, ptr));
		}

		/// <summary>
		/// Add a callback to a compute stream.
		/// </summary>
		/// <param name="hStream"></param>
		/// <param name="callback"></param>
		/// <param name="userData"></param>
		/// <param name="flags"></param>
		public static void StreamAddCallback(CUstream hStream, CUstreamCallback callback, IntPtr userData, uint flags) {
			CheckStatus(API.cuStreamAddCallback(hStream, callback, userData, flags));
		}

		/// <summary>
		/// Attach memory to a stream asynchronously.
		/// </summary>
		/// <param name="hStream"></param>
		/// <param name="dptr"></param>
		/// <param name="length"></param>
		/// <param name="flags"></param>
		public static void StreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, uint flags) {
			CheckStatus(API.cuStreamAttachMemAsync(hStream, dptr, length, flags));
		}

		/// <summary>
		/// Create a stream.
		/// </summary>
		/// <param name="Flags"></param>
		/// <returns></returns>
		public static CUstream StreamCreate(uint Flags) {
			CUstream phStream = IntPtr.Zero;
			CheckStatus(API.cuStreamCreate(ref phStream, Flags));
			return phStream;
		}

		/// <summary>
		/// Create a stream with the given priority.
		/// </summary>
		/// <param name="flags"></param>
		/// <param name="priority"></param>
		/// <returns></returns>
		public static CUstream StreamCreateWithPriority(uint flags, int priority) {
			CUstream phStream = IntPtr.Zero;
			CheckStatus(API.cuStreamCreateWithPriority(ref phStream, flags, priority));
			return phStream;
		}

		/// <summary>
		/// Destroys a stream.
		/// </summary>
		/// <param name="hStream"></param>
		public static void StreamDestroy(CUstream hStream) {
			CheckStatus(API.cuStreamDestroy(hStream));
		}

		/// <summary>
		/// Query the flags of a given stream.
		/// </summary>
		/// <param name="hStream"></param>
		/// <returns></returns>
		public static uint StreamGetFlags(CUstream hStream) {
			uint flags = 0;
			CheckStatus(API.cuStreamGetFlags(hStream, ref flags));
			return flags;
		}

		/// <summary>
		/// Query the priority of a given stream.
		/// </summary>
		/// <param name="hStream"></param>
		/// <returns></returns>
		public static int StreamGetPriority(CUstream hStream) {
			int priority = 0;
			CheckStatus(API.cuStreamGetPriority(hStream, ref priority));
			return priority;
		}

		/// <summary>
		/// Determine status of a compute stream.
		/// </summary>
		/// <param name="hStream"></param>
		/// <returns></returns>
		public static CUresult StreamQuery(CUstream hStream) {
			return API.cuStreamQuery(hStream);
		}

		/// <summary>
		/// Wait until a stream's tasks are completed.
		/// </summary>
		/// <param name="hStream"></param>
		public static void StreamSynchronize(CUstream hStream) {
			CheckStatus(API.cuStreamSynchronize(hStream));
		}

		/// <summary>
		/// Make a compute stream wait on an event.
		/// </summary>
		/// <param name="hStream"></param>
		/// <param name="hEvent"></param>
		/// <param name="Flags"></param>
		public static void StreamWaitEvent(CUstream hStream, CUevent hEvent, uint Flags) {
			CheckStatus(API.cuStreamWaitEvent(hStream, hEvent, Flags));
		}

		/// <summary>
		/// Creates an event.
		/// </summary>
		/// <param name="Flags"></param>
		/// <returns></returns>
		public static CUevent EventCreate(uint Flags) {
			CUevent phEvent = IntPtr.Zero;
			CheckStatus(API.cuEventCreate(ref phEvent, Flags));
			return phEvent;
		}

		/// <summary>
		/// Destroys an event.
		/// </summary>
		/// <param name="hEvent"></param>
		public static void EventDestroy(CUevent hEvent) {
			CheckStatus(API.cuEventDestroy(hEvent));
		}

		/// <summary>
		/// Computes the elapsed time between two events.
		/// </summary>
		/// <param name="hStart"></param>
		/// <param name="hEnd"></param>
		/// <returns></returns>
		public static float EventElapsedTime(CUevent hStart, CUevent hEnd) {
			float pMilliseconds = 0f;
			CheckStatus(API.cuEventElapsedTime(ref pMilliseconds, hStart, hEnd));
			return pMilliseconds;
		}

		/// <summary>
		/// Queries an event's status.
		/// </summary>
		/// <param name="hEvent"></param>
		/// <returns></returns>
		public static CUresult EventQuery(CUevent hEvent) {
			return API.cuEventQuery(hEvent);
		}

		/// <summary>
		/// Records an event.
		/// </summary>
		/// <param name="hEvent"></param>
		/// <param name="hStream"></param>
		public static void EventRecord(CUevent hEvent, CUstream hStream) {
			CheckStatus(API.cuEventRecord(hEvent, hStream));
		}

		/// <summary>
		/// Waits for an event to complete.
		/// </summary>
		/// <param name="hEvent"></param>
		public static void EventSynchronize(CUevent hEvent) {
			CheckStatus(API.cuEventSynchronize(hEvent));
		}

		public static void StreamBatchMemOp(CUstream stream, uint count, CUstreamBatchMemOpParams paramArray, uint flags) {
			CheckStatus(API.cuStreamBatchMemOp(stream, count, ref paramArray, flags));
		}

		public static void StreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, uint flags) {
			CheckStatus(API.cuStreamWaitValue32(stream, addr, value, flags));
		}

		public static void StreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, uint flags) {
			CheckStatus(API.cuStreamWriteValue32(stream, addr, value, flags));
		}

		public static int FuncGetAttribute(CUfunction_attribute attrib, CUfunction hfunc) {
			int pi = 0;
			CheckStatus(API.cuFuncGetAttribute(ref pi, attrib, hfunc));
			return pi;
		}

		public static void FuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
			CheckStatus(API.cuFuncSetCacheConfig(hfunc, config));
		}

		public static void FuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
			CheckStatus(API.cuFuncSetSharedMemConfig(hfunc, config));
		}

		public static void LaunchKernel(CUfunction f, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, CUstream hStream, IntPtr kernelParams, IntPtr extra) {
			CheckStatus(API.cuLaunchKernel(
				f,
				gridDimX, gridDimY, gridDimZ,
				blockDimX, blockDimY, blockDimZ,
				sharedMemBytes, hStream,
				kernelParams, extra
			));
		}

		/// <summary>
		/// Sets the block-dimensions for the function.
		/// </summary>
		/// <param name="hfunc"></param>
		/// <param name="x"></param>
		/// <param name="y"></param>
		/// <param name="z"></param>
		[Obsolete]
		public static void FuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
			CheckStatus(API.cuFuncSetBlockShape(hfunc, x, y, z));
		}

		/// <summary>
		/// Sets the dynamic shared-memory size for the function.
		/// </summary>
		/// <param name="hfunc"></param>
		/// <param name="bytes"></param>
		[Obsolete]
		public static void FuncSetSharedSize(CUfunction hfunc, uint bytes) {
			CheckStatus(API.cuFuncSetSharedSize(hfunc, bytes));
		}

		/// <summary>
		/// Launches a CUDA function.
		/// </summary>
		/// <param name="f"></param>
		[Obsolete]
		public static void Launch(CUfunction f) {
			CheckStatus(API.cuLaunch(f));
		}

		/// <summary>
		/// Launches a CUDA function.
		/// </summary>
		/// <param name="f"></param>
		/// <param name="grid_width"></param>
		/// <param name="grid_height"></param>
		[Obsolete]
		public static void LaunchGrid(CUfunction f, int grid_width, int grid_height) {
			CheckStatus(API.cuLaunchGrid(f, grid_width, grid_height));
		}

		/// <summary>
		/// Launches a CUDA function.
		/// </summary>
		/// <param name="f"></param>
		/// <param name="grid_width"></param>
		/// <param name="grid_height"></param>
		/// <param name="hStream"></param>
		[Obsolete]
		public static void LaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) {
			CheckStatus(API.cuLaunchGridAsync(f, grid_width, grid_height, hStream));
		}

		/// <summary>
		/// Sets the parameter size for the function.
		/// </summary>
		/// <param name="hfunc"></param>
		/// <param name="numbytes"></param>
		[Obsolete]
		public static void ParamSetSize(CUfunction hfunc, uint numbytes) {
			CheckStatus(API.cuParamSetSize(hfunc, numbytes));
		}

		/// <summary>
		/// Adds a texture-reference to the function's argument list.
		/// </summary>
		/// <param name="hfunc"></param>
		/// <param name="texunit"></param>
		/// <param name="hTexRef"></param>
		[Obsolete]
		public static void ParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
			CheckStatus(API.cuParamSetTexRef(hfunc, texunit, hTexRef));
		}

		/// <summary>
		/// Adds a floating-point parameter to the function's argument list.
		/// </summary>
		/// <param name="hfunc"></param>
		/// <param name="offset"></param>
		/// <param name="value"></param>
		[Obsolete]
		public static void ParamSetf(CUfunction hfunc, int offset, float value) {
			CheckStatus(API.cuParamSetf(hfunc, offset, value));
		}

		/// <summary>
		/// Adds an integer parameter to the function's argument list.
		/// </summary>
		/// <param name="hfunc"></param>
		/// <param name="offset"></param>
		/// <param name="value"></param>
		[Obsolete]
		public static void ParamSeti(CUfunction hfunc, int offset, uint value) {
			CheckStatus(API.cuParamSeti(hfunc, offset, value));
		}

		/// <summary>
		/// Adds arbitrary data to the function's argument list.
		/// </summary>
		/// <param name="hfunc"></param>
		/// <param name="offset"></param>
		/// <param name="ptr"></param>
		/// <param name="numbytes"></param>
		[Obsolete]
		public static void ParamSetv(CUfunction hfunc, int offset, IntPtr ptr, uint numbytes) {
			CheckStatus(API.cuParamSetv(hfunc, offset, ptr, numbytes));
		}

		/// <summary>
		/// Returns occupancy of a function.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="blockSize"></param>
		/// <param name="dynamicSMemSize"></param>
		/// <returns></returns>
		public static int OccupancyMaxActiveBlocksPerMultiprocessor(CUfunction func, int blockSize, size_t dynamicSMemSize) {
			int numBlocks = 0;
			CheckStatus(API.cuOccupancyMaxActiveBlocksPerMultiprocessor(ref numBlocks, func, blockSize, dynamicSMemSize));
			return numBlocks;
		}

		/// <summary>
		/// Returns occupancy of a function.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="blockSize"></param>
		/// <param name="dynamicSMemSize"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static int OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(CUfunction func, int blockSize, size_t dynamicSMemSize, uint flags) {
			int numBlocks = 0;
			CheckStatus(API.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref numBlocks, func, blockSize, dynamicSMemSize, flags));
			return numBlocks;
		}

		/// <summary>
		/// Suggest a launch configuration with reasonable occupancy.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="blockSizeToDynamicSMemSize"></param>
		/// <param name="dynamicSMemSize"></param>
		/// <param name="blockSizeLimit"></param>
		public static void OccupancyMaxPotentialBlockSize(CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) {
			int minGridSize = 0;
			int blockSize = 0;
			CheckStatus(API.cuOccupancyMaxPotentialBlockSize(ref minGridSize, ref blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit));
		}

		/// <summary>
		/// Suggest a launch configuration with reasonable occupancy.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="blockSizeToDynamicSMemSize"></param>
		/// <param name="dynamicSMemSize"></param>
		/// <param name="blockSizeLimit"></param>
		/// <param name="flags"></param>
		public static void OccupancyMaxPotentialBlockSizeWithFlags(CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, uint flags) {
			int minGridSize = 0;
			int blockSize = 0;
			CheckStatus(API.cuOccupancyMaxPotentialBlockSizeWithFlags(ref minGridSize, ref blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags));
		}

		public static CUdeviceptr TexRefGetAddress(CUtexref hTexRef) {
			CUdeviceptr pdptr = IntPtr.Zero;
			CheckStatus(API.cuTexRefGetAddress(ref pdptr, hTexRef));
			return pdptr;
		}

		public static CUaddress_mode TexRefGetAddressMode(CUtexref hTexRef, int dim) {
			CUaddress_mode pam = CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP;
			CheckStatus(API.cuTexRefGetAddressMode(ref pam, hTexRef, dim));
			return pam;
		}

		public static CUarray TexRefGetArray(CUtexref hTexRef) {
			CUarray phArray = IntPtr.Zero;
			CheckStatus(API.cuTexRefGetArray(ref phArray, hTexRef));
			return phArray;
		}

		public static float TexRefGetBorderColor(CUtexref hTexRef) {
			float pBorderColor = 0f;
			CheckStatus(API.cuTexRefGetBorderColor(ref pBorderColor, hTexRef));
			return pBorderColor;
		}

		public static CUfilter_mode TexRefGetFilterMode(CUtexref hTexRef) {
			CUfilter_mode pfm = CUfilter_mode.CU_TR_FILTER_MODE_POINT;
			CheckStatus(API.cuTexRefGetFilterMode(ref pfm, hTexRef));
			return pfm;
		}

		public static uint TexRefGetFlags(CUtexref hTexRef) {
			uint pFlags = 0;
			CheckStatus(API.cuTexRefGetFlags(ref pFlags, hTexRef));
			return pFlags;
		}

		public static CUarray_format TexRefGetFormat(CUtexref hTexRef) {
			CUarray_format pFormat = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
			int pNumChannels = 0;
			CheckStatus(API.cuTexRefGetFormat(ref pFormat, ref pNumChannels, hTexRef));
			return pFormat;
		}

		public static int TexRefGetMaxAnisotropy(CUtexref hTexRef) {
			int pmaxAniso = 0;
			CheckStatus(API.cuTexRefGetMaxAnisotropy(ref pmaxAniso, hTexRef));
			return pmaxAniso;
		}

		public static CUfilter_mode TexRefGetMipmapFilterMode(CUtexref hTexRef) {
			CUfilter_mode pfm = CUfilter_mode.CU_TR_FILTER_MODE_POINT;
			CheckStatus(API.cuTexRefGetMipmapFilterMode(ref pfm, hTexRef));
			return pfm;
		}

		public static float TexRefGetMipmapLevelBias(CUtexref hTexRef) {
			float pbias = 0f;
			CheckStatus(API.cuTexRefGetMipmapLevelBias(ref pbias, hTexRef));
			return pbias;
		}

		public static float[] TexRefGetMipmapLevelClamp(CUtexref hTexRef) {
			float pminMipmapLevelClamp = 0f;
			float pmaxMipmapLevelClamp = 0f;
			CheckStatus(API.cuTexRefGetMipmapLevelClamp(ref pminMipmapLevelClamp, ref pmaxMipmapLevelClamp, hTexRef));
			return new float[] { pminMipmapLevelClamp, pmaxMipmapLevelClamp };
		}

		public static CUmipmappedArray TexRefGetMipmappedArray(CUtexref hTexRef) {
			CUmipmappedArray phMipmappedArray = IntPtr.Zero;
			CheckStatus(API.cuTexRefGetMipmappedArray(ref phMipmappedArray, hTexRef));
			return phMipmappedArray;
		}

		public static size_t TexRefSetAddress(CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
			size_t ByteOffset = 0;
			CheckStatus(API.cuTexRefSetAddress(ref ByteOffset, hTexRef, dptr, bytes));
			return ByteOffset;
		}

		public static void TexRefSetAddress2D(CUtexref hTexRef, CUDA_ARRAY_DESCRIPTOR desc, CUdeviceptr dptr, size_t bytes) {
			CheckStatus(API.cuTexRefSetAddress2D(hTexRef, ref desc, dptr, bytes));
		}

		public static void TexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
			CheckStatus(API.cuTexRefSetAddressMode(hTexRef, dim, am));
		}

		public static void TexRefSetArray(CUtexref hTexRef, CUarray hArray, uint Flags) {
			CheckStatus(API.cuTexRefSetArray(hTexRef, hArray, Flags));
		}

		public static void TexRefSetBorderColor(CUtexref hTexRef, float pBorderColor) {
			CheckStatus(API.cuTexRefSetBorderColor(hTexRef, ref pBorderColor));
		}

		public static void TexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
			CheckStatus(API.cuTexRefSetFilterMode(hTexRef, fm));
		}

		public static void TexRefSetFlags(CUtexref hTexRef, uint Flags) {
			CheckStatus(API.cuTexRefSetFlags(hTexRef, Flags));
		}

		public static void TexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
			CheckStatus(API.cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents));
		}

		public static void TexRefSetMaxAnisotropy(CUtexref hTexRef, uint maxAniso) {
			CheckStatus(API.cuTexRefSetMaxAnisotropy(hTexRef, maxAniso));
		}

		public static void TexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
			CheckStatus(API.cuTexRefSetMipmapFilterMode(hTexRef, fm));
		}

		public static void TexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
			CheckStatus(API.cuTexRefSetMipmapLevelBias(hTexRef, bias));
		}

		public static void TexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) {
			CheckStatus(API.cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp));
		}

		public static void TexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, uint Flags) {
			CheckStatus(API.cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags));
		}

		[Obsolete]
		public static CUtexref TexRefCreate() {
			CUtexref pTexRef = IntPtr.Zero;
			CheckStatus(API.cuTexRefCreate(ref pTexRef));
			return pTexRef;
		}

		[Obsolete]
		public static void TexRefDestroy(CUtexref hTexRef) {
			CheckStatus(API.cuTexRefDestroy(hTexRef));
		}

		public static CUarray SurfRefGetArray(CUsurfref hSurfRef) {
			CUarray phArray = IntPtr.Zero;
			CheckStatus(API.cuSurfRefGetArray(ref phArray, hSurfRef));
			return phArray;
		}

		public static void SurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, uint Flags) {
			CheckStatus(API.cuSurfRefSetArray(hSurfRef, hArray, Flags));
		}

		//

		public static void SurfObjectDestroy(CUsurfObject surfObject) {
			CheckStatus(API.cuSurfObjectDestroy(surfObject));
		}

		//


		public static void CtxDisablePeerAccess(CUcontext peerContext) {
			CheckStatus(API.cuCtxDisablePeerAccess(peerContext));
		}

		public static void CtxEnablePeerAccess(CUcontext peerContext, uint Flags) {
			CheckStatus(API.cuCtxEnablePeerAccess(peerContext, Flags));
		}

		public static int DeviceCanAccessPeer(CUdevice dev, CUdevice peerDev) {
			int canAccessPeer = 0;
			CheckStatus(API.cuDeviceCanAccessPeer(ref canAccessPeer, dev, peerDev));
			return canAccessPeer;
		}

		public static int DeviceGetP2PAttribute(CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
			int value = 0;
			CheckStatus(API.cuDeviceGetP2PAttribute(ref value, attrib, srcDevice, dstDevice));
			return value;
		}

		public static void GraphicsMapResources(uint count, CUgraphicsResource resources, CUstream hStream) {
			CheckStatus(API.cuGraphicsMapResources(count, ref resources, hStream));
		}

		public static CUmipmappedArray GraphicsResourceGetMappedMipmappedArray(CUgraphicsResource resource) {
			CUmipmappedArray pMipmappedArray = IntPtr.Zero;
			CheckStatus(API.cuGraphicsResourceGetMappedMipmappedArray(ref pMipmappedArray, resource));
			return pMipmappedArray;
		}

		public static CUdeviceptr GraphicsResourceGetMappedPointer(CUgraphicsResource resource) {
			CUdeviceptr pDevPtr = IntPtr.Zero;
			size_t pSize = 0;
			CheckStatus(API.cuGraphicsResourceGetMappedPointer(ref pDevPtr, ref pSize, resource));
			return pDevPtr;
		}

		public static void GraphicsResourceSetMapFlags(CUgraphicsResource resource, uint flags) {
			CheckStatus(API.cuGraphicsResourceSetMapFlags(resource, flags));
		}

		public static CUarray GraphicsSubResourceGetMappedArray(CUgraphicsResource resource, uint arrayIndex, uint mipLevel) {
			CUarray pArray = IntPtr.Zero;
			CheckStatus(API.cuGraphicsSubResourceGetMappedArray(ref pArray, resource, arrayIndex, mipLevel));
			return pArray;
		}

		public static void GraphicsUnmapResources(uint count, CUgraphicsResource resources, CUstream hStream) {
			CheckStatus(API.cuGraphicsUnmapResources(count, ref resources, hStream));
		}

		/// <summary>
		/// Unregisters a graphics resource for access by CUDA.
		/// </summary>
		/// <param name="resource"></param>
		public static void GraphicsUnregisterResource(CUgraphicsResource resource) {
			CheckStatus(API.cuGraphicsUnregisterResource(resource));
		}

		/// <summary>
		/// Initialize the profiling.
		/// </summary>
		/// <param name="configFile"></param>
		/// <param name="outputFile"></param>
		/// <param name="outputMode"></param>
		[Obsolete]
		public static void ProfilerInitialize(string configFile, string outputFile, CUoutput_mode outputMode) {
			CheckStatus(API.cuProfilerInitialize(configFile, outputFile, outputMode));
		}

		/// <summary>
		/// Enable profiling.
		/// </summary>
		public static void ProfilerStart() {
			CheckStatus(API.cuProfilerStart());
		}

		/// <summary>
		/// Disable profiling.
		/// </summary>
		public static void ProfilerStop() {
			CheckStatus(API.cuProfilerStop());
		}

		static void CheckStatus(CUresult status) {
			if (status != CUresult.CUDA_SUCCESS) {
				throw new CudaException(status.ToString());
			}
		}
	}
}
