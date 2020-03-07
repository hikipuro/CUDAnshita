using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CUDAnshita {
	using cudaError_t = cudaError;
	using cudaEvent_t = IntPtr;
	using cudaExternalMemory_t = IntPtr;
	using cudaExternalSemaphore_t = IntPtr;
	using cudaIpcEventHandle_t = cudaIpcEventHandle;
	using cudaIpcMemHandle_t = cudaIpcMemHandle;
	using cudaStream_t = IntPtr;
	using cudaStreamCallback_t = IntPtr;
	using cudaHostFn_t = IntPtr;
	using cudaGraph_t = IntPtr;
	using cudaGraphNode_t = IntPtr;
	using cudaGraphExec_t = IntPtr;
	using cudaArray_t = IntPtr;
	using cudaArray_const_t = IntPtr;
	using cudaMipmappedArray_t = IntPtr;
	using cudaMipmappedArray_const_t = IntPtr;
	using cudaTextureObject_t = IntPtr;
	using cudaSurfaceObject_t = IntPtr;
	using size_t = Int64;

	/// <summary>
	/// NVIDIA CUDA Runtime API.
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/">http://docs.nvidia.com/cuda/cuda-runtime-api/</a>
	/// </remarks>
	public class Runtime {
		/// <summary>
		/// Runtime API DLL functions.
		/// </summary>
		public class API {
			//const string DLL_PATH = "cudart64_80.dll";
			//const string DLL_PATH = "cudart64_101.dll";
			const string DLL_PATH = "cudart64_102.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			// ----- Device Management

			/// <summary>
			/// Select compute-device which best matches criteria.
			/// </summary>
			/// <param name="device">Device with best match.</param>
			/// <param name="prop">Desired device properties.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaChooseDevice(ref int device, [In] ref cudaDeviceProp prop);

			/// <summary>
			/// Returns information about the device.
			/// </summary>
			/// <param name="value">Returned device attribute value.</param>
			/// <param name="attr">Device attribute to query.</param>
			/// <param name="device">Device number to query.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetAttribute(ref int value, cudaDeviceAttr attr, int device);

			/// <summary>
			/// Returns a handle to a compute device.
			/// </summary>
			/// <param name="device">Returned device ordinal.</param>
			/// <param name="pciBusId">String in one of the following forms:
			/// [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] 
			/// where domain, bus, device, and function are all hexadecimal values</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetByPCIBusId(ref int device, [In] string pciBusId);

			/// <summary>
			/// Returns the preferred cache configuration for the current device.
			/// </summary>
			/// <param name="pCacheConfig">Returned cache configuration.</param>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetCacheConfig(ref cudaFuncCache pCacheConfig);

			/// <summary>
			/// Returns resource limits.
			/// </summary>
			/// <param name="pValue">Returned size of the limit.</param>
			/// <param name="limit">Limit to query.</param>
			/// <returns>cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetLimit(ref size_t pValue, cudaLimit limit);

			/// <summary>
			/// Queries attributes of the link between two devices.
			/// </summary>
			/// <param name="value">Returned value of the requested attribute.</param>
			/// <param name="attr">The source device of the target link.</param>
			/// <param name="srcDevice">The source device of the target link.</param>
			/// <param name="dstDevice">The destination device of the target link.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetP2PAttribute(ref int value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);

			/// <summary>
			/// Returns a PCI Bus Id string for the device.
			/// </summary>
			/// <param name="pciBusId">Returned identifier string for the device in the following format 
			/// [domain]:[bus]:[device].[function] where domain, bus, device, 
			/// and function are all hexadecimal values. pciBusId should be 
			/// large enough to store 13 characters including the NULL-terminator.</param>
			/// <param name="len">Maximum length of string to store in name.</param>
			/// <param name="device">Device to get identifier string for.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetPCIBusId(StringBuilder pciBusId, int len, int device);

			/// <summary>
			/// Returns the shared memory configuration for the current device.
			/// </summary>
			/// <param name="pConfig">Returned cache configuration.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetSharedMemConfig(ref cudaSharedMemConfig pConfig);

			/// <summary>
			/// Returns numerical values that correspond to the least and greatest stream priorities.
			/// </summary>
			/// <param name="leastPriority">Pointer to an int in which the numerical value for least stream priority is returned.</param>
			/// <param name="greatestPriority">Pointer to an int in which the numerical value for greatest stream priority is returned.</param>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceGetStreamPriorityRange(ref int leastPriority, ref int greatestPriority);

			/// <summary>
			/// Destroy all allocations and reset all state on the current device in the current process.
			/// </summary>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceReset();

			/// <summary>
			/// Sets the preferred cache configuration for the current device.
			/// </summary>
			/// <param name="cacheConfig">Requested cache configuration.</param>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);

			/// <summary>
			/// Set resource limits.
			/// </summary>
			/// <param name="limit">Limit to set.</param>
			/// <param name="value">Size of limit.</param>
			/// <returns>cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue, cudaErrorMemoryAllocation</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);

			/// <summary>
			/// Sets the shared memory configuration for the current device.
			/// </summary>
			/// <param name="config">Requested cache configuration.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);

			/// <summary>
			/// Wait for compute device to finish.
			/// </summary>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceSynchronize();

			/// <summary>
			/// Returns which device is currently being used.
			/// </summary>
			/// <param name="device">Returns the device on which the active host thread executes the device code.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDevice(ref int device);

			/// <summary>
			/// Returns the number of compute-capable devices.
			/// </summary>
			/// <param name="count">Returns the number of devices with compute capability greater or equal to 2.0</param>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDeviceCount(ref int count);

			/// <summary>
			/// Gets the flags for the current device.
			/// </summary>
			/// <param name="flags">Pointer to store the device flags.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDeviceFlags(ref uint flags);

			/// <summary>
			/// Returns information about the compute-device.
			/// </summary>
			/// <param name="prop">Properties for the specified device.</param>
			/// <param name="device">Device number to get properties for.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetDeviceProperties(ref cudaDeviceProp prop, int device);

			/// <summary>
			/// Close memory mapped with cudaIpcOpenMemHandle.
			/// </summary>
			/// <param name="devPtr">Device pointer returned by cudaIpcOpenMemHandle.</param>
			/// <returns>cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle, cudaErrorNotSupported</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcCloseMemHandle(IntPtr devPtr);

			/// <summary>
			/// Gets an interprocess handle for a previously allocated event.
			/// </summary>
			/// <param name="handle">Pointer to a user allocated cudaIpcEventHandle in which to return the opaque event handle.</param>
			/// <param name="cudaEvent">Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcGetEventHandle(ref cudaIpcEventHandle_t handle, cudaEvent_t cudaEvent);

			/// <summary>
			/// Gets an interprocess memory handle for an existing device memory allocation.
			/// </summary>
			/// <param name="handle">Pointer to user allocated cudaIpcMemHandle to return the handle in.</param>
			/// <param name="devPtr">Base pointer to previously allocated device memory.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcGetMemHandle(ref cudaIpcMemHandle_t handle, IntPtr devPtr);

			/// <summary>
			/// Opens an interprocess event handle for use in the current process.
			/// </summary>
			/// <param name="cudaEvent">Returns the imported event.</param>
			/// <param name="handle">Interprocess handle to open.</param>
			/// <returns>cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle, cudaErrorNotSupported</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcOpenEventHandle(ref cudaEvent_t cudaEvent, cudaIpcEventHandle_t handle);

			/// <summary>
			/// Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
			/// </summary>
			/// <param name="devPtr">Returned device pointer.</param>
			/// <param name="handle">cudaIpcMemHandle to open.</param>
			/// <param name="flags">Flags for this operation. Must be specified as cudaIpcMemLazyEnablePeerAccess.</param>
			/// <returns>cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle, cudaErrorTooManyPeers, cudaErrorNotSupported</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaIpcOpenMemHandle(ref IntPtr devPtr, cudaIpcMemHandle_t handle, uint flags);

			/// <summary>
			/// Set device to be used for GPU executions.
			/// </summary>
			/// <param name="device">Device on which the active host thread should execute the device code.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice, cudaErrorDeviceAlreadyInUse</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDevice(int device);

			/// <summary>
			/// Sets flags to be used for device executions.
			/// </summary>
			/// <param name="flags">Parameters for device operation.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorSetOnActiveProcess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDeviceFlags(uint flags);

			/// <summary>
			/// Set a list of devices that can be used for CUDA.
			/// </summary>
			/// <param name="device_arr">List of devices to try.</param>
			/// <param name="len">Number of devices in specified list.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetValidDevices(int[] device_arr, int len);

			// ----- Thread Management [DEPRECATED]

			/// <summary>
			/// Exit and clean up from CUDA launches.
			/// </summary>
			/// <returns>cudaSuccess</returns>
			[Obsolete("Note that this function is deprecated because its name does not reflect its behavior. Its functionality is identical to the non-deprecated function cudaDeviceReset(), which should be used instead.")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadExit();

			/// <summary>
			/// Returns the preferred cache configuration for the current device.
			/// </summary>
			/// <param name="pCacheConfig">Returned cache configuration.</param>
			/// <returns>cudaSuccess</returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadGetCacheConfig(ref cudaFuncCache pCacheConfig);

			/// <summary>
			/// Returns resource limits.
			/// </summary>
			/// <param name="pValue">Returned size in bytes of limit.</param>
			/// <param name="limit">Limit to query.</param>
			/// <returns>cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue</returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadGetLimit(ref size_t pValue, cudaLimit limit);

			/// <summary>
			/// Sets the preferred cache configuration for the current device.
			/// </summary>
			/// <param name="cacheConfig">Requested cache configuration.</param>
			/// <returns>cudaSuccess</returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);

			/// <summary>
			/// Set resource limits.
			/// </summary>
			/// <param name="limit">Limit to set.</param>
			/// <param name="value">Size in bytes of limit.</param>
			/// <returns>cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue</returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value);

			/// <summary>
			/// Wait for compute device to finish.
			/// </summary>
			/// <returns>cudaSuccess</returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaThreadSynchronize();

			// ----- Error Handling

			/// <summary>
			/// Returns the string representation of an error code enum name.
			/// </summary>
			/// <param name="error">Error code to convert to string.</param>
			/// <returns>char* pointer to a NULL-terminated string.</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr cudaGetErrorName(cudaError_t error);

			/// <summary>
			/// Returns the description string for an error code.
			/// </summary>
			/// <param name="error">Error code to convert to string.</param>
			/// <returns>char* pointer to a NULL-terminated string.</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr cudaGetErrorString(cudaError_t error);

			/// <summary>
			/// Returns the last error from a runtime call.
			/// </summary>
			/// <returns>
			/// cudaSuccess, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation,
			/// cudaErrorInitializationError, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
			/// cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration,
			/// cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidSymbol,
			/// cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture,
			/// cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, cudaErrorInvalidMemcpyDirection,
			/// cudaErrorInvalidFilterSetting, cudaErrorInvalidNormSetting, cudaErrorUnknown,
			/// cudaErrorInvalidResourceHandle, cudaErrorInsufficientDriver, cudaErrorNoDevice,
			/// cudaErrorSetOnActiveProcess, cudaErrorStartupFailure, cudaErrorInvalidPtx,
			/// cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetLastError();

			/// <summary>
			/// Returns the last error from a runtime call.
			/// </summary>
			/// <returns>
			/// cudaSuccess, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation,
			/// cudaErrorInitializationError, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
			/// cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration,
			/// cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidSymbol,
			/// cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture,
			/// cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, cudaErrorInvalidMemcpyDirection,
			/// cudaErrorInvalidFilterSetting, cudaErrorInvalidNormSetting, cudaErrorUnknown,
			/// cudaErrorInvalidResourceHandle, cudaErrorInsufficientDriver, cudaErrorNoDevice,
			/// cudaErrorSetOnActiveProcess, cudaErrorStartupFailure, cudaErrorInvalidPtx,
			/// cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaPeekAtLastError();

			// ----- Stream Management

			/// <summary>
			/// Add a callback to a compute stream.
			/// </summary>
			/// <param name="stream">Stream to add callback to.</param>
			/// <param name="callback">The function to call once preceding stream operations are complete.</param>
			/// <param name="userData">User specified data to be passed to the callback function.</param>
			/// <param name="flags">Reserved for future use, must be 0.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorInvalidValue, cudaErrorNotSupported</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, IntPtr userData, uint flags);

			/// <summary>
			/// Attach memory to a stream asynchronously.
			/// </summary>
			/// <param name="stream"></param>
			/// <param name="devPtr"></param>
			/// <param name="length"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, IntPtr devPtr, size_t length = 0, uint flags = Defines.cudaMemAttachSingle);

			/// <summary>
			/// Begins graph capture on a stream.
			/// </summary>
			/// <param name="stream"></param>
			/// <param name="mode"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);

			/// <summary>
			/// Create an asynchronous stream.
			/// </summary>
			/// <param name="pStream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamCreate(ref cudaStream_t pStream);

			/// <summary>
			/// Create an asynchronous stream.
			/// </summary>
			/// <param name="pStream"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamCreateWithFlags(ref cudaStream_t pStream, uint flags);

			/// <summary>
			/// Create an asynchronous stream with the specified priority.
			/// </summary>
			/// <param name="pStream"></param>
			/// <param name="flags"></param>
			/// <param name="priority"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamCreateWithPriority(ref cudaStream_t pStream, uint flags, int priority);

			/// <summary>
			/// Destroys and cleans up an asynchronous stream.
			/// </summary>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamDestroy(cudaStream_t stream);

			/// <summary>
			/// Ends capture on a stream, returning the captured graph.
			/// </summary>
			/// <param name="stream"></param>
			/// <param name="pGraph"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, ref cudaGraph_t pGraph);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus** pCaptureStatus, unsigned long long* pId);

			/// <summary>
			/// Query the flags of a stream.
			/// </summary>
			/// <param name="hStream"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, ref uint flags);

			/// <summary>
			/// Query the priority of a stream.
			/// </summary>
			/// <param name="hStream"></param>
			/// <param name="priority"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, ref int priority);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus** pCaptureStatus);

			/// <summary>
			/// Queries an asynchronous stream for completion status.
			/// </summary>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamQuery(cudaStream_t stream);

			/// <summary>
			/// Waits for stream tasks to complete.
			/// </summary>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamSynchronize(cudaStream_t stream);

			/// <summary>
			/// Make a compute stream wait on an event.
			/// </summary>
			/// <param name="stream"></param>
			/// <param name="cudaEvent"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t cudaEvent, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode** mode);

			// ----- Event Management

			/// <summary>
			/// Creates an event object.
			/// </summary>
			/// <param name="cudaEvent">Newly created event.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure, cudaErrorMemoryAllocation</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventCreate(ref cudaEvent_t cudaEvent);

			/// <summary>
			/// Creates an event object with the specified flags.
			/// </summary>
			/// <param name="cudaEvent">Newly created event.</param>
			/// <param name="flags">Flags for new event.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure, cudaErrorMemoryAllocation</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventCreateWithFlags(ref cudaEvent_t cudaEvent, uint flags);

			/// <summary>
			/// Destroys an event object.
			/// </summary>
			/// <param name="cudaEvent">Event to destroy.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventDestroy(cudaEvent_t cudaEvent);

			/// <summary>
			/// Computes the elapsed time between events.
			/// </summary>
			/// <param name="ms">Time between start and end in ms.</param>
			/// <param name="start">Starting event.</param>
			/// <param name="end">Ending event.</param>
			/// <returns>cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventElapsedTime(ref float ms, cudaEvent_t start, cudaEvent_t end);

			/// <summary>
			/// Queries an event's status.
			/// </summary>
			/// <param name="cudaEvent">Event to query.</param>
			/// <returns>cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventQuery(cudaEvent_t cudaEvent);

			/// <summary>
			/// Records an event.
			/// </summary>
			/// <param name="cudaEvent">Event to record.</param>
			/// <param name="stream">Stream in which to record event.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventRecord(cudaEvent_t cudaEvent, cudaStream_t stream);

			/// <summary>
			/// Waits for an event to complete.
			/// </summary>
			/// <param name="cudaEvent">Event to wait for.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaEventSynchronize(cudaEvent_t cudaEvent);

			// ----- External Resource Interoperability

			/// <summary>
			/// Destroys an external memory object.
			/// </summary>
			/// <param name="extMem">External memory object to be destroyed.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem);

			/// <summary>
			/// Destroys an external semaphore.
			/// </summary>
			/// <param name="extSem">External semaphore to be destroyed.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);

			/// <summary>
			/// Maps a buffer onto an imported memory object.
			/// </summary>
			/// <param name="devPtr">Returned device pointer to buffer.</param>
			/// <param name="extMem">Handle to external memory object.</param>
			/// <param name="bufferDesc">Buffer descriptor.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaExternalMemoryGetMappedBuffer(
													IntPtr devPtr, // void**
													cudaExternalMemory_t extMem,
													ref cudaExternalMemoryBufferDesc bufferDesc); // const cudaExternalMemoryBufferDesc*

			/// <summary>
			/// Maps a CUDA mipmapped array onto an external memory object.
			/// </summary>
			/// <param name="mipmap">Returned CUDA mipmapped array.</param>
			/// <param name="extMem">Handle to external memory object.</param>
			/// <param name="mipmapDesc">CUDA array descriptor.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(
													cudaMipmappedArray_t mipmap, // cudaMipmappedArray_t*
													cudaExternalMemory_t extMem,
													ref cudaExternalMemoryMipmappedArrayDesc mipmapDesc); // const cudaExternalMemoryMipmappedArrayDesc*

			/// <summary>
			/// Imports an external memory object.
			/// </summary>
			/// <param name="extMem_out">Returned handle to an external memory object.</param>
			/// <param name="memHandleDesc">Memory import handle descriptor.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaImportExternalMemory(
													cudaExternalMemory_t extMem_out, // cudaExternalMemory_t*
													ref cudaExternalMemoryHandleDesc memHandleDesc); // const cudaExternalMemoryHandleDesc*

			/// <summary>
			/// Imports an external semaphore.
			/// </summary>
			/// <param name="extSem_out">Returned handle to an external semaphore.</param>
			/// <param name="semHandleDesc">Semaphore import handle descriptor.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaImportExternalSemaphore(
													cudaExternalSemaphore_t extSem_out, // cudaExternalSemaphore_t*
													[In] ref cudaExternalSemaphoreHandleDesc semHandleDesc); // const cudaExternalSemaphoreHandleDesc*

			/// <summary>
			/// Signals a set of external semaphore objects.
			/// </summary>
			/// <param name="extSemArray">Set of external semaphores to be signaled.</param>
			/// <param name="paramsArray">Array of semaphore parameters.</param>
			/// <param name="numExtSems">Number of semaphores to signal.</param>
			/// <param name="stream">Stream to enqueue the signal operations in.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSignalExternalSemaphoresAsync(
													[In] ref cudaExternalSemaphore_t extSemArray, // const cudaExternalSemaphore_t*
													[In] ref cudaExternalSemaphoreSignalParams paramsArray, // const cudaExternalSemaphoreSignalParams*
													uint numExtSems,
													cudaStream_t stream = default(IntPtr));

			/// <summary>
			/// Waits on a set of external semaphore objects.
			/// </summary>
			/// <param name="extSemArray">External semaphores to be waited on.</param>
			/// <param name="paramsArray">Array of semaphore parameters.</param>
			/// <param name="numExtSems">Number of semaphores to wait on.</param>
			/// <param name="stream">Stream to enqueue the wait operations in.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaWaitExternalSemaphoresAsync(
													[In] ref cudaExternalSemaphore_t extSemArray, // const cudaExternalSemaphore_t*
													[In] ref cudaExternalSemaphoreWaitParams paramsArray, // const cudaExternalSemaphoreWaitParams*
													uint numExtSems,
													cudaStream_t stream = default(IntPtr));

			// ----- Execution Control

			/// <summary>
			/// Find out attributes for a given function.
			/// </summary>
			/// <param name="attr">Return pointer to function's attributes.</param>
			/// <param name="func">Device function symbol.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDeviceFunction</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFuncGetAttributes(ref cudaFuncAttributes attr, IntPtr func);

			/// <summary>
			/// Set attributes for a given function.
			/// </summary>
			/// <param name="func">Function to get attributes of.</param>
			/// <param name="attr">Attribute to set.</param>
			/// <param name="value">Value to set.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFuncSetAttribute([In] IntPtr func, ref cudaFuncAttribute attr, int value);

			/// <summary>
			/// Sets the preferred cache configuration for a device function.
			/// </summary>
			/// <param name="func"></param>
			/// <param name="cacheConfig"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFuncSetCacheConfig([In] IntPtr func, cudaFuncCache cacheConfig);

			/// <summary>
			/// Sets the shared memory configuration for a device function.
			/// </summary>
			/// <param name="func">Device function symbol.</param>
			/// <param name="config">Requested shared memory configuration.</param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFuncSetSharedMemConfig(IntPtr func, cudaSharedMemConfig config);

			/// <summary>
			/// Obtains a parameter buffer.
			/// </summary>
			/// <param name="alignment"></param>
			/// <param name="size"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr cudaGetParameterBuffer(size_t alignment, size_t size);

			/// <summary>
			/// Launches a specified kernel.
			/// </summary>
			/// <param name="func"></param>
			/// <param name="gridDimension"></param>
			/// <param name="blockDimension"></param>
			/// <param name="sharedMemSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr cudaGetParameterBufferV2(IntPtr func, dim3 gridDimension, dim3 blockDimension, uint sharedMemSize);

			/// <summary>
			/// Launches a device function where thread blocks can cooperate and synchronize as they execute.
			/// </summary>
			/// <param name="func"></param>
			/// <param name="gridDim"></param>
			/// <param name="blockDim"></param>
			/// <param name="args"></param>
			/// <param name="sharedMem"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaLaunchCooperativeKernel(IntPtr func, dim3 gridDim, dim3 blockDim, IntPtr args, size_t sharedMem, cudaStream_t stream);

			/// <summary>
			/// Launches device functions on multiple devices where thread blocks can cooperate and synchronize as they execute.
			/// </summary>
			/// <param name="launchParamsList">List of launch parameters, one per device.</param>
			/// <param name="numDevices">Size of the launchParamsList array.</param>
			/// <param name="flags">Flags to control launch behavior.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration, cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources, cudaErrorCooperativeLaunchTooLarge, cudaErrorSharedObjectInitFailed</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(
				ref cudaLaunchParams launchParamsList, // cudaLaunchParams*
				uint numDevices,
				uint flags = 0);

			/// <summary>
			/// Enqueues a host function call in a stream.
			/// </summary>
			/// <param name="stream"></param>
			/// <param name="fn">The function to call once preceding stream operations are complete.</param>
			/// <param name="userData">User-specified data to be passed to the function.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorInvalidValue, cudaErrorNotSupported</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaLaunchHostFunc(
				cudaStream_t stream,
				cudaHostFn_t fn,
				IntPtr userData); // void*

			/// <summary>
			/// Launches a device function.
			/// </summary>
			/// <param name="func">Device function symbol.</param>
			/// <param name="gridDim">Grid dimentions.</param>
			/// <param name="blockDim">Block dimentions.</param>
			/// <param name="args">Arguments.</param>
			/// <param name="sharedMem">Shared memory.</param>
			/// <param name="stream">Stream identifier.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration, cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources, cudaErrorSharedObjectInitFailed, cudaErrorInvalidPtx, cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaLaunchKernel(IntPtr func, dim3 gridDim, dim3 blockDim, ref IntPtr args, size_t sharedMem, cudaStream_t stream);

			/// <summary>
			/// Converts a double argument to be executed on a device.
			/// </summary>
			/// <param name="d">Double to convert.</param>
			/// <returns>cudaSuccess</returns>
			[Obsolete("This function is deprecated as of CUDA 7.5")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDoubleForDevice(ref double d);

			/// <summary>
			/// Converts a double argument after execution on a device.
			/// </summary>
			/// <param name="d">Double to convert.</param>
			/// <returns>cudaSuccess</returns>
			[Obsolete("This function is deprecated as of CUDA 7.5")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaSetDoubleForHost(ref double d);

			// ----- Occupancy

			/// <summary>
			/// Returns occupancy for a device function.
			/// </summary>
			/// <param name="numBlocks">Returned occupancy.</param>
			/// <param name="func">Kernel function for which occupancy is calculated.</param>
			/// <param name="blockSize">Block size the kernel is intended to be launched with.</param>
			/// <param name="dynamicSMemSize">Per-block dynamic shared memory usage intended, in bytes.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue, cudaErrorUnknown</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(ref int numBlocks, IntPtr func, int blockSize, size_t dynamicSMemSize);

			/// <summary>
			/// Returns occupancy for a device function with the specified flags.
			/// </summary>
			/// <param name="numBlocks">Returned occupancy.</param>
			/// <param name="func">Kernel function for which occupancy is calculated.</param>
			/// <param name="blockSize">Block size the kernel is intended to be launched with.</param>
			/// <param name="dynamicSMemSize">Per-block dynamic shared memory usage intended, in bytes.</param>
			/// <param name="flags">Requested behavior for the occupancy calculator.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue, cudaErrorUnknown</returns>
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

			/// <summary>
			/// Gets info about the specified cudaArray.
			/// </summary>
			/// <param name="desc"></param>
			/// <param name="extent"></param>
			/// <param name="flags"></param>
			/// <param name="array"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaArrayGetInfo(ref cudaChannelFormatDesc desc, ref cudaExtent extent, ref uint flags, cudaArray_t array);

			/// <summary>
			/// Frees memory on the device.
			/// </summary>
			/// <param name="devPtr">Device pointer to memory to free.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFree(IntPtr devPtr);

			/// <summary>
			/// Frees an array on the device.
			/// </summary>
			/// <param name="array"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFreeArray(cudaArray_t array);

			/// <summary>
			/// Frees page-locked memory.
			/// </summary>
			/// <param name="ptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFreeHost(IntPtr ptr);

			/// <summary>
			/// Frees a mipmapped array on the device.
			/// </summary>
			/// <param name="mipmappedArray"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);

			/// <summary>
			/// Gets a mipmap level of a CUDA mipmapped array.
			/// </summary>
			/// <param name="levelArray"></param>
			/// <param name="mipmappedArray"></param>
			/// <param name="level"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetMipmappedArrayLevel(ref cudaArray_t levelArray, cudaMipmappedArray_const_t mipmappedArray, uint level);

			/// <summary>
			/// Finds the address associated with a CUDA symbol.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="symbol"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSymbolAddress(ref IntPtr devPtr, IntPtr symbol);

			/// <summary>
			/// Finds the size of the object associated with a CUDA symbol.
			/// </summary>
			/// <param name="size"></param>
			/// <param name="symbol"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSymbolSize(ref size_t size, IntPtr symbol);

			/// <summary>
			/// Allocates page-locked memory on the host.
			/// </summary>
			/// <param name="pHost"></param>
			/// <param name="size"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostAlloc(ref IntPtr pHost, size_t size, uint flags);

			/// <summary>
			/// Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
			/// </summary>
			/// <param name="pDevice"></param>
			/// <param name="pHost"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostGetDevicePointer(ref IntPtr pDevice, IntPtr pHost, uint flags);

			/// <summary>
			/// Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.
			/// </summary>
			/// <param name="pFlags"></param>
			/// <param name="pHost"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostGetFlags(ref uint pFlags, IntPtr pHost);

			/// <summary>
			/// Registers an existing host memory range for use by CUDA.
			/// </summary>
			/// <param name="ptr"></param>
			/// <param name="size"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostRegister(IntPtr ptr, size_t size, uint flags);

			/// <summary>
			/// Unregisters a memory range that was registered with cudaHostRegister.
			/// </summary>
			/// <param name="ptr"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaHostUnregister(IntPtr ptr);

			/// <summary>
			/// Allocate memory on the device.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="size"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMalloc(ref IntPtr devPtr, size_t size);

			/// <summary>
			/// Allocates logical 1D, 2D, or 3D memory objects on the device.
			/// </summary>
			/// <param name="pitchedDevPtr"></param>
			/// <param name="extent"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMalloc3D(ref cudaPitchedPtr pitchedDevPtr, cudaExtent extent);

			/// <summary>
			/// Allocate an array on the device.
			/// </summary>
			/// <param name="array"></param>
			/// <param name="desc"></param>
			/// <param name="extent"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMalloc3DArray(ref cudaArray_t array, ref cudaChannelFormatDesc desc, cudaExtent extent, uint flags = 0);

			/// <summary>
			/// Allocate an array on the device.
			/// </summary>
			/// <param name="array"></param>
			/// <param name="desc"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocArray(ref cudaArray_t array, ref cudaChannelFormatDesc desc, size_t width, size_t height = 0, uint flags = 0);

			/// <summary>
			/// Allocates page-locked memory on the host.
			/// </summary>
			/// <param name="ptr"></param>
			/// <param name="size"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocHost(ref IntPtr ptr, size_t size);

			/// <summary>
			/// Allocates memory that will be automatically managed by the Unified Memory system.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="size"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocManaged(ref IntPtr devPtr, size_t size, uint flags = Defines.cudaMemAttachGlobal);

			/// <summary>
			/// Allocate a mipmapped array on the device.
			/// </summary>
			/// <param name="mipmappedArray"></param>
			/// <param name="desc"></param>
			/// <param name="extent"></param>
			/// <param name="numLevels"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocMipmappedArray(ref cudaMipmappedArray_t mipmappedArray, ref cudaChannelFormatDesc desc, cudaExtent extent, uint numLevels, uint flags = 0);

			/// <summary>
			/// Allocates pitched memory on the device.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="pitch"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMallocPitch(ref IntPtr devPtr, ref size_t pitch, size_t width, size_t height);

			/// <summary>
			/// Advise about the usage of a given memory range.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="count"></param>
			/// <param name="advice"></param>
			/// <param name="device"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemAdvise(IntPtr devPtr, size_t count, cudaMemoryAdvise advice, int device);

			/// <summary>
			/// Gets free and total device memory.
			/// </summary>
			/// <param name="free"></param>
			/// <param name="total"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemGetInfo(ref size_t free, ref size_t total);

			/// <summary>
			/// Prefetches memory to the specified destination device.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="count"></param>
			/// <param name="dstDevice"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemPrefetchAsync(IntPtr devPtr, size_t count, int dstDevice, cudaStream_t stream);

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
			public static extern cudaError_t cudaMemRangeGetAttribute(IntPtr data, size_t dataSize, cudaMemRangeAttribute attribute, IntPtr devPtr, size_t count);

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
			public static extern cudaError_t cudaMemRangeGetAttributes(ref IntPtr data, ref size_t dataSizes, ref cudaMemRangeAttribute[] attributes, size_t numAttributes, IntPtr devPtr, size_t count);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			unsafe public static extern cudaError_t cudaMemcpy(IntPtr dst, void* src, size_t count, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			unsafe public static extern cudaError_t cudaMemcpy(void* dst, IntPtr src, size_t count, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="dpitch"></param>
			/// <param name="src"></param>
			/// <param name="spitch"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2D(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="wOffsetDst"></param>
			/// <param name="hOffsetDst"></param>
			/// <param name="src"></param>
			/// <param name="wOffsetSrc"></param>
			/// <param name="hOffsetSrc"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="dpitch"></param>
			/// <param name="src"></param>
			/// <param name="spitch"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DAsync(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="dpitch"></param>
			/// <param name="src"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DFromArray(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="dpitch"></param>
			/// <param name="src"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DFromArrayAsync(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="src"></param>
			/// <param name="spitch"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="src"></param>
			/// <param name="spitch"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Copies data between 3D objects.
			/// </summary>
			/// <param name="p"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3D(ref cudaMemcpy3DParms p);

			/// <summary>
			/// Copies data between 3D objects.
			/// </summary>
			/// <param name="p"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3DAsync(ref cudaMemcpy3DParms p, cudaStream_t stream);

			/// <summary>
			/// Copies memory between devices.
			/// </summary>
			/// <param name="p"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3DPeer(ref cudaMemcpy3DPeerParms p);

			/// <summary>
			/// Copies memory between devices asynchronously.
			/// </summary>
			/// <param name="p"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpy3DPeerAsync(ref cudaMemcpy3DPeerParms p, cudaStream_t stream);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="wOffsetDst"></param>
			/// <param name="hOffsetDst"></param>
			/// <param name="src"></param>
			/// <param name="wOffsetSrc"></param>
			/// <param name="hOffsetSrc"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyAsync(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromArray(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="src"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromArrayAsync(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Copies data from the given symbol on the device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="symbol"></param>
			/// <param name="count"></param>
			/// <param name="offset"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromSymbol(IntPtr dst, IntPtr symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToHost);

			/// <summary>
			/// Copies data from the given symbol on the device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="symbol"></param>
			/// <param name="count"></param>
			/// <param name="offset"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyFromSymbolAsync(IntPtr dst, IntPtr symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Copies memory between two devices.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="dstDevice"></param>
			/// <param name="src"></param>
			/// <param name="srcDevice"></param>
			/// <param name="count"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count);

			/// <summary>
			/// Copies memory between two devices asynchronously.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="dstDevice"></param>
			/// <param name="src"></param>
			/// <param name="srcDevice"></param>
			/// <param name="count"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, cudaStream_t stream);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind);

			/// <summary>
			/// Copies data between host and device.
			/// </summary>
			/// <param name="dst"></param>
			/// <param name="wOffset"></param>
			/// <param name="hOffset"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Copies data to the given symbol on the device.
			/// </summary>
			/// <param name="symbol"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="offset"></param>
			/// <param name="kind"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToSymbol(IntPtr symbol, IntPtr src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyHostToDevice);

			/// <summary>
			/// Copies data to the given symbol on the device.
			/// </summary>
			/// <param name="symbol"></param>
			/// <param name="src"></param>
			/// <param name="count"></param>
			/// <param name="offset"></param>
			/// <param name="kind"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemcpyToSymbolAsync(IntPtr symbol, IntPtr src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);

			/// <summary>
			/// Initializes or sets device memory to a value.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="value"></param>
			/// <param name="count"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset(IntPtr devPtr, int value, size_t count);

			/// <summary>
			/// Initializes or sets device memory to a value.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="pitch"></param>
			/// <param name="value"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset2D(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height);

			/// <summary>
			/// Initializes or sets device memory to a value.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="pitch"></param>
			/// <param name="value"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset2DAsync(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);

			/// <summary>
			/// Initializes or sets device memory to a value.
			/// </summary>
			/// <param name="pitchedDevPtr"></param>
			/// <param name="value"></param>
			/// <param name="extent"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent);

			/// <summary>
			/// Initializes or sets device memory to a value.
			/// </summary>
			/// <param name="pitchedDevPtr"></param>
			/// <param name="value"></param>
			/// <param name="extent"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream);

			/// <summary>
			/// Initializes or sets device memory to a value.
			/// </summary>
			/// <param name="devPtr"></param>
			/// <param name="value"></param>
			/// <param name="count"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaMemsetAsync(IntPtr devPtr, int value, size_t count, cudaStream_t stream);

			/// <summary>
			/// Returns a cudaExtent based on input parameters.
			/// </summary>
			/// <param name="w"></param>
			/// <param name="h"></param>
			/// <param name="d"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);

			/// <summary>
			/// Returns a cudaPitchedPtr based on input parameters.
			/// </summary>
			/// <param name="d"></param>
			/// <param name="p"></param>
			/// <param name="xsz"></param>
			/// <param name="ysz"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaPitchedPtr make_cudaPitchedPtr(IntPtr d, size_t p, size_t xsz, size_t ysz);

			/// <summary>
			/// Returns a cudaPos based on input parameters.
			/// </summary>
			/// <param name="x"></param>
			/// <param name="y"></param>
			/// <param name="z"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaPos make_cudaPos(size_t x, size_t y, size_t z);

			// ----- Unified Addressing

			/// <summary>
			/// Returns attributes about a specified pointer.
			/// </summary>
			/// <param name="attributes">Attributes for the specified pointer.</param>
			/// <param name="ptr">Pointer to get attributes for.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaPointerGetAttributes(ref cudaPointerAttributes attributes, IntPtr ptr);

			// ----- Peer Device Memory Access

			/// <summary>
			/// Queries if a device may directly access a peer device's memory.
			/// </summary>
			/// <param name="canAccessPeer"></param>
			/// <param name="device"></param>
			/// <param name="peerDevice"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceCanAccessPeer(ref int canAccessPeer, int device, int peerDevice);

			/// <summary>
			/// Disables direct access to memory allocations on a peer device.
			/// </summary>
			/// <param name="peerDevice"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);

			/// <summary>
			/// Enables direct access to memory allocations on a peer device.
			/// </summary>
			/// <param name="peerDevice"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, uint flags);

			// ----- OpenGL Interoperability

			/// <summary>
			/// Gets the CUDA devices associated with the current OpenGL context.
			/// </summary>
			/// <param name="pCudaDeviceCount"></param>
			/// <param name="pCudaDevices"></param>
			/// <param name="cudaDeviceCount"></param>
			/// <param name="deviceList"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGLGetDevices(
				ref uint pCudaDeviceCount,
				ref int pCudaDevices,
				uint cudaDeviceCount,
				cudaGLDeviceList deviceList);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsGLRegisterBuffer(ref cudaGraphicsResource[] resource, GLuint buffer, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGraphicsGLRegisterImage(ref cudaGraphicsResource[] resource, GLuint image, GLenum target, uint flags);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaWGLGetDevice(ref int device, HGPUNV hGpu);

			// ----- OpenGL Interoperability [DEPRECATED]

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLMapBufferObject(ref IntPtr devPtr, GLuint bufObj);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLMapBufferObjectAsync(ref IntPtr devPtr, GLuint bufObj, cudaStream_t stream);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLRegisterBufferObject(GLuint bufObj);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj, uint flags);

			//[Obsolete("This function is deprecated as of CUDA 5.0")]
			//​[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLSetGLDevice(int device);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLUnmapBufferObject(GLuint bufObj);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj, cudaStream_t stream);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj);

			// ----- Direct3D 9 Interoperability

			/// <summary>
			/// Gets the device number for an adapter.
			/// </summary>
			/// <param name="device"></param>
			/// <param name="pszAdapterName"></param>
			/// <returns></returns>
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

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9MapResources(int count, ref IDirect3DResource9[] ppResources);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9RegisterResource(ref IDirect3DResource9 pResource, uint flags);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedArray(ref cudaArray[] ppArray, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedPitch(ref size_t pPitch, ref size_t pPitchSlice, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedPointer(ref IntPtr pPointer, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetMappedSize(ref size_t pSize, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceGetSurfaceDimensions(ref size_t pWidth, ref size_t pHeight, ref size_t pDepth, ref IDirect3DResource9 pResource, uint face, uint level);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9ResourceSetMapFlags(ref IDirect3DResource9 pResource, uint flags);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D9UnmapResources(int count, ref IDirect3DResource9[] ppResources);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
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

			//[Obsolete("This function is deprecated as of CUDA 5.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10GetDirect3DDevice(ref ID3D10Device[] ppD3D10Device);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10MapResources(int count, ref ID3D10Resource[] ppResources);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10RegisterResource(ref ID3D10Resource pResource, uint flags);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedArray(ref cudaArray[] ppArray, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedPitch(ref size_t pPitch, ref size_t pPitchSlice, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedPointer(ref IntPtr pPointer, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetMappedSize(ref size_t pSize, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceGetSurfaceDimensions(ref size_t pWidth, ref size_t pHeight, ref size_t pDepth, ref ID3D10Resource pResource, uint subResource);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10ResourceSetMapFlags(ref ID3D10Resource pResource, uint flags);

			//[Obsolete("This function is deprecated as of CUDA 5.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10SetDirect3DDevice(ref ID3D10Device pD3D10Device, int device = -1);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D10UnmapResources(int count, ref ID3D10Resource[] ppResources);

			//[Obsolete("This function is deprecated as of CUDA 3.0")]
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

			//[Obsolete("This function is deprecated as of CUDA 5.0")]
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudaError_t cudaD3D11GetDirect3DDevice(ref ID3D11Device[] ppD3D11Device);

			//[Obsolete("This function is deprecated as of CUDA 5.0")]
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

			/// <summary>
			/// Binds a memory area to a texture.
			/// </summary>
			/// <param name="offset"></param>
			/// <param name="texref"></param>
			/// <param name="devPtr"></param>
			/// <param name="desc"></param>
			/// <param name="size"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTexture(ref size_t offset, ref textureReference texref, IntPtr devPtr, ref cudaChannelFormatDesc desc, size_t size);

			/// <summary>
			/// Binds a 2D memory area to a texture.
			/// </summary>
			/// <param name="offset"></param>
			/// <param name="texref"></param>
			/// <param name="devPtr"></param>
			/// <param name="desc"></param>
			/// <param name="width"></param>
			/// <param name="height"></param>
			/// <param name="pitch"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTexture2D(ref size_t offset, ref textureReference texref, IntPtr devPtr, ref cudaChannelFormatDesc desc, size_t width, size_t height, size_t pitch);

			/// <summary>
			/// Binds an array to a texture.
			/// </summary>
			/// <param name="texref"></param>
			/// <param name="array"></param>
			/// <param name="desc"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTextureToArray(ref textureReference texref, cudaArray_const_t array, ref cudaChannelFormatDesc desc);

			/// <summary>
			/// Binds a mipmapped array to a texture.
			/// </summary>
			/// <param name="texref"></param>
			/// <param name="mipmappedArray"></param>
			/// <param name="desc"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindTextureToMipmappedArray(ref textureReference texref, cudaMipmappedArray_const_t mipmappedArray, ref cudaChannelFormatDesc desc);

			/// <summary>
			/// Get the alignment offset of a texture.
			/// </summary>
			/// <param name="offset"></param>
			/// <param name="texref"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureAlignmentOffset(ref size_t offset, ref textureReference texref);

			/// <summary>
			/// Get the texture reference associated with a symbol.
			/// </summary>
			/// <param name="texref"></param>
			/// <param name="symbol"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureReference(ref textureReference texref, IntPtr symbol);

			/// <summary>
			/// Unbinds a texture.
			/// </summary>
			/// <param name="texref"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaUnbindTexture(textureReference texref);

			// ----- Surface Reference Management

			/// <summary>
			/// Binds an array to a surface.
			/// </summary>
			/// <param name="surfref"></param>
			/// <param name="array"></param>
			/// <param name="desc"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaBindSurfaceToArray(ref surfaceReference surfref, cudaArray_const_t array, ref cudaChannelFormatDesc desc);

			/// <summary>
			/// Get the surface reference associated with a symbol.
			/// </summary>
			/// <param name="surfref"></param>
			/// <param name="symbol"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSurfaceReference(ref surfaceReference surfref, IntPtr symbol);

			// ----- Texture Object Management

			/// <summary>
			/// Returns a channel descriptor using the specified format.
			/// </summary>
			/// <param name="x"></param>
			/// <param name="y"></param>
			/// <param name="z"></param>
			/// <param name="w"></param>
			/// <param name="f"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f);

			/// <summary>
			/// Creates a texture object.
			/// </summary>
			/// <param name="pTexObject"></param>
			/// <param name="pResDesc"></param>
			/// <param name="pTexDesc"></param>
			/// <param name="pResViewDesc"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaCreateTextureObject(ref cudaTextureObject_t pTexObject, ref cudaResourceDesc pResDesc, ref cudaTextureDesc pTexDesc, ref cudaResourceViewDesc pResViewDesc);

			/// <summary>
			/// Destroys a texture object.
			/// </summary>
			/// <param name="texObject"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);

			/// <summary>
			/// Get the channel descriptor of an array.
			/// </summary>
			/// <param name="desc"></param>
			/// <param name="array"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetChannelDesc(ref cudaChannelFormatDesc desc, cudaArray_const_t array);

			/// <summary>
			/// Returns a texture object's resource descriptor.
			/// </summary>
			/// <param name="pResDesc"></param>
			/// <param name="texObject"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureObjectResourceDesc(ref cudaResourceDesc pResDesc, cudaTextureObject_t texObject);

			/// <summary>
			/// Returns a texture object's resource view descriptor.
			/// </summary>
			/// <param name="pResViewDesc"></param>
			/// <param name="texObject"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureObjectResourceViewDesc(ref cudaResourceViewDesc pResViewDesc, cudaTextureObject_t texObject);

			/// <summary>
			/// Returns a texture object's texture descriptor.
			/// </summary>
			/// <param name="pTexDesc"></param>
			/// <param name="texObject"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetTextureObjectTextureDesc(ref cudaTextureDesc pTexDesc, cudaTextureObject_t texObject);

			// ----- Surface Object Management

			/// <summary>
			/// Creates a surface object.
			/// </summary>
			/// <param name="pSurfObject"></param>
			/// <param name="pResDesc"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaCreateSurfaceObject(ref cudaSurfaceObject_t pSurfObject, ref cudaResourceDesc pResDesc);

			/// <summary>
			/// Destroys a surface object.
			/// </summary>
			/// <param name="surfObject"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);

			/// <summary>
			/// Returns a surface object's resource descriptor Returns the resource descriptor for the surface object specified by surfObject.
			/// </summary>
			/// <param name="pResDesc"></param>
			/// <param name="surfObject"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGetSurfaceObjectResourceDesc(ref cudaResourceDesc pResDesc, cudaSurfaceObject_t surfObject);

			// ----- Version Management

			/// <summary>
			/// Returns the latest version of CUDA supported by the driver.
			/// </summary>
			/// <param name="driverVersion">Returns the CUDA driver version.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaDriverGetVersion(ref int driverVersion);

			/// <summary>
			/// Returns the CUDA Runtime version.
			/// </summary>
			/// <param name="runtimeVersion">Returns the CUDA Runtime version.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaRuntimeGetVersion(ref int runtimeVersion);

			// ----- Graph Management

			/// <summary>
			/// Creates a child graph node and adds it to a graph.
			/// </summary>
			/// <param name="pGraphNode"></param>
			/// <param name="graph"></param>
			/// <param name="pDependencies"></param>
			/// <param name="numDependencies"></param>
			/// <param name="childGraph"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphAddChildGraphNode(
				ref cudaGraphNode_t pGraphNode, // cudaGraphNode_t*
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t pDependencies, // const cudaGraphNode_t*
				size_t numDependencies,
				cudaGraph_t childGraph);

			/// <summary>
			/// Adds dependency edges to a graph.
			/// </summary>
			/// <param name="graph"></param>
			/// <param name="from"></param>
			/// <param name="to"></param>
			/// <param name="numDependencies"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphAddDependencies(
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t from, // const cudaGraphNode_t*
				[In] ref cudaGraphNode_t to, // const cudaGraphNode_t*
				size_t numDependencies);

			/// <summary>
			/// Creates an empty node and adds it to a graph.
			/// </summary>
			/// <param name="pGraphNode"></param>
			/// <param name="graph"></param>
			/// <param name="pDependencies"></param>
			/// <param name="numDependencies"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphAddEmptyNode(
				ref cudaGraphNode_t pGraphNode,
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t pDependencies, // const cudaGraphNode_t*
				size_t numDependencies);

			/// <summary>
			/// Creates a host execution node and adds it to a graph.
			/// </summary>
			/// <param name="pGraphNode"></param>
			/// <param name="graph"></param>
			/// <param name="pDependencies"></param>
			/// <param name="numDependencies"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphAddHostNode(
				ref cudaGraphNode_t pGraphNode,
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t pDependencies,
				size_t numDependencies,
				[In] ref cudaHostNodeParams pNodeParams);

			/// <summary>
			/// Creates a kernel execution node and adds it to a graph.
			/// </summary>
			/// <param name="pGraphNode"></param>
			/// <param name="graph"></param>
			/// <param name="pDependencies"></param>
			/// <param name="numDependencies"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphAddKernelNode(
				ref cudaGraphNode_t pGraphNode,
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t pDependencies,
				size_t numDependencies,
				[In] ref cudaKernelNodeParams pNodeParams);

			/// <summary>
			/// Creates a memcpy node and adds it to a graph.
			/// </summary>
			/// <param name="pGraphNode"></param>
			/// <param name="graph"></param>
			/// <param name="pDependencies"></param>
			/// <param name="numDependencies"></param>
			/// <param name="pCopyParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphAddMemcpyNode(
				ref cudaGraphNode_t pGraphNode,
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t pDependencies,
				size_t numDependencies,
				[In] ref cudaMemcpy3DParms pCopyParams);

			/// <summary>
			/// Creates a memset node and adds it to a graph.
			/// </summary>
			/// <param name="pGraphNode"></param>
			/// <param name="graph"></param>
			/// <param name="pDependencies"></param>
			/// <param name="numDependencies"></param>
			/// <param name="pMemsetParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphAddMemsetNode(
				ref cudaGraphNode_t pGraphNode,
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t pDependencies,
				size_t numDependencies,
				[In] ref cudaMemsetParams pMemsetParams);

			/// <summary>
			/// Gets a handle to the embedded graph of a child graph node.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pGraph"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphChildGraphNodeGetGraph(
				cudaGraphNode_t node,
				ref cudaGraph_t pGraph);

			/// <summary>
			/// Clones a graph.
			/// </summary>
			/// <param name="pGraphClone"></param>
			/// <param name="originalGraph"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphClone(
				ref cudaGraph_t pGraphClone,
				cudaGraph_t originalGraph);

			/// <summary>
			/// Creates a graph.
			/// </summary>
			/// <param name="pGraph"></param>
			/// <param name="flags"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphCreate(ref cudaGraph_t pGraph, uint flags);

			/// <summary>
			/// Destroys a graph.
			/// </summary>
			/// <param name="graph"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphDestroy(cudaGraph_t graph);

			/// <summary>
			/// Remove a node from the graph.
			/// </summary>
			/// <param name="node"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node);

			/// <summary>
			/// Destroys an executable graph.
			/// </summary>
			/// <param name="graphExec"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);

			/// <summary>
			/// Sets the parameters for a kernel node in the given graphExec.
			/// </summary>
			/// <param name="hGraphExec"></param>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphExecKernelNodeSetParams(
				cudaGraphExec_t hGraphExec,
				cudaGraphNode_t node,
				[In] ref cudaKernelNodeParams pNodeParams);

			/// <summary>
			/// Returns a graph's dependency edges.
			/// </summary>
			/// <param name="graph"></param>
			/// <param name="from"></param>
			/// <param name="to"></param>
			/// <param name="numEdges"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphGetEdges(
				cudaGraph_t graph,
				ref cudaGraphNode_t from,
				ref cudaGraphNode_t to,
				ref size_t numEdges);

			/// <summary>
			/// Returns a graph's nodes.
			/// </summary>
			/// <param name="graph"></param>
			/// <param name="nodes"></param>
			/// <param name="numNodes"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, ref cudaGraphNode_t nodes, ref size_t numNodes);

			/// <summary>
			/// Returns a graph's root nodes.
			/// </summary>
			/// <param name="graph"></param>
			/// <param name="pRootNodes"></param>
			/// <param name="pNumRootNodes"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, ref cudaGraphNode_t pRootNodes, ref size_t pNumRootNodes);

			/// <summary>
			/// Returns a host node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, ref cudaHostNodeParams pNodeParams);

			/// <summary>
			/// Sets a host node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, ref cudaHostNodeParams pNodeParams);

			/// <summary>
			/// Creates an executable graph from a graph.
			/// </summary>
			/// <param name="pGraphExec"></param>
			/// <param name="graph"></param>
			/// <param name="pErrorNode"></param>
			/// <param name="pLogBuffer"></param>
			/// <param name="bufferSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphInstantiate(
				ref cudaGraphExec_t pGraphExec,
				cudaGraph_t graph,
				ref cudaGraphNode_t pErrorNode,
				string pLogBuffer, // char*
				size_t bufferSize);

			/// <summary>
			/// Returns a kernel node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, ref cudaKernelNodeParams pNodeParams);

			/// <summary>
			/// Sets a kernel node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, [In] ref cudaKernelNodeParams pNodeParams);

			/// <summary>
			/// Launches an executable graph in a stream.
			/// </summary>
			/// <param name="graphExec"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);

			/// <summary>
			/// Returns a memcpy node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, ref cudaMemcpy3DParms pNodeParams);

			/// <summary>
			/// Sets a memcpy node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, [In] ref cudaMemcpy3DParms pNodeParams);

			/// <summary>
			/// Returns a memset node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, ref cudaMemsetParams pNodeParams);

			/// <summary>
			/// Sets a memset node's parameters.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pNodeParams"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, [In] ref cudaMemsetParams pNodeParams);

			/// <summary>
			/// Finds a cloned version of a node.
			/// </summary>
			/// <param name="pNode"></param>
			/// <param name="originalNode"></param>
			/// <param name="clonedGraph"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphNodeFindInClone(ref cudaGraphNode_t pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);

			/// <summary>
			/// Returns a node's dependencies.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pDependencies"></param>
			/// <param name="pNumDependencies"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, ref cudaGraphNode_t pDependencies, ref size_t pNumDependencies);

			/// <summary>
			/// Returns a node's dependent nodes.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pDependentNodes"></param>
			/// <param name="pNumDependentNodes"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, ref cudaGraphNode_t pDependentNodes, ref size_t pNumDependentNodes);

			/// <summary>
			/// Returns a node's type.
			/// </summary>
			/// <param name="node"></param>
			/// <param name="pType"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphNodeGetType(
				cudaGraphNode_t node,
				ref cudaGraphNodeType pType); // cudaGraphNodeType**

			/// <summary>
			/// Removes dependency edges from a graph.
			/// </summary>
			/// <param name="graph"></param>
			/// <param name="from"></param>
			/// <param name="to"></param>
			/// <param name="numDependencies"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaGraphRemoveDependencies(
				cudaGraph_t graph,
				[In] ref cudaGraphNode_t from,
				[In] ref cudaGraphNode_t to,
				size_t numDependencies);

			// ----- Profiler Control

			/// <summary>
			/// Initialize the CUDA profiler.
			/// </summary>
			/// <param name="configFile">Name of the config file that lists the counters/options for profiling.</param>
			/// <param name="outputFile">Name of the outputFile where the profiling results will be stored.</param>
			/// <param name="outputMode">outputMode, can be cudaKeyValuePair OR cudaCSV.</param>
			/// <returns>cudaSuccess, cudaErrorInvalidValue, cudaErrorProfilerDisabled</returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode);

			/// <summary>
			/// Enable profiling.
			/// </summary>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaProfilerStart();

			/// <summary>
			/// Disable profiling.
			/// </summary>
			/// <returns>cudaSuccess</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cudaError_t cudaProfilerStop();
		}

		/// <summary>
		/// Select compute-device which best matches criteria.
		/// </summary>
		/// <param name="prop">Desired device properties.</param>
		/// <returns>Device with best match.</returns>
		public static int ChooseDevice(cudaDeviceProp prop) {
			int device = 0;
			CheckStatus(API.cudaChooseDevice(ref device, ref prop));
			return device;
		}

		/// <summary>
		/// Returns information about the device.
		/// </summary>
		/// <param name="attr">Device attribute to query.</param>
		/// <param name="device">Device number to query.</param>
		/// <returns>Device attribute value.</returns>
		public static int DeviceGetAttribute(cudaDeviceAttr attr, int device) {
			int value = 0;
			CheckStatus(API.cudaDeviceGetAttribute(ref value, attr, device));
			return value;
		}

		/// <summary>
		/// Returns a handle to a compute device.
		/// </summary>
		/// <param name="pciBusId">String in one of the following forms: 
		/// [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function]
		/// where domain, bus, device, and function are all hexadecimal values</param>
		/// <returns>Returned device ordinal.</returns>
		public static int DeviceGetByPCIBusId(string pciBusId) {
			int device = 0;
			CheckStatus(API.cudaDeviceGetByPCIBusId(ref device, pciBusId));
			return device;
		}

		/// <summary>
		/// Returns the preferred cache configuration for the current device.
		/// </summary>
		/// <returns>Cache configuration.</returns>
		public static cudaFuncCache DeviceGetCacheConfig() {
			cudaFuncCache pCacheConfig = cudaFuncCache.cudaFuncCachePreferEqual;
			CheckStatus(API.cudaDeviceGetCacheConfig(ref pCacheConfig));
			return pCacheConfig;
		}

		/// <summary>
		/// Returns resource limits.
		/// </summary>
		/// <param name="limit">Limit to query.</param>
		/// <returns>Size of the limit.</returns>
		public static size_t DeviceGetLimit(cudaLimit limit) {
			size_t pValue = 0;
			CheckStatus(API.cudaDeviceGetLimit(ref pValue, limit));
			return pValue;
		}

		/// <summary>
		/// Queries attributes of the link between two devices.
		/// </summary>
		/// <param name="attr">The source device of the target link.</param>
		/// <param name="srcDevice">The source device of the target link.</param>
		/// <param name="dstDevice">The destination device of the target link.</param>
		/// <returns>Value of the requested attribute.</returns>
		public static int DeviceGetP2PAttribute(cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
			int value = 0;
			CheckStatus(API.cudaDeviceGetP2PAttribute(ref value, attr, srcDevice, dstDevice));
			return value;
		}

		/// <summary>
		/// Returns a PCI Bus Id string for the device.
		/// </summary>
		/// <param name="device">Device to get identifier string for.</param>
		/// <returns>
		/// Returned identifier string for the device in the following format
		/// [domain]:[bus]:[device].[function] where domain, bus, device,
		/// and function are all hexadecimal values.
		/// pciBusId should be large enough to store 13 characters including the NULL-terminator.
		/// </returns>
		public static string DeviceGetPCIBusId(int device) {
			StringBuilder pciBusId = new StringBuilder(32);
			CheckStatus(API.cudaDeviceGetPCIBusId(pciBusId, 32, device));
			return pciBusId.ToString();
		}

		/// <summary>
		/// Returns the shared memory configuration for the current device.
		/// </summary>
		/// <returns>Cache configuration.</returns>
		public static cudaSharedMemConfig DeviceGetSharedMemConfig() {
			cudaSharedMemConfig pConfig = new cudaSharedMemConfig();
			CheckStatus(API.cudaDeviceGetSharedMemConfig(ref pConfig));
			return pConfig;
		}

		/// <summary>
		/// Returns numerical values that correspond to the least and greatest stream priorities.
		/// </summary>
		/// <returns></returns>
		public static int[] DeviceGetStreamPriorityRange() {
			int leastPriority = 0;
			int greatestPriority = 0;
			CheckStatus(API.cudaDeviceGetStreamPriorityRange(ref leastPriority, ref greatestPriority));
			return new int[] { leastPriority, greatestPriority };
		}

		/// <summary>
		/// Destroy all allocations and reset all state on the current device in the current process.
		/// </summary>
		public static void DeviceReset() {
			CheckStatus(API.cudaDeviceReset());
		}

		/// <summary>
		/// Sets the preferred cache configuration for the current device.
		/// </summary>
		/// <param name="cacheConfig">Requested cache configuration.</param>
		public static void DeviceSetCacheConfig(cudaFuncCache cacheConfig) {
			CheckStatus(API.cudaDeviceSetCacheConfig(cacheConfig));
		}

		/// <summary>
		/// Set resource limits.
		/// </summary>
		/// <param name="limit">Limit to set.</param>
		/// <param name="value">Size of limit.</param>
		public static void DeviceSetLimit(cudaLimit limit, size_t value) {
			CheckStatus(API.cudaDeviceSetLimit(limit, value));
		}

		/// <summary>
		/// Sets the shared memory configuration for the current device.
		/// </summary>
		/// <param name="config">Requested cache configuration.</param>
		public static void DeviceSetSharedMemConfig(cudaSharedMemConfig config) {
			CheckStatus(API.cudaDeviceSetSharedMemConfig(config));
		}

		/// <summary>
		/// Wait for compute device to finish.
		/// </summary>
		public static void DeviceSynchronize() {
			CheckStatus(API.cudaDeviceSynchronize());
		}

		/// <summary>
		/// Returns which device is currently being used.
		/// </summary>
		/// <returns>Returns the device on which the active host thread executes the device code.</returns>
		public static int GetDevice() {
			int device = 0;
			CheckStatus(API.cudaGetDevice(ref device));
			return device;
		}

		/// <summary>
		/// Returns the number of compute-capable devices.
		/// </summary>
		/// <returns>Returns the number of devices with compute capability greater or equal to 2.0.</returns>
		public static int GetDeviceCount() {
			int count = 0;
			CheckStatus(API.cudaGetDeviceCount(ref count));
			return count;
		}

		/// <summary>
		/// Gets the flags for the current device.
		/// </summary>
		/// <returns>Pointer to store the device flags.</returns>
		public static uint GetDeviceFlags() {
			uint flags = 0;
			CheckStatus(API.cudaGetDeviceFlags(ref flags));
			return flags;
		}

		/// <summary>
		/// Returns information about the compute-device.
		/// </summary>
		/// <param name="device">Device number to get properties for.</param>
		/// <returns>Properties for the specified device.</returns>
		public static cudaDeviceProp GetDeviceProperties(int device) {
			cudaDeviceProp prop = new cudaDeviceProp();
			CheckStatus(API.cudaGetDeviceProperties(ref prop, device));
			return prop;
		}

		/// <summary>
		/// Close memory mapped with cudaIpcOpenMemHandle.
		/// </summary>
		/// <param name="devPtr">Device pointer returned by cudaIpcOpenMemHandle.</param>
		public static void IpcCloseMemHandle(IntPtr devPtr) {
			CheckStatus(API.cudaIpcCloseMemHandle(devPtr));
		}

		/// <summary>
		/// Gets an interprocess handle for a previously allocated event.
		/// </summary>
		/// <param name="cudaEvent">Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags.</param>
		/// <returns>Pointer to a user allocated cudaIpcEventHandle in which to return the opaque event handle.</returns>
		public static cudaIpcEventHandle_t IpcGetEventHandle(cudaEvent_t cudaEvent) {
			cudaIpcEventHandle_t handle = new cudaIpcEventHandle_t();
			CheckStatus(API.cudaIpcGetEventHandle(ref handle, cudaEvent));
			return handle;
		}

		/// <summary>
		/// Gets an interprocess memory handle for an existing device memory allocation.
		/// </summary>
		/// <param name="devPtr">Base pointer to previously allocated device memory.</param>
		/// <returns>Pointer to user allocated cudaIpcMemHandle to return the handle in.</returns>
		public static cudaIpcMemHandle_t IpcGetMemHandle(IntPtr devPtr) {
			cudaIpcMemHandle_t handle = new cudaIpcMemHandle_t();
			CheckStatus(API.cudaIpcGetMemHandle(ref handle, devPtr));
			return handle;
		}

		/// <summary>
		/// Opens an interprocess event handle for use in the current process.
		/// </summary>
		/// <param name="handle">Interprocess handle to open.</param>
		/// <returns>Returns the imported event.</returns>
		public static cudaEvent_t IpcOpenEventHandle(cudaIpcEventHandle_t handle) {
			cudaEvent_t cudaEvent = IntPtr.Zero;
			CheckStatus(API.cudaIpcOpenEventHandle(ref cudaEvent, handle));
			return cudaEvent;
		}

		/// <summary>
		/// Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
		/// </summary>
		/// <param name="handle">cudaIpcMemHandle to open.</param>
		/// <param name="flags">Flags for this operation. Must be specified as cudaIpcMemLazyEnablePeerAccess.</param>
		/// <returns>Returned device pointer.</returns>
		public static IntPtr IpcOpenMemHandle(cudaIpcMemHandle_t handle, uint flags) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaIpcOpenMemHandle(ref devPtr, handle, flags));
			return devPtr;
		}

		/// <summary>
		/// Set device to be used for GPU executions.
		/// </summary>
		/// <param name="device">Device on which the active host thread should execute the device code.</param>
		public static void SetDevice(int device) {
			CheckStatus(API.cudaSetDevice(device));
		}

		/// <summary>
		/// Sets flags to be used for device executions.
		/// </summary>
		/// <param name="flags">Parameters for device operation.</param>
		public static void SetDeviceFlags(uint flags) {
			CheckStatus(API.cudaSetDeviceFlags(flags));
		}

		/// <summary>
		/// Set a list of devices that can be used for CUDA.
		/// </summary>
		/// <param name="devices">List of devices to try.</param>
		public static void SetValidDevices(int[] devices) {
			CheckStatus(API.cudaSetValidDevices(devices, devices.Length));
		}

		/// <summary>
		/// Exit and clean up from CUDA launches.
		/// </summary>
		[Obsolete]
		public static void ThreadExit() {
			CheckStatus(API.cudaThreadExit());
		}

		/// <summary>
		/// Returns the preferred cache configuration for the current device.
		/// </summary>
		/// <returns>Returned cache configuration.</returns>
		[Obsolete]
		public static cudaFuncCache ThreadGetCacheConfig() {
			cudaFuncCache pCacheConfig = new cudaFuncCache();
			CheckStatus(API.cudaThreadGetCacheConfig(ref pCacheConfig));
			return pCacheConfig;
		}

		/// <summary>
		/// Returns resource limits.
		/// </summary>
		/// <param name="limit">Limit to query.</param>
		/// <returns>Returned size in bytes of limit.</returns>
		[Obsolete]
		public static size_t ThreadGetLimit(cudaLimit limit) {
			size_t pValue = 0;
			CheckStatus(API.cudaThreadGetLimit(ref pValue, limit));
			return pValue;
		}

		/// <summary>
		/// Sets the preferred cache configuration for the current device.
		/// </summary>
		/// <param name="cacheConfig">Requested cache configuration.</param>
		[Obsolete]
		public static void ThreadSetCacheConfig(cudaFuncCache cacheConfig) {
			CheckStatus(API.cudaThreadSetCacheConfig(cacheConfig));
		}

		/// <summary>
		/// Set resource limits.
		/// </summary>
		/// <param name="limit">Limit to set.</param>
		/// <param name="value">Size in bytes of limit.</param>
		[Obsolete]
		public static void ThreadSetLimit(cudaLimit limit, size_t value) {
			CheckStatus(API.cudaThreadSetLimit(limit, value));
		}

		/// <summary>
		/// Wait for compute device to finish.
		/// </summary>
		[Obsolete]
		public static void ThreadSynchronize() {
			CheckStatus(API.cudaThreadSynchronize());
		}

		/// <summary>
		/// Returns the string representation of an error code enum name.
		/// </summary>
		/// <param name="error">Error code to convert to string.</param>
		/// <returns>string</returns>
		public static string GetErrorName(cudaError_t error) {
			IntPtr ptr = API.cudaGetErrorName(error);
			return Marshal.PtrToStringAnsi(ptr);
		}

		/// <summary>
		/// Returns the description string for an error code.
		/// </summary>
		/// <param name="error">Error code to convert to string.</param>
		/// <returns>string</returns>
		public static string GetErrorString(cudaError_t error) {
			IntPtr ptr = API.cudaGetErrorString(error);
			return Marshal.PtrToStringAnsi(ptr);
		}

		/// <summary>
		/// Returns the last error from a runtime call.
		/// </summary>
		/// <returns></returns>
		public static cudaError_t GetLastError() {
			return API.cudaGetLastError();
		}

		/// <summary>
		/// Returns the last error from a runtime call.
		/// </summary>
		/// <returns></returns>
		public static cudaError_t PeekAtLastError() {
			return API.cudaPeekAtLastError();
		}

		/// <summary>
		/// Add a callback to a compute stream.
		/// </summary>
		/// <param name="stream"></param>
		/// <param name="callback"></param>
		/// <param name="userData"></param>
		/// <param name="flags"></param>
		public static void StreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, IntPtr userData, uint flags) {
			CheckStatus(API.cudaStreamAddCallback(stream, callback, userData, flags));
		}

		/// <summary>
		/// Attach memory to a stream asynchronously.
		/// </summary>
		/// <param name="stream"></param>
		/// <param name="devPtr"></param>
		/// <param name="length"></param>
		/// <param name="flags"></param>
		public static void StreamAttachMemAsync(cudaStream_t stream, IntPtr devPtr, size_t length = 0, uint flags = Defines.cudaMemAttachSingle) {
			CheckStatus(API.cudaStreamAttachMemAsync(stream, devPtr, length, flags));
		}

		/// <summary>
		/// Create an asynchronous stream.
		/// </summary>
		/// <returns></returns>
		public static cudaStream_t StreamCreate() {
			cudaStream_t pStream = IntPtr.Zero;
			CheckStatus(API.cudaStreamCreate(ref pStream));
			return pStream;
		}

		/// <summary>
		/// Create an asynchronous stream.
		/// </summary>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static cudaStream_t StreamCreateWithFlags(uint flags) {
			cudaStream_t pStream = IntPtr.Zero;
			CheckStatus(API.cudaStreamCreateWithFlags(ref pStream, flags));
			return pStream;
		}

		/// <summary>
		/// Create an asynchronous stream with the specified priority.
		/// </summary>
		/// <param name="flags"></param>
		/// <param name="priority"></param>
		/// <returns></returns>
		public static cudaStream_t StreamCreateWithPriority(uint flags, int priority) {
			cudaStream_t pStream = IntPtr.Zero;
			CheckStatus(API.cudaStreamCreateWithPriority(ref pStream, flags, priority));
			return pStream;
		}

		/// <summary>
		/// Destroys and cleans up an asynchronous stream.
		/// </summary>
		/// <param name="stream"></param>
		public static void StreamDestroy(cudaStream_t stream) {
			CheckStatus(API.cudaStreamDestroy(stream));
		}

		/// <summary>
		/// Query the flags of a stream.
		/// </summary>
		/// <param name="hStream"></param>
		/// <returns></returns>
		public static uint StreamGetFlags(cudaStream_t hStream) {
			uint flags = 0;
			CheckStatus(API.cudaStreamGetFlags(hStream, ref flags));
			return flags;
		}

		/// <summary>
		/// Query the priority of a stream.
		/// </summary>
		/// <param name="hStream"></param>
		/// <returns></returns>
		public static int StreamGetPriority(cudaStream_t hStream) {
			int priority = 0;
			CheckStatus(API.cudaStreamGetPriority(hStream, ref priority));
			return priority;
		}

		/// <summary>
		/// Queries an asynchronous stream for completion status.
		/// </summary>
		/// <param name="stream"></param>
		/// <returns></returns>
		public static cudaError StreamQuery(cudaStream_t stream) {
			return API.cudaStreamQuery(stream);
		}

		/// <summary>
		/// Waits for stream tasks to complete.
		/// </summary>
		/// <param name="stream"></param>
		public static void StreamSynchronize(cudaStream_t stream) {
			CheckStatus(API.cudaStreamSynchronize(stream));
		}

		/// <summary>
		/// Make a compute stream wait on an event.
		/// </summary>
		/// <param name="stream"></param>
		/// <param name="cudaEvent"></param>
		/// <param name="flags"></param>
		public static void StreamWaitEvent(cudaStream_t stream, cudaEvent_t cudaEvent, uint flags) {
			CheckStatus(API.cudaStreamWaitEvent(stream, cudaEvent, flags));
		}

		/// <summary>
		/// Creates an event object.
		/// </summary>
		/// <returns></returns>
		public static cudaEvent_t EventCreate() {
			cudaEvent_t cudaEvent = IntPtr.Zero;
			CheckStatus(API.cudaEventCreate(ref cudaEvent));
			return cudaEvent;
		}

		/// <summary>
		/// Creates an event object with the specified flags.
		/// </summary>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static cudaEvent_t EventCreateWithFlags(uint flags) {
			cudaEvent_t cudaEvent = IntPtr.Zero;
			CheckStatus(API.cudaEventCreateWithFlags(ref cudaEvent, flags));
			return cudaEvent;
		}

		/// <summary>
		/// Destroys an event object.
		/// </summary>
		/// <param name="cudaEvent"></param>
		public static void EventDestroy(cudaEvent_t cudaEvent) {
			CheckStatus(API.cudaEventDestroy(cudaEvent));
		}

		/// <summary>
		/// Computes the elapsed time between events.
		/// </summary>
		/// <param name="start"></param>
		/// <param name="end"></param>
		/// <returns></returns>
		public static float EventElapsedTime(cudaEvent_t start, cudaEvent_t end) {
			float ms = 0;
			CheckStatus(API.cudaEventElapsedTime(ref ms, start, end));
			return ms;
		}

		/// <summary>
		/// Queries an event's status.
		/// </summary>
		/// <param name="cudaEvent"></param>
		/// <returns></returns>
		public static cudaError EventQuery(cudaEvent_t cudaEvent) {
			return API.cudaEventQuery(cudaEvent);
		}

		/// <summary>
		/// Records an event.
		/// </summary>
		/// <param name="cudaEvent"></param>
		/// <param name="stream"></param>
		public static void EventRecord(cudaEvent_t cudaEvent, cudaStream_t stream) {
			CheckStatus(API.cudaEventRecord(cudaEvent, stream));
		}

		/// <summary>
		/// Waits for an event to complete.
		/// </summary>
		/// <param name="cudaEvent"></param>
		public static void EventSynchronize(cudaEvent_t cudaEvent) {
			CheckStatus(API.cudaEventSynchronize(cudaEvent));
		}

		/// <summary>
		/// Find out attributes for a given function.
		/// </summary>
		/// <param name="func"></param>
		/// <returns></returns>
		public static cudaFuncAttributes FuncGetAttributes(IntPtr func) {
			cudaFuncAttributes attr = new cudaFuncAttributes();
			CheckStatus(API.cudaFuncGetAttributes(ref attr, func));
			return attr;
		}

		/// <summary>
		/// Sets the preferred cache configuration for a device function.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="cacheConfig"></param>
		public static void FuncSetCacheConfig(IntPtr func, cudaFuncCache cacheConfig) {
			CheckStatus(API.cudaFuncSetCacheConfig(func, cacheConfig));
		}

		/// <summary>
		/// Sets the shared memory configuration for a device function.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="config"></param>
		public static void FuncSetSharedMemConfig(IntPtr func, cudaSharedMemConfig config) {
			CheckStatus(API.cudaFuncSetSharedMemConfig(func, config));
		}

		/// <summary>
		/// Launches a device function.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="gridDim"></param>
		/// <param name="blockDim"></param>
		/// <param name="args"></param>
		/// <param name="sharedMem"></param>
		/// <param name="stream"></param>
		public static void LaunchKernel(IntPtr func, dim3 gridDim, dim3 blockDim, IntPtr args, size_t sharedMem, cudaStream_t stream) {
			CheckStatus(API.cudaLaunchKernel(func, gridDim, blockDim, ref args, sharedMem, stream));
		}

		/// <summary>
		/// Converts a double argument to be executed on a device.
		/// </summary>
		/// <param name="d"></param>
		[Obsolete("This function is deprecated as of CUDA 7.5")]
		public static void SetDoubleForDevice(double d) {
			CheckStatus(API.cudaSetDoubleForDevice(ref d));
		}

		/// <summary>
		/// Converts a double argument after execution on a device.
		/// </summary>
		/// <param name="d"></param>
		[Obsolete("This function is deprecated as of CUDA 7.5")]
		public static void SetDoubleForHost(double d) {
			CheckStatus(API.cudaSetDoubleForHost(ref d));
		}

		/// <summary>
		/// Returns occupancy for a device function.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="blockSize"></param>
		/// <param name="dynamicSMemSize"></param>
		/// <returns></returns>
		public static int OccupancyMaxActiveBlocksPerMultiprocessor(IntPtr func, int blockSize, size_t dynamicSMemSize) {
			int numBlocks = 0;
			CheckStatus(API.cudaOccupancyMaxActiveBlocksPerMultiprocessor(ref numBlocks, func, blockSize, dynamicSMemSize));
			return numBlocks;
		}

		/// <summary>
		/// Returns occupancy for a device function with the specified flags.
		/// </summary>
		/// <param name="func"></param>
		/// <param name="blockSize"></param>
		/// <param name="dynamicSMemSize"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
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

		/// <summary>
		/// Gets info about the specified cudaArray.
		/// </summary>
		/// <param name="array"></param>
		public static void ArrayGetInfo(cudaArray_t array) {
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			cudaExtent extent = new cudaExtent();
			uint flags = 0;
			CheckStatus(API.cudaArrayGetInfo(ref desc, ref extent, ref flags, array));
		}

		/// <summary>
		/// Frees memory on the device.
		/// </summary>
		/// <param name="devPtr"></param>
		public static void Free(IntPtr devPtr) {
			CheckStatus(API.cudaFree(devPtr));
		}

		/// <summary>
		/// Frees an array on the device.
		/// </summary>
		/// <param name="array"></param>
		public static void FreeArray(cudaArray_t array) {
			CheckStatus(API.cudaFreeArray(array));
		}

		/// <summary>
		/// Frees page-locked memory.
		/// </summary>
		/// <param name="ptr"></param>
		public static void FreeHost(IntPtr ptr) {
			CheckStatus(API.cudaFreeHost(ptr));
		}

		/// <summary>
		/// Frees a mipmapped array on the device.
		/// </summary>
		/// <param name="mipmappedArray"></param>
		public static void FreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
			CheckStatus(API.cudaFreeMipmappedArray(mipmappedArray));
		}

		/// <summary>
		/// Gets a mipmap level of a CUDA mipmapped array.
		/// </summary>
		/// <param name="mipmappedArray"></param>
		/// <param name="level"></param>
		/// <returns></returns>
		public static cudaArray_t GetMipmappedArrayLevel(cudaMipmappedArray_const_t mipmappedArray, uint level) {
			cudaArray_t levelArray = IntPtr.Zero;
			CheckStatus(API.cudaGetMipmappedArrayLevel(ref levelArray, mipmappedArray, level));
			return levelArray;
		}

		/// <summary>
		/// Finds the address associated with a CUDA symbol.
		/// </summary>
		/// <param name="symbol"></param>
		/// <returns></returns>
		public static IntPtr GetSymbolAddress(IntPtr symbol) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaGetSymbolAddress(ref devPtr, symbol));
			return devPtr;
		}

		/// <summary>
		/// Finds the size of the object associated with a CUDA symbol.
		/// </summary>
		/// <param name="symbol"></param>
		/// <returns></returns>
		public static size_t GetSymbolSize(IntPtr symbol) {
			size_t size = 0;
			CheckStatus(API.cudaGetSymbolSize(ref size, symbol));
			return size;
		}

		/// <summary>
		/// Allocates page-locked memory on the host.
		/// </summary>
		/// <param name="size"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static IntPtr HostAlloc(size_t size, uint flags) {
			IntPtr pHost = IntPtr.Zero;
			CheckStatus(API.cudaHostAlloc(ref pHost, size, flags));
			return pHost;
		}

		/// <summary>
		/// Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
		/// </summary>
		/// <param name="pHost"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static IntPtr HostGetDevicePointer(IntPtr pHost, uint flags) {
			IntPtr pDevice = IntPtr.Zero;
			CheckStatus(API.cudaHostGetDevicePointer(ref pDevice, pHost, flags));
			return pHost;
		}

		/// <summary>
		/// Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.
		/// </summary>
		/// <param name="pHost"></param>
		/// <returns></returns>
		public static uint HostGetFlags(IntPtr pHost) {
			uint pFlags = 0;
			CheckStatus(API.cudaHostGetFlags(ref pFlags, pHost));
			return pFlags;
		}

		/// <summary>
		/// Registers an existing host memory range for use by CUDA.
		/// </summary>
		/// <param name="ptr"></param>
		/// <param name="size"></param>
		/// <param name="flags"></param>
		public static void HostRegister(IntPtr ptr, size_t size, uint flags) {
			CheckStatus(API.cudaHostRegister(ptr, size, flags));
		}

		/// <summary>
		/// Unregisters a memory range that was registered with cudaHostRegister.
		/// </summary>
		/// <param name="ptr"></param>
		public static void HostUnregister(IntPtr ptr) {
			CheckStatus(API.cudaHostUnregister(ptr));
		}

		/// <summary>
		/// Allocate memory on the device.
		/// </summary>
		/// <param name="size"></param>
		/// <returns></returns>
		public static IntPtr Malloc(size_t size) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaMalloc(ref devPtr, size));
			return devPtr;
		}

		/// <summary>
		/// Allocates logical 1D, 2D, or 3D memory objects on the device.
		/// </summary>
		/// <param name="extent"></param>
		/// <returns></returns>
		public static cudaPitchedPtr Malloc3D(cudaExtent extent) {
			cudaPitchedPtr pitchedDevPtr = new cudaPitchedPtr();
			CheckStatus(API.cudaMalloc3D(ref pitchedDevPtr, extent));
			return pitchedDevPtr;
		}

		/// <summary>
		/// Allocate an array on the device.
		/// </summary>
		/// <param name="extent"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static cudaArray_t Malloc3DArray(cudaExtent extent, uint flags = 0) {
			cudaArray_t array = IntPtr.Zero;
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaMalloc3DArray(ref array, ref desc, extent, flags));
			return array;
		}

		/// <summary>
		/// Allocate an array on the device.
		/// </summary>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static cudaArray_t MallocArray(size_t width, size_t height = 0, uint flags = 0) {
			cudaArray_t array = IntPtr.Zero;
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaMallocArray(ref array, ref desc, width, height, flags));
			return array;
		}

		/// <summary>
		/// Allocates page-locked memory on the host.
		/// </summary>
		/// <param name="size"></param>
		/// <returns></returns>
		public static IntPtr MallocHost(size_t size) {
			IntPtr ptr = IntPtr.Zero;
			CheckStatus(API.cudaMallocHost(ref ptr, size));
			return ptr;
		}

		/// <summary>
		/// Allocates memory that will be automatically managed by the Unified Memory system.
		/// </summary>
		/// <param name="size"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static IntPtr MallocManaged(size_t size, uint flags = Defines.cudaMemAttachGlobal) {
			IntPtr devPtr = IntPtr.Zero;
			CheckStatus(API.cudaMallocManaged(ref devPtr, size, flags));
			return devPtr;
		}

		/// <summary>
		/// Allocate a mipmapped array on the device.
		/// </summary>
		/// <param name="extent"></param>
		/// <param name="numLevels"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public static cudaMipmappedArray_t MallocMipmappedArray(cudaExtent extent, uint numLevels, uint flags = 0) {
			cudaMipmappedArray_t mipmappedArray = new cudaMipmappedArray_t();
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaMallocMipmappedArray(ref mipmappedArray, ref desc, extent, numLevels, flags));
			return mipmappedArray;
		}

		/// <summary>
		/// Allocates pitched memory on the device.
		/// </summary>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <returns></returns>
		public static IntPtr MallocPitch(size_t width, size_t height) {
			IntPtr devPtr = IntPtr.Zero;
			size_t pitch = 0;
			CheckStatus(API.cudaMallocPitch(ref devPtr, ref pitch, width, height));
			return devPtr;
		}

		/// <summary>
		/// Advise about the usage of a given memory range.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="count"></param>
		/// <param name="advice"></param>
		/// <param name="device"></param>
		public static void MemAdvise(IntPtr devPtr, size_t count, cudaMemoryAdvise advice, int device) {
			CheckStatus(API.cudaMemAdvise(devPtr, count, advice, device));
		}

		/// <summary>
		/// Gets free and total device memory.
		/// </summary>
		/// <returns></returns>
		public static size_t[] MemGetInfo() {
			size_t free = 0;
			size_t total = 0;
			CheckStatus(API.cudaMemGetInfo(ref free, ref total));
			return new size_t[] { free, total };
		}

		/// <summary>
		/// Prefetches memory to the specified destination device.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="count"></param>
		/// <param name="dstDevice"></param>
		/// <param name="stream"></param>
		public static void MemPrefetchAsync(IntPtr devPtr, size_t count, int dstDevice, cudaStream_t stream) {
			CheckStatus(API.cudaMemPrefetchAsync(devPtr, count, dstDevice, stream));
		}

		/// <summary>
		/// Query an attribute of a given memory range.
		/// </summary>
		/// <param name="data"></param>
		/// <param name="dataSize"></param>
		/// <param name="attribute"></param>
		/// <param name="devPtr"></param>
		/// <param name="count"></param>
		public static void MemRangeGetAttribute(IntPtr data, size_t dataSize, cudaMemRangeAttribute attribute, IntPtr devPtr, size_t count) {
			CheckStatus(API.cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count));
		}

		//public static void MemRangeGetAttributes(IntPtr data, size_t dataSize, cudaMemRangeAttribute attribute, IntPtr devPtr, size_t count) {
		//	CheckStatus(API.cudaMemRangeGetAttributes(data, dataSize, attribute, devPtr, count));
		//}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <param name="kind"></param>
		public static void Memcpy(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy(dst, src, count, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="src"></param>
		/// <returns></returns>
		public static T MemcpyD2H<T>(IntPtr src) {
			int byteSize = Marshal.SizeOf(typeof(T));
			IntPtr dst = Marshal.AllocHGlobal(byteSize);

			Memcpy(dst, src, byteSize, cudaMemcpyKind.cudaMemcpyDeviceToHost);

			T[] result = new T[1];
			MarshalUtil.Copy<T>(dst, result, 0, 1);
			Marshal.FreeHGlobal(dst);
			return result[0];
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <returns></returns>
		public static T[] MemcpyD2H<T>(IntPtr src, int count) {
			int byteSize = Marshal.SizeOf(typeof(T)) * count;
			IntPtr dst = Marshal.AllocHGlobal(byteSize);

			Memcpy(dst, src, byteSize, cudaMemcpyKind.cudaMemcpyDeviceToHost);

			T[] result = new T[count];
			MarshalUtil.Copy<T>(dst, result, 0, count);
			Marshal.FreeHGlobal(dst);
			return result;
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <returns></returns>
		unsafe public static float[] MemcpyD2H(IntPtr src, int count) {
			float[] result = new float[count];
			int byteSize = Marshal.SizeOf(typeof(float)) * count;
			fixed (float* dst = result) {
				CheckStatus(API.cudaMemcpy(dst, src, byteSize, cudaMemcpyKind.cudaMemcpyDeviceToHost));
			}
			return result;
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		public static void MemcpyH2D<T>(IntPtr dst, T src) {
			int byteSize = Marshal.SizeOf(typeof(T));
			IntPtr srcPointer = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(new T[] { src }, 0, srcPointer, 1);

			Memcpy(dst, srcPointer, byteSize, cudaMemcpyKind.cudaMemcpyHostToDevice);
			Marshal.FreeHGlobal(srcPointer);
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		/// <param name="count"></param>
		public static void MemcpyH2D<T>(IntPtr dst, T[] src, int count) {
			int byteSize = Marshal.SizeOf(typeof(T)) * count;
			IntPtr srcPointer = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(src, 0, srcPointer, count);

			Memcpy(dst, srcPointer, byteSize, cudaMemcpyKind.cudaMemcpyHostToDevice);
			Marshal.FreeHGlobal(srcPointer);
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		unsafe public static void MemcpyH2D(IntPtr dst, float[] src) {
			int byteSize = Marshal.SizeOf(typeof(float)) * src.Length;
			fixed (float* s = src) {
				CheckStatus(API.cudaMemcpy(dst, s, byteSize, cudaMemcpyKind.cudaMemcpyHostToDevice));
			}
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="dpitch"></param>
		/// <param name="src"></param>
		/// <param name="spitch"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="kind"></param>
		public static void Memcpy2D(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="wOffsetDst"></param>
		/// <param name="hOffsetDst"></param>
		/// <param name="src"></param>
		/// <param name="wOffsetSrc"></param>
		/// <param name="hOffsetSrc"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="kind"></param>
		public static void Memcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice) {
			CheckStatus(API.cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="dpitch"></param>
		/// <param name="src"></param>
		/// <param name="spitch"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		public static void Memcpy2DAsync(IntPtr dst, size_t dpitch, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="dpitch"></param>
		/// <param name="src"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="kind"></param>
		public static void Memcpy2DFromArray(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="dpitch"></param>
		/// <param name="src"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		public static void Memcpy2DFromArrayAsync(IntPtr dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="src"></param>
		/// <param name="spitch"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="kind"></param>
		public static void Memcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="src"></param>
		/// <param name="spitch"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		public static void Memcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream));
		}

		/// <summary>
		/// Copies data between 3D objects.
		/// </summary>
		/// <param name="p"></param>
		public static void Memcpy3D(cudaMemcpy3DParms p) {
			CheckStatus(API.cudaMemcpy3D(ref p));
		}

		/// <summary>
		/// Copies data between 3D objects.
		/// </summary>
		/// <param name="p"></param>
		/// <param name="stream"></param>
		public static void Memcpy3DAsync(cudaMemcpy3DParms p, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy3DAsync(ref p, stream));
		}

		/// <summary>
		/// Copies memory between devices.
		/// </summary>
		/// <param name="p"></param>
		public static void Memcpy3DPeer(cudaMemcpy3DPeerParms p) {
			CheckStatus(API.cudaMemcpy3DPeer(ref p));
		}

		/// <summary>
		/// Copies memory between devices asynchronously.
		/// </summary>
		/// <param name="p"></param>
		/// <param name="stream"></param>
		public static void Memcpy3DPeerAsync(cudaMemcpy3DPeerParms p, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpy3DPeerAsync(ref p, stream));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="wOffsetDst"></param>
		/// <param name="hOffsetDst"></param>
		/// <param name="src"></param>
		/// <param name="wOffsetSrc"></param>
		/// <param name="hOffsetSrc"></param>
		/// <param name="count"></param>
		/// <param name="kind"></param>
		[Obsolete]
		public static void MemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice) {
			CheckStatus(API.cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		public static void MemcpyAsync(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyAsync(dst, src, count, kind, stream));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="count"></param>
		/// <param name="kind"></param>
		[Obsolete]
		public static void MemcpyFromArray(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="src"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="count"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		[Obsolete]
		public static void MemcpyFromArrayAsync(IntPtr dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream));
		}

		/// <summary>
		/// Copies data from the given symbol on the device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="symbol"></param>
		/// <param name="count"></param>
		/// <param name="offset"></param>
		/// <param name="kind"></param>
		public static void MemcpyFromSymbol(IntPtr dst, IntPtr symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyDeviceToHost) {
			CheckStatus(API.cudaMemcpyFromSymbol(dst, symbol, count, offset, kind));
		}

		/// <summary>
		/// Copies data from the given symbol on the device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="symbol"></param>
		/// <param name="count"></param>
		/// <param name="offset"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		public static void MemcpyFromSymbolAsync(IntPtr dst, IntPtr symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream));
		}

		/// <summary>
		/// Copies memory between two devices.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="dstDevice"></param>
		/// <param name="src"></param>
		/// <param name="srcDevice"></param>
		/// <param name="count"></param>
		public static void MemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count) {
			CheckStatus(API.cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count));
		}

		/// <summary>
		/// Copies memory between two devices asynchronously.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="dstDevice"></param>
		/// <param name="src"></param>
		/// <param name="srcDevice"></param>
		/// <param name="count"></param>
		/// <param name="stream"></param>
		public static void MemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <param name="kind"></param>
		[Obsolete]
		public static void MemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind) {
			CheckStatus(API.cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind));
		}

		/// <summary>
		/// Copies data between host and device.
		/// </summary>
		/// <param name="dst"></param>
		/// <param name="wOffset"></param>
		/// <param name="hOffset"></param>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		[Obsolete]
		public static void MemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, IntPtr src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream));
		}

		/// <summary>
		/// Copies data to the given symbol on the device.
		/// </summary>
		/// <param name="symbol"></param>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <param name="offset"></param>
		/// <param name="kind"></param>
		public static void MemcpyToSymbol(IntPtr symbol, IntPtr src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyKind.cudaMemcpyHostToDevice) {
			CheckStatus(API.cudaMemcpyToSymbol(symbol, src, count, offset, kind));
		}

		/// <summary>
		/// Copies data to the given symbol on the device.
		/// </summary>
		/// <param name="symbol"></param>
		/// <param name="src"></param>
		/// <param name="count"></param>
		/// <param name="offset"></param>
		/// <param name="kind"></param>
		/// <param name="stream"></param>
		public static void MemcpyToSymbolAsync(IntPtr symbol, IntPtr src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
			CheckStatus(API.cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream));
		}

		/// <summary>
		/// Initializes or sets device memory to a value.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="value"></param>
		/// <param name="count"></param>
		public static void Memset(IntPtr devPtr, int value, size_t count) {
			CheckStatus(API.cudaMemset(devPtr, value, count));
		}

		/// <summary>
		/// Initializes or sets device memory to a value.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="pitch"></param>
		/// <param name="value"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		public static void Memset2D(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height) {
			CheckStatus(API.cudaMemset2D(devPtr, pitch, value, width, height));
		}

		/// <summary>
		/// Initializes or sets device memory to a value.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="pitch"></param>
		/// <param name="value"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="stream"></param>
		public static void Memset2DAsync(IntPtr devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
			CheckStatus(API.cudaMemset2DAsync(devPtr, pitch, value, width, height, stream));
		}

		/// <summary>
		/// Initializes or sets device memory to a value.
		/// </summary>
		/// <param name="pitchedDevPtr"></param>
		/// <param name="value"></param>
		/// <param name="extent"></param>
		public static void Memset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) {
			CheckStatus(API.cudaMemset3D(pitchedDevPtr, value, extent));
		}

		/// <summary>
		/// Initializes or sets device memory to a value.
		/// </summary>
		/// <param name="pitchedDevPtr"></param>
		/// <param name="value"></param>
		/// <param name="extent"></param>
		/// <param name="stream"></param>
		public static void Memset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) {
			CheckStatus(API.cudaMemset3DAsync(pitchedDevPtr, value, extent, stream));
		}

		/// <summary>
		/// Initializes or sets device memory to a value.
		/// </summary>
		/// <param name="devPtr"></param>
		/// <param name="value"></param>
		/// <param name="count"></param>
		/// <param name="stream"></param>
		public static void MemsetAsync(IntPtr devPtr, int value, size_t count, cudaStream_t stream) {
			CheckStatus(API.cudaMemsetAsync(devPtr, value, count, stream));
		}

		/// <summary>
		/// Returns a cudaExtent based on input parameters.
		/// </summary>
		/// <param name="w"></param>
		/// <param name="h"></param>
		/// <param name="d"></param>
		/// <returns></returns>
		public static cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
			return API.make_cudaExtent(w, h, d);
		}

		/// <summary>
		/// Returns a cudaPitchedPtr based on input parameters.
		/// </summary>
		/// <param name="d"></param>
		/// <param name="p"></param>
		/// <param name="xsz"></param>
		/// <param name="ysz"></param>
		/// <returns></returns>
		public static cudaPitchedPtr make_cudaPitchedPtr(IntPtr d, size_t p, size_t xsz, size_t ysz) {
			return API.make_cudaPitchedPtr(d, p, xsz, ysz);
		}

		/// <summary>
		/// Returns a cudaPos based on input parameters.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="y"></param>
		/// <param name="z"></param>
		/// <returns></returns>
		public static cudaPos make_cudaPos(size_t x, size_t y, size_t z) {
			return API.make_cudaPos(x, y, z);
		}

		/// <summary>
		/// Returns attributes about a specified pointer.
		/// </summary>
		/// <param name="ptr"></param>
		/// <returns></returns>
		public static cudaPointerAttributes PointerGetAttributes(IntPtr ptr) {
			cudaPointerAttributes attributes = new cudaPointerAttributes();
			CheckStatus(API.cudaPointerGetAttributes(ref attributes, ptr));
			return attributes;
		}

		/// <summary>
		/// Queries if a device may directly access a peer device's memory.
		/// </summary>
		/// <param name="device"></param>
		/// <param name="peerDevice"></param>
		/// <returns></returns>
		public static int DeviceCanAccessPeer(int device, int peerDevice) {
			int canAccessPeer = 0;
			CheckStatus(API.cudaDeviceCanAccessPeer(ref canAccessPeer, device, peerDevice));
			return canAccessPeer;
		}

		/// <summary>
		/// Disables direct access to memory allocations on a peer device.
		/// </summary>
		/// <param name="peerDevice"></param>
		public static void DeviceDisablePeerAccess(int peerDevice) {
			CheckStatus(API.cudaDeviceDisablePeerAccess(peerDevice));
		}

		/// <summary>
		/// Enables direct access to memory allocations on a peer device.
		/// </summary>
		/// <param name="peerDevice"></param>
		/// <param name="flags"></param>
		public static void DeviceEnablePeerAccess(int peerDevice, uint flags) {
			CheckStatus(API.cudaDeviceEnablePeerAccess(peerDevice, flags));
		}

		/// <summary>
		/// Binds a memory area to a texture.
		/// </summary>
		/// <param name="offset"></param>
		/// <param name="texref"></param>
		/// <param name="devPtr"></param>
		/// <param name="desc"></param>
		/// <param name="size"></param>
		[Obsolete]
		public static void BindTexture(size_t offset, textureReference texref, IntPtr devPtr, cudaChannelFormatDesc desc, size_t size = uint.MaxValue) {
			CheckStatus(API.cudaBindTexture(ref offset, ref texref, devPtr, ref desc, size));
		}

		/// <summary>
		/// Binds a 2D memory area to a texture.
		/// </summary>
		/// <param name="offset"></param>
		/// <param name="texref"></param>
		/// <param name="devPtr"></param>
		/// <param name="desc"></param>
		/// <param name="width"></param>
		/// <param name="height"></param>
		/// <param name="pitch"></param>
		[Obsolete]
		public static void BindTexture2D(size_t offset, textureReference texref, IntPtr devPtr, cudaChannelFormatDesc desc, size_t width, size_t height, size_t pitch) {
			CheckStatus(API.cudaBindTexture2D(ref offset, ref texref, devPtr, ref desc, width, height, pitch));
		}

		/// <summary>
		/// Binds an array to a texture.
		/// </summary>
		/// <param name="texref"></param>
		/// <param name="array"></param>
		/// <param name="desc"></param>
		[Obsolete]
		public static void BindTextureToArray(textureReference texref, cudaArray_const_t array, cudaChannelFormatDesc desc) {
			CheckStatus(API.cudaBindTextureToArray(ref texref, array, ref desc));
		}

		/// <summary>
		/// Binds a mipmapped array to a texture.
		/// </summary>
		/// <param name="texref"></param>
		/// <param name="mipmappedArray"></param>
		/// <param name="desc"></param>
		[Obsolete]
		public static void BindTextureToMipmappedArray(textureReference texref, cudaMipmappedArray_const_t mipmappedArray, cudaChannelFormatDesc desc) {
			CheckStatus(API.cudaBindTextureToMipmappedArray(ref texref, mipmappedArray, ref desc));
		}

		/// <summary>
		/// Returns a channel descriptor using the specified format.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="y"></param>
		/// <param name="z"></param>
		/// <param name="w"></param>
		/// <param name="f"></param>
		/// <returns></returns>
		public static cudaChannelFormatDesc CreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) {
			return API.cudaCreateChannelDesc(x, y, z, w, f);
		}

		/// <summary>
		/// Get the channel descriptor of an array.
		/// </summary>
		/// <param name="array"></param>
		/// <returns></returns>
		public static cudaChannelFormatDesc GetChannelDesc(cudaArray_const_t array) {
			cudaChannelFormatDesc desc = new cudaChannelFormatDesc();
			CheckStatus(API.cudaGetChannelDesc(ref desc, array));
			return desc;
		}

		/// <summary>
		/// Get the alignment offset of a texture.
		/// </summary>
		/// <param name="texref"></param>
		/// <returns></returns>
		[Obsolete]
		public static size_t GetTextureAlignmentOffset(textureReference texref) {
			size_t offset = 0;
			CheckStatus(API.cudaGetTextureAlignmentOffset(ref offset, ref texref));
			return offset;
		}

		/// <summary>
		/// Get the texture reference associated with a symbol.
		/// </summary>
		/// <param name="symbol"></param>
		/// <returns></returns>
		[Obsolete]
		public static textureReference GetTextureReference(IntPtr symbol) {
			textureReference texref = new textureReference();
			CheckStatus(API.cudaGetTextureReference(ref texref, symbol));
			return texref;
		}

		/// <summary>
		/// Unbinds a texture.
		/// </summary>
		/// <param name="texref"></param>
		[Obsolete]
		public static void UnbindTexture(textureReference texref) {
			CheckStatus(API.cudaUnbindTexture(texref));
		}

		/// <summary>
		/// Binds an array to a surface.
		/// </summary>
		/// <param name="surfref"></param>
		/// <param name="array"></param>
		/// <param name="desc"></param>
		[Obsolete]
		public static void BindSurfaceToArray(surfaceReference surfref, cudaArray_const_t array, cudaChannelFormatDesc desc) {
			CheckStatus(API.cudaBindSurfaceToArray(ref surfref, array, ref desc));
		}

		/// <summary>
		/// Get the surface reference associated with a symbol.
		/// </summary>
		/// <param name="symbol"></param>
		/// <returns></returns>
		[Obsolete]
		public static surfaceReference GetSurfaceReference(IntPtr symbol) {
			surfaceReference surfref = new surfaceReference();
			CheckStatus(API.cudaGetSurfaceReference(ref surfref, symbol));
			return surfref;
		}

		/// <summary>
		/// Creates a texture object.
		/// </summary>
		/// <param name="pResDesc"></param>
		/// <param name="pTexDesc"></param>
		/// <param name="pResViewDesc"></param>
		/// <returns></returns>
		public static cudaTextureObject_t CreateTextureObject(cudaResourceDesc pResDesc, cudaTextureDesc pTexDesc, cudaResourceViewDesc pResViewDesc) {
			cudaTextureObject_t pTexObject = IntPtr.Zero;
			CheckStatus(API.cudaCreateTextureObject(ref pTexObject, ref pResDesc, ref pTexDesc, ref pResViewDesc));
			return pTexObject;
		}

		/// <summary>
		/// Destroys a texture object.
		/// </summary>
		/// <param name="texObject"></param>
		public static void DestroyTextureObject(cudaTextureObject_t texObject) {
			CheckStatus(API.cudaDestroyTextureObject(texObject));
		}

		/// <summary>
		/// Returns a texture object's resource descriptor.
		/// </summary>
		/// <param name="texObject"></param>
		/// <returns></returns>
		public static cudaResourceDesc GetTextureObjectResourceDesc(cudaTextureObject_t texObject) {
			cudaResourceDesc pResDesc = new cudaResourceDesc();
			CheckStatus(API.cudaGetTextureObjectResourceDesc(ref pResDesc, texObject));
			return pResDesc;
		}

		/// <summary>
		/// Returns a texture object's resource view descriptor.
		/// </summary>
		/// <param name="texObject"></param>
		/// <returns></returns>
		public static cudaResourceViewDesc GetTextureObjectResourceViewDesc(cudaTextureObject_t texObject) {
			cudaResourceViewDesc pResViewDesc = new cudaResourceViewDesc();
			CheckStatus(API.cudaGetTextureObjectResourceViewDesc(ref pResViewDesc, texObject));
			return pResViewDesc;
		}

		/// <summary>
		/// Returns a texture object's texture descriptor.
		/// </summary>
		/// <param name="texObject"></param>
		/// <returns></returns>
		public static cudaTextureDesc GetTextureObjectTextureDesc(cudaTextureObject_t texObject) {
			cudaTextureDesc pTexDesc = new cudaTextureDesc();
			CheckStatus(API.cudaGetTextureObjectTextureDesc(ref pTexDesc, texObject));
			return pTexDesc;
		}

		/// <summary>
		/// Creates a surface object.
		/// </summary>
		/// <param name="pResDesc"></param>
		/// <returns></returns>
		public static cudaSurfaceObject_t CreateSurfaceObject(cudaResourceDesc pResDesc) {
			cudaSurfaceObject_t pSurfObject = IntPtr.Zero;
			CheckStatus(API.cudaCreateSurfaceObject(ref pSurfObject, ref pResDesc));
			return pSurfObject;
		}

		/// <summary>
		/// Destroys a surface object.
		/// </summary>
		/// <param name="surfObject"></param>
		public static void DestroySurfaceObject(cudaSurfaceObject_t surfObject) {
			CheckStatus(API.cudaDestroySurfaceObject(surfObject));
		}

		/// <summary>
		/// Returns a surface object's resource descriptor Returns the resource descriptor for the surface object specified by surfObject.
		/// </summary>
		/// <param name="surfObject"></param>
		/// <returns></returns>
		public static cudaResourceDesc GetSurfaceObjectResourceDesc(cudaSurfaceObject_t surfObject) {
			cudaResourceDesc pResDesc = new cudaResourceDesc();
			CheckStatus(API.cudaGetSurfaceObjectResourceDesc(ref pResDesc, surfObject));
			return pResDesc;
		}

		/// <summary>
		/// Returns the latest version of CUDA supported by the driver.
		/// </summary>
		/// <returns>Returns the CUDA driver version.</returns>
		public static int DriverGetVersion() {
			int driverVersion = 0;
			CheckStatus(API.cudaDriverGetVersion(ref driverVersion));
			return driverVersion;
		}

		/// <summary>
		/// Returns the CUDA Runtime version.
		/// </summary>
		/// <returns>Returns the CUDA Runtime version.</returns>
		public static int RuntimeGetVersion() {
			int runtimeVersion = 0;
			CheckStatus(API.cudaRuntimeGetVersion(ref runtimeVersion));
			return runtimeVersion;
		}

		/// <summary>
		/// Initialize the CUDA profiler.
		/// </summary>
		/// <param name="configFile">Name of the config file that lists the counters/options for profiling.</param>
		/// <param name="outputFile">Name of the outputFile where the profiling results will be stored.</param>
		/// <param name="outputMode">outputMode, can be cudaKeyValuePair OR cudaCSV.</param>
		[Obsolete]
		public static void ProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode) {
			CheckStatus(API.cudaProfilerInitialize(configFile, outputFile, outputMode));
		}

		/// <summary>
		/// Enable profiling.
		/// </summary>
		public static void ProfilerStart() {
			CheckStatus(API.cudaProfilerStart());
		}

		/// <summary>
		/// Disable profiling.
		/// </summary>
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
