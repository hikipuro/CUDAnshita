namespace CUDAnshita {
	/// <summary>
	/// (CUPTI) Error and result codes returned by CUPTI functions.
	/// </summary>
	public enum CUptiResult {
		/// <summary>
		/// No error.
		/// </summary>
		CUPTI_SUCCESS = 0,

		/// <summary>
		/// One or more of the parameters is invalid.
		/// </summary>
		CUPTI_ERROR_INVALID_PARAMETER = 1,

		/// <summary>
		/// The device does not correspond to a valid CUDA device.
		/// </summary>
		CUPTI_ERROR_INVALID_DEVICE = 2,

		/// <summary>
		/// The context is NULL or not valid.
		/// </summary>
		CUPTI_ERROR_INVALID_CONTEXT = 3,

		/// <summary>
		/// The event domain id is invalid.
		/// </summary>
		CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID = 4,

		/// <summary>
		/// The event id is invalid.
		/// </summary>
		CUPTI_ERROR_INVALID_EVENT_ID = 5,

		/// <summary>
		/// The event name is invalid.
		/// </summary>
		CUPTI_ERROR_INVALID_EVENT_NAME = 6,

		/// <summary>
		/// The current operation cannot be performed
		/// due to dependency on other factors.
		/// </summary>
		CUPTI_ERROR_INVALID_OPERATION = 7,

		/// <summary>
		/// Unable to allocate enough memory to perform the requested operation.
		/// </summary>
		CUPTI_ERROR_OUT_OF_MEMORY = 8,

		/// <summary>
		/// An error occurred on the performance monitoring hardware.
		/// </summary>
		CUPTI_ERROR_HARDWARE = 9,

		/// <summary>
		/// The output buffer size is not sufficient to return all requested data.
		/// </summary>
		CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT = 10,

		/// <summary>
		/// API is not implemented.
		/// </summary>
		CUPTI_ERROR_API_NOT_IMPLEMENTED = 11,

		/// <summary>
		/// The maximum limit is reached.
		/// </summary>
		CUPTI_ERROR_MAX_LIMIT_REACHED = 12,

		/// <summary>
		/// The object is not yet ready to perform the requested operation.
		/// </summary>
		CUPTI_ERROR_NOT_READY = 13,

		/// <summary>
		/// The current operation is not compatible with
		/// the current state of the object
		/// </summary>
		CUPTI_ERROR_NOT_COMPATIBLE = 14,

		/// <summary>
		/// CUPTI is unable to initialize its connection to the CUDA driver.
		/// </summary>
		CUPTI_ERROR_NOT_INITIALIZED = 15,

		/// <summary>
		/// The metric id is invalid.
		/// </summary>
		CUPTI_ERROR_INVALID_METRIC_ID = 16,

		/// <summary>
		/// The metric name is invalid.
		/// </summary>
		CUPTI_ERROR_INVALID_METRIC_NAME = 17,

		/// <summary>
		/// The queue is empty.
		/// </summary>
		CUPTI_ERROR_QUEUE_EMPTY = 18,

		/// <summary>
		/// Invalid handle (internal?).
		/// </summary>
		CUPTI_ERROR_INVALID_HANDLE = 19,

		/// <summary>
		/// Invalid stream.
		/// </summary>
		CUPTI_ERROR_INVALID_STREAM = 20,

		/// <summary>
		/// Invalid kind.
		/// </summary>
		CUPTI_ERROR_INVALID_KIND = 21,

		/// <summary>
		/// Invalid event value.
		/// </summary>
		CUPTI_ERROR_INVALID_EVENT_VALUE = 22,

		/// <summary>
		/// CUPTI is disabled due to conflicts with other enabled profilers
		/// </summary>
		CUPTI_ERROR_DISABLED = 23,

		/// <summary>
		/// Invalid module.
		/// </summary>
		CUPTI_ERROR_INVALID_MODULE = 24,

		/// <summary>
		/// Invalid metric value.
		/// </summary>
		CUPTI_ERROR_INVALID_METRIC_VALUE = 25,

		/// <summary>
		/// The performance monitoring hardware is in use by other client.
		/// </summary>
		CUPTI_ERROR_HARDWARE_BUSY = 26,

		/// <summary>
		/// The attempted operation is not supported on the current system or device.
		/// </summary>
		CUPTI_ERROR_NOT_SUPPORTED = 27,

		/// <summary>
		/// Unified memory profiling is not supported on the system.
		/// Potential reason could be unsupported OS or architecture.
		/// </summary>
		CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED = 28,

		/// <summary>
		/// Unified memory profiling is not supported on the device
		/// </summary>
		CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE = 29,

		/// <summary>
		/// Unified memory profiling is not supported on a multi-GPU
		/// configuration without P2P support between any pair of devices
		/// </summary>
		CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES = 30,

		/// <summary>
		/// Unified memory profiling is not supported under
		/// the Multi-Process Service (MPS) environment.
		/// CUDA 7.5 removes this restriction.
		/// </summary>
		CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS = 31,

		/// <summary>
		/// In CUDA 9.0, devices with compute capability 7.0
		/// don't support CDP tracing
		/// </summary>
		CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED = 32,

		/// <summary>
		/// Profiling on virtualized GPU is not supported.
		/// </summary>
		CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED = 33,

		/// <summary>
		/// Profiling results might be incorrect for CUDA applications
		/// compiled with nvcc version older than 9.0 for devices with
		/// compute capability 6.0 and 6.1.
		/// Profiling session will continue and CUPTI will notify it
		/// using this error code. User is advised to recompile the
		/// application code with nvcc version 9.0 or later.
		/// Ignore this warning if code is already compiled with
		/// the recommended nvcc version.
		/// </summary>
		CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE = 34,

		/// <summary>
		/// User doesn't have sufficient privileges which are required
		/// to start the profiling session.
		/// One possible reason for this may be that the NVIDIA driver
		/// or your system administrator may have restricted access to
		/// the NVIDIA GPU performance counters.
		/// To learn how to resolve this issue and find more information,
		/// please visit
		/// https://developer.nvidia.com/CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
		/// </summary>
		CUPTI_ERROR_INSUFFICIENT_PRIVILEGES = 35,

		/// <summary>
		/// Old profiling api's are not supported with new profiling api's
		/// </summary>
		CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED = 36,

		/// <summary>
		/// Missing definition of the OpenACC API routine in the linked OpenACC library.
		/// One possible reason is that OpenACC library is linked statically
		/// in the user application, which might not have the definition of
		/// all the OpenACC API routines needed for the OpenACC profiling,
		/// as compiler might ignore definitions for the functions not used in
		/// the application. This issue can be mitigated by linking the OpenACC
		/// library dynamically.
		/// </summary>
		CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE = 37,

		/// <summary>
		/// An unknown internal error has occurred. Legacy CUPTI Profiling
		/// is not supported on devices with Compute Capability 7.5 or higher (Turing+).
		/// Using this error to specify this case and differentiate it
		/// from other errors.
		/// </summary>
		CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED = 38,

		CUPTI_ERROR_UNKNOWN = 999,
		CUPTI_ERROR_FORCE_INT = 0x7fffffff
	}

}
