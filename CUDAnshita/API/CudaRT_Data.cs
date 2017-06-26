using System;

namespace CUDAnshita {
	using size_t = Int64;

	class Defines {
		public const int CUDA_EGL_MAX_PLANES = 3;
		public const int CUDA_IPC_HANDLE_SIZE = 64;
		public const int cudaArrayCubemap = 0x04;
		public const int cudaArrayDefault = 0x00;
		public const int cudaArrayLayered = 0x01;
		public const int cudaArraySurfaceLoadStore = 0x02;
		public const int cudaArrayTextureGather = 0x08;
		public const int cudaCpuDeviceId = ((int)-1);
		public const int cudaDeviceBlockingSync = 0x04;
		public const int cudaDeviceLmemResizeToMax = 0x10;
		public const int cudaDeviceMapHost = 0x08;
		public const int cudaDeviceMask = 0x1f;
		//public const int cudaDevicePropDontCare;
		public const int cudaDeviceScheduleAuto = 0x00;
		public const int cudaDeviceScheduleBlockingSync = 0x04;
		public const int cudaDeviceScheduleMask = 0x07;
		public const int cudaDeviceScheduleSpin = 0x01;
		public const int cudaDeviceScheduleYield = 0x02;
		public const int cudaEventBlockingSync = 0x01;
		public const int cudaEventDefault = 0x00;
		public const int cudaEventDisableTiming = 0x02;
		public const int cudaEventInterprocess = 0x04;
		public const int cudaHostAllocDefault = 0x00;
		public const int cudaHostAllocMapped = 0x02;
		public const int cudaHostAllocPortable = 0x01;
		public const int cudaHostAllocWriteCombined = 0x04;
		public const int cudaHostRegisterDefault = 0x00;
		public const int cudaHostRegisterIoMemory = 0x04;
		public const int cudaHostRegisterMapped = 0x02;
		public const int cudaHostRegisterPortable = 0x01;
		public const int cudaInvalidDeviceId = ((int)-2);
		public const int cudaIpcMemLazyEnablePeerAccess = 0x01;
		public const int cudaMemAttachGlobal = 0x01;
		public const int cudaMemAttachHost = 0x02;
		public const int cudaMemAttachSingle = 0x04;
		public const int cudaOccupancyDefault = 0x00;
		public const int cudaOccupancyDisableCachingOverride = 0x01;
		public const int cudaPeerAccessDefault = 0x00;
		public const int cudaStreamDefault = 0x00;
		//public const int cudaStreamLegacy = ((cudaStream_t)0x1);
		public const int cudaStreamNonBlocking = 0x01;
		//public const int cudaStreamPerThread = ((cudaStream_t)0x2);
	}

	/// <summary>
	/// CUDA Channel format descriptor
	/// </summary>
	public struct cudaChannelFormatDesc {
		cudaChannelFormatKind f;
		int w;
		int x;
		int y;
		int z;
	}

	/*
	public unsafe struct cudaDeviceProp {
		int ECCEnabled;
		int asyncEngineCount;
		int canMapHostMemory;
		int clockRate;
		int computeMode;
		int concurrentKernels;
		int concurrentManagedAccess;
		int deviceOverlap;
		int globalL1CacheSupported;
		int hostNativeAtomicSupported;
		int integrated;
		int isMultiGpuBoard;
		int kernelExecTimeoutEnabled;
		int l2CacheSize;
		int localL1CacheSupported;
		int major;
		int managedMemory;
		fixed int maxGridSize[3];
		int maxSurface1D;
		fixed int maxSurface1DLayered[2];
		fixed int maxSurface2D[2];
		fixed int maxSurface2DLayered[3];
		fixed int maxSurface3D[3];
		int maxSurfaceCubemap;
		fixed int maxSurfaceCubemapLayered[2];
		int maxTexture1D;
		fixed int maxTexture1DLayered[2];
		int maxTexture1DLinear;
		int maxTexture1DMipmap;
		fixed int maxTexture2D[2];
		fixed int maxTexture2DGather[2];
		fixed int maxTexture2DLayered[3];
		fixed int maxTexture2DLinear[3];
		fixed int maxTexture2DMipmap[2];
		fixed int maxTexture3D[3];
		fixed int maxTexture3DAlt[3];
		int maxTextureCubemap;
		fixed int maxTextureCubemapLayered[2];
		fixed int maxThreadsDim[3];
		int maxThreadsPerBlock;
		int maxThreadsPerMultiProcessor;
		size_t memPitch;
		int memoryBusWidth;
		int memoryClockRate;
		int minor;
		int multiGpuBoardGroupID;
		int multiProcessorCount;
		fixed char name[256];
		int pageableMemoryAccess;
		int pciBusID;
		int pciDeviceID;
		int pciDomainID;
		int regsPerBlock;
		int regsPerMultiprocessor;
		size_t sharedMemPerBlock;
		size_t sharedMemPerMultiprocessor;
		int singleToDoublePrecisionPerfRatio;
		int streamPrioritiesSupported;
		size_t surfaceAlignment;
		int tccDriver;
		size_t textureAlignment;
		size_t texturePitchAlignment;
		size_t totalConstMem;
		size_t totalGlobalMem;
		int unifiedAddressing;
		int warpSize;
	}
	//*/

	public struct cudaEglFrame {
	}

	public struct cudaEglPlaneDesc {
	}

	public struct cudaExtent {
	}

	public struct cudaFuncAttributes {
	}

	public struct cudaIpcEventHandle_t {
	}
	public struct cudaIpcMemHandle_t {
	}
	public struct cudaMemcpy3DParms {
	}
	public struct cudaMemcpy3DPeerParms {
	}
	public struct cudaPitchedPtr {
	}
	public struct cudaPointerAttributes {
	}
	public struct cudaPos {
		size_t x;
		size_t y;
		size_t z;
	}
	public struct cudaResourceDesc {
	}
	public struct cudaResourceViewDesc {
	}
	public struct cudaTextureDesc {
	}
	public struct surfaceReference {
	}
	public struct textureReference {
	}


	/// <summary>
	/// Channel format kind
	/// </summary>
	public enum cudaChannelFormatKind {
		// Signed channel format
		cudaChannelFormatKindSigned = 0,
		// Unsigned channel format
		cudaChannelFormatKindUnsigned = 1,
		// Float channel format
		cudaChannelFormatKindFloat = 2,
		// No channel format
		cudaChannelFormatKindNone = 3
	}

	/// <summary>
	/// CUDA device compute modes
	/// </summary>
	public enum cudaComputeMode {
		// Default compute mode (Multiple threads can use cudaSetDevice() with this device)
		cudaComputeModeDefault = 0,
		// Compute-exclusive-thread mode (Only one thread in one process will be able to use cudaSetDevice() with this device)
		cudaComputeModeExclusive = 1,
		// Compute-prohibited mode (No threads can use cudaSetDevice() with this device)
		cudaComputeModeProhibited = 2,
		// Compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice() with this device)
		cudaComputeModeExclusiveProcess = 3
	}

	/// <summary>
	/// CUDA device attributes
	/// </summary>
	public enum cudaDeviceAttr {
		// Maximum number of threads per block
		cudaDevAttrMaxThreadsPerBlock = 1,
		// Maximum block dimension X
		cudaDevAttrMaxBlockDimX = 2,
		// Maximum block dimension Y
		cudaDevAttrMaxBlockDimY = 3,
		// Maximum block dimension Z
		cudaDevAttrMaxBlockDimZ = 4,
		// Maximum grid dimension X
		cudaDevAttrMaxGridDimX = 5,
		// Maximum grid dimension Y
		cudaDevAttrMaxGridDimY = 6,
		// Maximum grid dimension Z
		cudaDevAttrMaxGridDimZ = 7,
		// Maximum shared memory available per block in bytes
		cudaDevAttrMaxSharedMemoryPerBlock = 8,
		// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
		cudaDevAttrTotalConstantMemory = 9,
		// Warp size in threads
		cudaDevAttrWarpSize = 10,
		// Maximum pitch in bytes allowed by memory copies
		cudaDevAttrMaxPitch = 11,
		// Maximum number of 32-bit registers available per block
		cudaDevAttrMaxRegistersPerBlock = 12,
		// Peak clock frequency in kilohertz
		cudaDevAttrClockRate = 13,
		// Alignment requirement for textures
		cudaDevAttrTextureAlignment = 14,
		// Device can possibly copy memory and execute a kernel concurrently
		cudaDevAttrGpuOverlap = 15,
		// Number of multiprocessors on device
		cudaDevAttrMultiProcessorCount = 16,
		// Specifies whether there is a run time limit on kernels
		cudaDevAttrKernelExecTimeout = 17,
		// Device is integrated with host memory
		cudaDevAttrIntegrated = 18,
		// Device can map host memory into CUDA address space
		cudaDevAttrCanMapHostMemory = 19,
		// Compute mode (See cudaComputeMode for details)
		cudaDevAttrComputeMode = 20,
		// Maximum 1D texture width
		cudaDevAttrMaxTexture1DWidth = 21,
		// Maximum 2D texture width
		cudaDevAttrMaxTexture2DWidth = 22,
		// Maximum 2D texture height
		cudaDevAttrMaxTexture2DHeight = 23,
		// Maximum 3D texture width
		cudaDevAttrMaxTexture3DWidth = 24,
		// Maximum 3D texture height
		cudaDevAttrMaxTexture3DHeight = 25,
		// Maximum 3D texture depth
		cudaDevAttrMaxTexture3DDepth = 26,
		// Maximum 2D layered texture width
		cudaDevAttrMaxTexture2DLayeredWidth = 27,
		// Maximum 2D layered texture height
		cudaDevAttrMaxTexture2DLayeredHeight = 28,
		// Maximum layers in a 2D layered texture
		cudaDevAttrMaxTexture2DLayeredLayers = 29,
		// Alignment requirement for surfaces
		cudaDevAttrSurfaceAlignment = 30,
		// Device can possibly execute multiple kernels concurrently
		cudaDevAttrConcurrentKernels = 31,
		// Device has ECC support enabled
		cudaDevAttrEccEnabled = 32,
		// PCI bus ID of the device
		cudaDevAttrPciBusId = 33,
		// PCI device ID of the device
		cudaDevAttrPciDeviceId = 34,
		// Device is using TCC driver model
		cudaDevAttrTccDriver = 35,
		// Peak memory clock frequency in kilohertz
		cudaDevAttrMemoryClockRate = 36,
		// Global memory bus width in bits
		cudaDevAttrGlobalMemoryBusWidth = 37,
		// Size of L2 cache in bytes
		cudaDevAttrL2CacheSize = 38,
		// Maximum resident threads per multiprocessor
		cudaDevAttrMaxThreadsPerMultiProcessor = 39,
		// Number of asynchronous engines
		cudaDevAttrAsyncEngineCount = 40,
		// Device shares a unified address space with the host
		cudaDevAttrUnifiedAddressing = 41,
		// Maximum 1D layered texture width
		cudaDevAttrMaxTexture1DLayeredWidth = 42,
		// Maximum layers in a 1D layered texture
		cudaDevAttrMaxTexture1DLayeredLayers = 43,
		// Maximum 2D texture width if cudaArrayTextureGather is set
		cudaDevAttrMaxTexture2DGatherWidth = 45,
		// Maximum 2D texture height if cudaArrayTextureGather is set
		cudaDevAttrMaxTexture2DGatherHeight = 46,
		// Alternate maximum 3D texture width
		cudaDevAttrMaxTexture3DWidthAlt = 47,
		// Alternate maximum 3D texture height
		cudaDevAttrMaxTexture3DHeightAlt = 48,
		// Alternate maximum 3D texture depth
		cudaDevAttrMaxTexture3DDepthAlt = 49,
		// PCI domain ID of the device
		cudaDevAttrPciDomainId = 50,
		// Pitch alignment requirement for textures
		cudaDevAttrTexturePitchAlignment = 51,
		// Maximum cubemap texture width/height
		cudaDevAttrMaxTextureCubemapWidth = 52,
		// Maximum cubemap layered texture width/height
		cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
		// Maximum layers in a cubemap layered texture
		cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
		// Maximum 1D surface width
		cudaDevAttrMaxSurface1DWidth = 55,
		// Maximum 2D surface width
		cudaDevAttrMaxSurface2DWidth = 56,
		// Maximum 2D surface height
		cudaDevAttrMaxSurface2DHeight = 57,
		// Maximum 3D surface width
		cudaDevAttrMaxSurface3DWidth = 58,
		// Maximum 3D surface height
		cudaDevAttrMaxSurface3DHeight = 59,
		// Maximum 3D surface depth
		cudaDevAttrMaxSurface3DDepth = 60,
		// Maximum 1D layered surface width
		cudaDevAttrMaxSurface1DLayeredWidth = 61,
		// Maximum layers in a 1D layered surface
		cudaDevAttrMaxSurface1DLayeredLayers = 62,
		// Maximum 2D layered surface width
		cudaDevAttrMaxSurface2DLayeredWidth = 63,
		// Maximum 2D layered surface height
		cudaDevAttrMaxSurface2DLayeredHeight = 64,
		// Maximum layers in a 2D layered surface
		cudaDevAttrMaxSurface2DLayeredLayers = 65,
		// Maximum cubemap surface width
		cudaDevAttrMaxSurfaceCubemapWidth = 66,
		// Maximum cubemap layered surface width
		cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
		// Maximum layers in a cubemap layered surface
		cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
		// Maximum 1D linear texture width
		cudaDevAttrMaxTexture1DLinearWidth = 69,
		// Maximum 2D linear texture width
		cudaDevAttrMaxTexture2DLinearWidth = 70,
		// Maximum 2D linear texture height
		cudaDevAttrMaxTexture2DLinearHeight = 71,
		// Maximum 2D linear texture pitch in bytes
		cudaDevAttrMaxTexture2DLinearPitch = 72,
		// Maximum mipmapped 2D texture width
		cudaDevAttrMaxTexture2DMipmappedWidth = 73,
		// Maximum mipmapped 2D texture height
		cudaDevAttrMaxTexture2DMipmappedHeight = 74,
		// Major compute capability version number
		cudaDevAttrComputeCapabilityMajor = 75,
		// Minor compute capability version number
		cudaDevAttrComputeCapabilityMinor = 76,
		// Maximum mipmapped 1D texture width
		cudaDevAttrMaxTexture1DMipmappedWidth = 77,
		// Device supports stream priorities
		cudaDevAttrStreamPrioritiesSupported = 78,
		// Device supports caching globals in L1
		cudaDevAttrGlobalL1CacheSupported = 79,
		// Device supports caching locals in L1
		cudaDevAttrLocalL1CacheSupported = 80,
		// Maximum shared memory available per multiprocessor in bytes
		cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
		// Maximum number of 32-bit registers available per multiprocessor
		cudaDevAttrMaxRegistersPerMultiprocessor = 82,
		// Device can allocate managed memory on this system
		cudaDevAttrManagedMemory = 83,
		// Device is on a multi-GPU board
		cudaDevAttrIsMultiGpuBoard = 84,
		// Unique identifier for a group of devices on the same multi-GPU board
		cudaDevAttrMultiGpuBoardGroupID = 85,
		// Link between the device and the host supports native atomic operations
		cudaDevAttrHostNativeAtomicSupported = 86,
		// Ratio of single precision performance (in floating-point operations per second) to double precision performance
		cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
		// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
		cudaDevAttrPageableMemoryAccess = 88,
		// Device can coherently access managed memory concurrently with the CPU
		cudaDevAttrConcurrentManagedAccess = 89,
		// Device supports Compute Preemption
		cudaDevAttrComputePreemptionSupported = 90,
		// Device can access host registered memory at the same virtual address as the CPU
		cudaDevAttrCanUseHostPointerForRegisteredMem = 91
	}

	/// <summary>
	/// CUDA device P2P attributes
	/// </summary>
	public enum cudaDeviceP2PAttr {
		// A relative value indicating the performance of the link between two devices
		cudaDevP2PAttrPerformanceRank = 1,
		// Peer access is enabled
		cudaDevP2PAttrAccessSupported = 2,
		// Native atomic operation over the link supported
		cudaDevP2PAttrNativeAtomicSupported = 3
	}

	/// <summary>
	/// CUDA EGL Color Format - The different planar and multiplanar formats currently supported for CUDA_EGL interops.
	/// </summary>
	public enum cudaEglColorFormat {
		// Y, U, V in three surfaces, each in a separate surface, U/V width = 1 / 2 Y width, U/V height = 1 / 2 Y height.
		cudaEglColorFormatYUV420Planar = 0,
		// Y, UV in two surfaces (UV as one surface), width, height ratio same as YUV420Planar.
		cudaEglColorFormatYUV420SemiPlanar = 1,
		// Y, U, V each in a separate surface, U/V width = 1 / 2 Y width, U/V height = Y height.
		cudaEglColorFormatYUV422Planar = 2,
		// Y, UV in two surfaces, width, height ratio same as YUV422Planar.
		cudaEglColorFormatYUV422SemiPlanar = 3,
		// R/G/B three channels in one surface with RGB byte ordering.
		cudaEglColorFormatRGB = 4,
		// R/G/B three channels in one surface with BGR byte ordering.
		cudaEglColorFormatBGR = 5,
		// R/G/B/A four channels in one surface with ARGB byte ordering.
		cudaEglColorFormatARGB = 6,
		// R/G/B/A four channels in one surface with RGBA byte ordering.
		cudaEglColorFormatRGBA = 7,
		// single luminance channel in one surface.
		cudaEglColorFormatL = 8,
		// single color channel in one surface.
		cudaEglColorFormatR = 9,
		// Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.
		cudaEglColorFormatYUV444Planar = 10,
		// Y, UV in two surfaces (UV as one surface), width, height ratio same as YUV444Planar.
		cudaEglColorFormatYUV444SemiPlanar = 11,
		// Y, U, V in one surface, interleaved as YUYV.
		cudaEglColorFormatYUYV422 = 12,
		// Y, U, V in one surface, interleaved as UYVY.
		cudaEglColorFormatUYVY422 = 13
	}

	/// <summary>
	/// CUDA EglFrame type - array or pointer
	/// </summary>
	public enum cudaEglFrameType {
		// Frame type CUDA array
		cudaEglFrameTypeArray = 0,
		// Frame type CUDA pointer
		cudaEglFrameTypePitch = 1
	}

	/// <summary>
	/// Resource location flags- sysmem or vidmem
	/// </summary>
	public enum cudaEglResourceLocationFlags {
		// Resource location sysmem
		cudaEglResourceLocationSysmem = 0x00,
		// Resource location vidmem
		cudaEglResourceLocationVidmem = 0x01
	}

	/// <summary>
	/// CUDA error types
	/// </summary>
	public enum cudaError {
		cudaSuccess = 0,
		cudaErrorMissingConfiguration = 1,
		cudaErrorMemoryAllocation = 2,
		cudaErrorInitializationError = 3,
		cudaErrorLaunchFailure = 4,
		cudaErrorPriorLaunchFailure = 5,
		cudaErrorLaunchTimeout = 6,
		cudaErrorLaunchOutOfResources = 7,
		cudaErrorInvalidDeviceFunction = 8,
		cudaErrorInvalidConfiguration = 9,
		cudaErrorInvalidDevice = 10,
		cudaErrorInvalidValue = 11,
		cudaErrorInvalidPitchValue = 12,
		cudaErrorInvalidSymbol = 13,
		cudaErrorMapBufferObjectFailed = 14,
		cudaErrorUnmapBufferObjectFailed = 15,
		cudaErrorInvalidHostPointer = 16,
		cudaErrorInvalidDevicePointer = 17,
		cudaErrorInvalidTexture = 18,
		cudaErrorInvalidTextureBinding = 19,
		cudaErrorInvalidChannelDescriptor = 20,
		cudaErrorInvalidMemcpyDirection = 21,
		cudaErrorAddressOfConstant = 22,
		cudaErrorTextureFetchFailed = 23,
		cudaErrorTextureNotBound = 24,
		cudaErrorSynchronizationError = 25,
		cudaErrorInvalidFilterSetting = 26,
		cudaErrorInvalidNormSetting = 27,
		cudaErrorMixedDeviceExecution = 28,
		cudaErrorCudartUnloading = 29,
		cudaErrorUnknown = 30,
		cudaErrorNotYetImplemented = 31,
		cudaErrorMemoryValueTooLarge = 32,
		cudaErrorInvalidResourceHandle = 33,
		cudaErrorNotReady = 34,
		cudaErrorInsufficientDriver = 35,
		cudaErrorSetOnActiveProcess = 36,
		cudaErrorInvalidSurface = 37,
		cudaErrorNoDevice = 38,
		cudaErrorECCUncorrectable = 39,
		cudaErrorSharedObjectSymbolNotFound = 40,
		cudaErrorSharedObjectInitFailed = 41,
		cudaErrorUnsupportedLimit = 42,
		cudaErrorDuplicateVariableName = 43,
		cudaErrorDuplicateTextureName = 44,
		cudaErrorDuplicateSurfaceName = 45,
		cudaErrorDevicesUnavailable = 46,
		cudaErrorInvalidKernelImage = 47,
		cudaErrorNoKernelImageForDevice = 48,
		cudaErrorIncompatibleDriverContext = 49,
		cudaErrorPeerAccessAlreadyEnabled = 50,
		cudaErrorPeerAccessNotEnabled = 51,
		cudaErrorDeviceAlreadyInUse = 54,
		cudaErrorProfilerDisabled = 55,
		cudaErrorProfilerNotInitialized = 56,
		cudaErrorProfilerAlreadyStarted = 57,
		cudaErrorProfilerAlreadyStopped = 58,
		cudaErrorAssert = 59,
		cudaErrorTooManyPeers = 60,
		cudaErrorHostMemoryAlreadyRegistered = 61,
		cudaErrorHostMemoryNotRegistered = 62,
		cudaErrorOperatingSystem = 63,
		cudaErrorPeerAccessUnsupported = 64,
		cudaErrorLaunchMaxDepthExceeded = 65,
		cudaErrorLaunchFileScopedTex = 66,
		cudaErrorLaunchFileScopedSurf = 67,
		cudaErrorSyncDepthExceeded = 68,
		cudaErrorLaunchPendingCountExceeded = 69,
		cudaErrorNotPermitted = 70,
		cudaErrorNotSupported = 71,
		cudaErrorHardwareStackError = 72,
		cudaErrorIllegalInstruction = 73,
		cudaErrorMisalignedAddress = 74,
		cudaErrorInvalidAddressSpace = 75,
		cudaErrorInvalidPc = 76,
		cudaErrorIllegalAddress = 77,
		cudaErrorInvalidPtx = 78,
		cudaErrorInvalidGraphicsContext = 79,
		cudaErrorNvlinkUncorrectable = 80,
		cudaErrorStartupFailure = 0x7f,
		cudaErrorApiFailureBase = 10000
	}

	/// <summary>
	/// CUDA function cache configurations
	/// </summary>
	public enum cudaFuncCache {
		// Default function cache configuration, no preference
		cudaFuncCachePreferNone = 0,
		// Prefer larger shared memory and smaller L1 cache
		cudaFuncCachePreferShared = 1,
		// Prefer larger L1 cache and smaller shared memory
		cudaFuncCachePreferL1 = 2,
		// Prefer equal size L1 cache and shared memory
		cudaFuncCachePreferEqual = 3
	}

	/// <summary>
	/// CUDA graphics interop array indices for cube maps
	/// </summary>
	public enum cudaGraphicsCubeFace {
		// Positive X face of cubemap
		cudaGraphicsCubeFacePositiveX = 0x00,
		// Negative X face of cubemap
		cudaGraphicsCubeFaceNegativeX = 0x01,
		// Positive Y face of cubemap
		cudaGraphicsCubeFacePositiveY = 0x02,
		// Negative Y face of cubemap
		cudaGraphicsCubeFaceNegativeY = 0x03,
		// Positive Z face of cubemap
		cudaGraphicsCubeFacePositiveZ = 0x04,
		// Negative Z face of cubemap
		cudaGraphicsCubeFaceNegativeZ = 0x05
	}

	/// <summary>
	/// CUDA graphics interop register flags
	/// </summary>
	public enum cudaGraphicsMapFlags {
		// Default
		cudaGraphicsRegisterFlagsNone = 0,
		// CUDA will not write to this resource
		cudaGraphicsRegisterFlagsReadOnly = 1,
		// CUDA will only write to and will not read from this resource
		cudaGraphicsRegisterFlagsWriteDiscard = 2,
		// CUDA will bind this resource to a surface reference
		cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
		// CUDA will perform texture gather operations on this resource
		cudaGraphicsRegisterFlagsTextureGather = 8
	}

	/// <summary>
	/// CUDA Limits
	/// </summary>
	public enum cudaLimit {
		// GPU thread stack size
		cudaLimitStackSize = 0x00,
		// GPU printf/fprintf FIFO size
		cudaLimitPrintfFifoSize = 0x01,
		// GPU malloc heap size
		cudaLimitMallocHeapSize = 0x02,
		// GPU device runtime synchronize depth
		cudaLimitDevRuntimeSyncDepth = 0x03,
		// GPU device runtime pending launch count
		cudaLimitDevRuntimePendingLaunchCount = 0x04
	}

	/// <summary>
	/// CUDA range attributes
	/// </summary>
	public enum cudaMemRangeAttribute {
		// Whether the range will mostly be read and only occassionally be written to
		cudaMemRangeAttributeReadMostly = 1,
		// The preferred location of the range
		cudaMemRangeAttributePreferredLocation = 2,
		// Memory range has cudaMemAdviseSetAccessedBy set for specified device
		cudaMemRangeAttributeAccessedBy = 3,
		// The last location to which the range was prefetched
		cudaMemRangeAttributeLastPrefetchLocation = 4
	}

	/// <summary>
	/// CUDA memory copy types
	/// </summary>
	public enum cudaMemcpyKind {
		// Host -> Host
		cudaMemcpyHostToHost = 0,
		// Host -> Device
		cudaMemcpyHostToDevice = 1,
		// Device -> Host
		cudaMemcpyDeviceToHost = 2,
		// Device -> Device
		cudaMemcpyDeviceToDevice = 3,
		// Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing
		cudaMemcpyDefault = 4
	}

	/// <summary>
	/// CUDA Memory Advise values
	/// </summary>
	public enum cudaMemoryAdvise {
		// Data will mostly be read and only occassionally be written to
		cudaMemAdviseSetReadMostly = 1,
		// Undo the effect of cudaMemAdviseSetReadMostly
		cudaMemAdviseUnsetReadMostly = 2,
		// Set the preferred location for the data as the specified device
		cudaMemAdviseSetPreferredLocation = 3,
		// Clear the preferred location for the data
		cudaMemAdviseUnsetPreferredLocation = 4,
		// Data will be accessed by the specified device, so prevent page faults as much as possible
		cudaMemAdviseSetAccessedBy = 5,
		// Let the Unified Memory subsystem decide on the page faulting policy for the specified device
		cudaMemAdviseUnsetAccessedBy = 6
	}

	/// <summary>
	/// CUDA memory types
	/// </summary>
	public enum cudaMemoryType {
		// Host memory
		cudaMemoryTypeHost = 1,
		// Device memory
		cudaMemoryTypeDevice = 2
	}

	/// <summary>
	/// CUDA Profiler Output modes
	/// </summary>
	public enum cudaOutputMode {
		// Output mode Key-Value pair format.
		cudaKeyValuePair = 0x00,
		// Output mode Comma separated values format.
		cudaCSV = 0x01
	}

	/// <summary>
	/// CUDA resource types
	/// </summary>
	public enum cudaResourceType {
		// Array resource
		cudaResourceTypeArray = 0x00,
		// Mipmapped array resource
		cudaResourceTypeMipmappedArray = 0x01,
		// Linear resource
		cudaResourceTypeLinear = 0x02,
		// Pitch 2D resource
		cudaResourceTypePitch2D = 0x03
	}

	/// <summary>
	/// CUDA texture resource view formats
	/// </summary>
	public enum cudaResourceViewFormat {
		// No resource view format (use underlying resource format)
		cudaResViewFormatNone = 0x00,
		// 1 channel unsigned 8-bit integers
		cudaResViewFormatUnsignedChar1 = 0x01,
		// 2 channel unsigned 8-bit integers
		cudaResViewFormatUnsignedChar2 = 0x02,
		// 4 channel unsigned 8-bit integers
		cudaResViewFormatUnsignedChar4 = 0x03,
		// 1 channel signed 8-bit integers
		cudaResViewFormatSignedChar1 = 0x04,
		// 2 channel signed 8-bit integers
		cudaResViewFormatSignedChar2 = 0x05,
		// 4 channel signed 8-bit integers
		cudaResViewFormatSignedChar4 = 0x06,
		// 1 channel unsigned 16-bit integers
		cudaResViewFormatUnsignedShort1 = 0x07,
		// 2 channel unsigned 16-bit integers
		cudaResViewFormatUnsignedShort2 = 0x08,
		// 4 channel unsigned 16-bit integers
		cudaResViewFormatUnsignedShort4 = 0x09,
		// 1 channel signed 16-bit integers
		cudaResViewFormatSignedShort1 = 0x0a,
		// 2 channel signed 16-bit integers
		cudaResViewFormatSignedShort2 = 0x0b,
		// 4 channel signed 16-bit integers
		cudaResViewFormatSignedShort4 = 0x0c,
		// 1 channel unsigned 32-bit integers
		cudaResViewFormatUnsignedInt1 = 0x0d,
		// 2 channel unsigned 32-bit integers
		cudaResViewFormatUnsignedInt2 = 0x0e,
		// 4 channel unsigned 32-bit integers
		cudaResViewFormatUnsignedInt4 = 0x0f,
		// 1 channel signed 32-bit integers
		cudaResViewFormatSignedInt1 = 0x10,
		// 2 channel signed 32-bit integers
		cudaResViewFormatSignedInt2 = 0x11,
		// 4 channel signed 32-bit integers
		cudaResViewFormatSignedInt4 = 0x12,
		// 1 channel 16-bit floating point
		cudaResViewFormatHalf1 = 0x13,
		// 2 channel 16-bit floating point
		cudaResViewFormatHalf2 = 0x14,
		// 4 channel 16-bit floating point
		cudaResViewFormatHalf4 = 0x15,
		// 1 channel 32-bit floating point
		cudaResViewFormatFloat1 = 0x16,
		// 2 channel 32-bit floating point
		cudaResViewFormatFloat2 = 0x17,
		// 4 channel 32-bit floating point
		cudaResViewFormatFloat4 = 0x18,
		// Block compressed 1
		cudaResViewFormatUnsignedBlockCompressed1 = 0x19,
		// Block compressed 2
		cudaResViewFormatUnsignedBlockCompressed2 = 0x1a,
		// Block compressed 3
		cudaResViewFormatUnsignedBlockCompressed3 = 0x1b,
		// Block compressed 4 unsigned
		cudaResViewFormatUnsignedBlockCompressed4 = 0x1c,
		// Block compressed 4 signed
		cudaResViewFormatSignedBlockCompressed4 = 0x1d,
		// Block compressed 5 unsigned
		cudaResViewFormatUnsignedBlockCompressed5 = 0x1e,
		// Block compressed 5 signed
		cudaResViewFormatSignedBlockCompressed5 = 0x1f,
		// Block compressed 6 unsigned half-float
		cudaResViewFormatUnsignedBlockCompressed6H = 0x20,
		// Block compressed 6 signed half-float
		cudaResViewFormatSignedBlockCompressed6H = 0x21,
		// Block compressed 7
		cudaResViewFormatUnsignedBlockCompressed7 = 0x22
	}

	/// <summary>
	/// CUDA shared memory configuration
	/// </summary>
	public enum cudaSharedMemConfig {
		cudaSharedMemBankSizeDefault = 0,
		cudaSharedMemBankSizeFourByte = 1,
		cudaSharedMemBankSizeEightByte = 2
	}

	/// <summary>
	/// CUDA Surface boundary modes
	/// </summary>
	public enum cudaSurfaceBoundaryMode {
		// Zero boundary mode
		cudaBoundaryModeZero = 0,
		// Clamp boundary mode
		cudaBoundaryModeClamp = 1,
		// Trap boundary mode
		cudaBoundaryModeTrap = 2
	}

	/// <summary>
	/// CUDA Surface format modes
	/// </summary>
	public enum cudaSurfaceFormatMode {
		// Forced format mode
		cudaFormatModeForced = 0,
		// Auto format mode
		cudaFormatModeAuto = 1
	}

	/// <summary>
	/// CUDA texture address modes
	/// </summary>
	public enum cudaTextureAddressMode {
		// Wrapping address mode
		cudaAddressModeWrap = 0,
		// Clamp to edge address mode
		cudaAddressModeClamp = 1,
		// Mirror address mode
		cudaAddressModeMirror = 2,
		// Border address mode
		cudaAddressModeBorder = 3
	}

	/// <summary>
	/// CUDA texture filter modes
	/// </summary>
	public enum cudaTextureFilterMode {
		// Point filter mode
		cudaFilterModePoint = 0,
		// Linear filter mode
		cudaFilterModeLinear = 1
	}

	/// <summary>
	/// CUDA texture read modes
	/// </summary>
	public enum cudaTextureReadMode {
		// Read texture as specified element type
		cudaReadModeElementType = 0,
		// Read texture as normalized float
		cudaReadModeNormalizedFloat = 1
	}
}
