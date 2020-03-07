using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using cudaArray_t = IntPtr;
	using cudaStream_t = IntPtr;
	using cudaMipmappedArray_t = IntPtr;
	using cudaUUID_t = CUuuid_st;
	using size_t = Int64;

	public partial class Defines {
		/// <summary>
		/// (Runtime API) Maximum number of planes per frame
		/// </summary>
		public const int CUDA_EGL_MAX_PLANES = 3;

		/// <summary>
		/// (Runtime API) CUDA IPC Handle Size
		/// </summary>
		public const int CUDA_IPC_HANDLE_SIZE = 64;

		/// <summary>
		/// (Runtime API) Must be set in cudaExternalMemoryGetMappedMipmappedArray
		/// if the mipmapped array is used as a color target in a graphics API
		/// </summary>
		public const int cudaArrayColorAttachment = 0x20;

		//public const int cudaDevicePropDontCare;

		/// <summary>
		/// (Runtime API) Default page-locked allocation flag.
		/// </summary>
		public const int cudaHostAllocDefault = 0x00;

		/// <summary>
		/// (Runtime API) Pinned memory accessible by all CUDA contexts.
		/// </summary>
		public const int cudaHostAllocPortable = 0x01;

		/// <summary>
		/// (Runtime API) Map allocation into device space.
		/// </summary>
		public const int cudaHostAllocMapped = 0x02;

		/// <summary>
		/// (Runtime API) Write-combined memory.
		/// </summary>
		public const int cudaHostAllocWriteCombined = 0x04;


		/// <summary>
		/// (Runtime API) Default host memory registration flag.
		/// </summary>
		public const int cudaHostRegisterDefault = 0x00;

		/// <summary>
		/// (Runtime API) Pinned memory accessible by all CUDA contexts.
		/// </summary>
		public const int cudaHostRegisterPortable = 0x01;

		/// <summary>
		/// (Runtime API) Map registered memory into device space.
		/// </summary>
		public const int cudaHostRegisterMapped = 0x02;

		/// <summary>
		/// (Runtime API) Memory-mapped I/O space.
		/// </summary>
		public const int cudaHostRegisterIoMemory = 0x04;


		/// <summary>
		/// (Runtime API) Default peer addressing enable flag.
		/// </summary>
		public const int cudaPeerAccessDefault = 0x00;

		/// <summary>
		/// (Runtime API) Default stream flag.
		/// </summary>
		public const int cudaStreamDefault = 0x00;

		/// <summary>
		/// (Runtime API) Stream does not synchronize with stream 0 (the NULL stream).
		/// </summary>
		public const int cudaStreamNonBlocking = 0x01;

		/// <summary>
		/// (Runtime API) Legacy stream handle.
		/// </summary>
		/// <remarks>
		/// Stream handle that can be passed as a cudaStream_t to use an implicit stream
		/// with legacy synchronization behavior.
		///
		/// See details of the \link_sync_behavior
		/// </remarks>
		public readonly cudaStream_t cudaStreamLegacy = new IntPtr(0x1);

		/// <summary>
		/// (Runtime API) Per-thread stream handle.
		/// </summary>
		/// <remarks>
		/// Stream handle that can be passed as a cudaStream_t to use an implicit stream
		/// with per-thread synchronization behavior.
		///
		/// See details of the \link_sync_behavior
		/// </remarks>
		public readonly cudaStream_t cudaStreamPerThread = new IntPtr(0x2);


		/// <summary>
		/// (Runtime API) Default event flag.
		/// </summary>
		public const int cudaEventDefault = 0x00;

		/// <summary>
		/// (Runtime API) Event uses blocking synchronization.
		/// </summary>
		public const int cudaEventBlockingSync = 0x01;

		/// <summary>
		/// (Runtime API) Event will not record timing data.
		/// </summary>
		public const int cudaEventDisableTiming = 0x02;

		/// <summary>
		/// (Runtime API) Event is suitable for interprocess use. cudaEventDisableTiming must be set.
		/// </summary>
		public const int cudaEventInterprocess = 0x04;


		/// <summary>
		/// (Runtime API) Device flag - Automatic scheduling.
		/// </summary>
		public const int cudaDeviceScheduleAuto = 0x00;

		/// <summary>
		/// (Runtime API) Device flag - Spin default scheduling.
		/// </summary>
		public const int cudaDeviceScheduleSpin = 0x01;

		/// <summary>
		/// (Runtime API) Device flag - Yield default scheduling.
		/// </summary>
		public const int cudaDeviceScheduleYield = 0x02;

		/// <summary>
		/// (Runtime API) Device flag - Use blocking synchronization.
		/// </summary>
		public const int cudaDeviceScheduleBlockingSync = 0x04;

		/// <summary>
		/// (Runtime API) Device flag - Use blocking synchronization
		/// \deprecated This flag was deprecated as of CUDA 4.0 and
		/// replaced with ::cudaDeviceScheduleBlockingSync.
		/// </summary>
		[Obsolete("This flag was deprecated as of CUDA 4.0")]
		public const int cudaDeviceBlockingSync = 0x04;

		/// <summary>
		/// (Runtime API) Device schedule flags mask.
		/// </summary>
		public const int cudaDeviceScheduleMask = 0x07;

		/// <summary>
		/// (Runtime API) Device flag - Support mapped pinned allocations.
		/// </summary>
		public const int cudaDeviceMapHost = 0x08;

		/// <summary>
		/// (Runtime API) Device flag - Keep local memory allocation after launch.
		/// </summary>
		public const int cudaDeviceLmemResizeToMax = 0x10;

		/// <summary>
		/// (Runtime API) Device flags mask.
		/// </summary>
		public const int cudaDeviceMask = 0x1f;


		/// <summary>
		/// (Runtime API) Default CUDA array allocation flag.
		/// </summary>
		public const int cudaArrayDefault = 0x00;

		/// <summary>
		/// (Runtime API) Must be set in cudaMalloc3DArray to create a layered CUDA array.
		/// </summary>
		public const int cudaArrayLayered = 0x01;

		/// <summary>
		/// (Runtime API) Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array.
		/// </summary>
		public const int cudaArraySurfaceLoadStore = 0x02;

		/// <summary>
		/// (Runtime API) Must be set in cudaMalloc3DArray to create a cubemap CUDA array.
		/// </summary>
		public const int cudaArrayCubemap = 0x04;

		/// <summary>
		/// (Runtime API) Must be set in cudaMallocArray or cudaMalloc3DArray in order to
		/// perform texture gather operations on the CUDA array.
		/// </summary>
		public const int cudaArrayTextureGather = 0x08;


		/// <summary>
		/// (Runtime API) Automatically enable peer access between remote devices as needed.
		/// </summary>
		public const int cudaIpcMemLazyEnablePeerAccess = 0x01;


		/// <summary>
		/// (Runtime API) Memory can be accessed by any stream on any device.
		/// </summary>
		public const int cudaMemAttachGlobal = 0x01;

		/// <summary>
		/// (Runtime API) Memory cannot be accessed by any stream on any device.
		/// </summary>
		public const int cudaMemAttachHost = 0x02;

		/// <summary>
		/// (Runtime API) Memory can only be accessed by a single stream on the associated device.
		/// </summary>
		public const int cudaMemAttachSingle = 0x04;


		/// <summary>
		/// (Runtime API) Default behavior.
		/// </summary>
		public const int cudaOccupancyDefault = 0x00;

		/// <summary>
		/// (Runtime API) Assume global caching is enabled and cannot be automatically turned off.
		/// </summary>
		public const int cudaOccupancyDisableCachingOverride = 0x01;


		/// <summary>
		/// (Runtime API) Device id that represents the CPU.
		/// </summary>
		public const int cudaCpuDeviceId = ((int)-1);

		/// <summary>
		/// (Runtime API) Device id that represents an invalid device.
		/// </summary>
		public const int cudaInvalidDeviceId = ((int)-2);
	}

	/// <summary>
	/// (Runtime API) CUDA Channel format descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaChannelFormatDesc {
		/// <summary>x</summary>
		public int x;
		/// <summary>y</summary>
		public int y;
		/// <summary>z</summary>
		public int z;
		/// <summary>w</summary>
		public int w;
		/// <summary>Channel format kind</summary>
		public cudaChannelFormatKind f;
	}

	/// <summary>
	/// (Runtime API) CUDA device properties.
	/// </summary>
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public struct cudaDeviceProp {
		/// <summary>
		/// ASCII string identifying device
		/// </summary>
		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
		public string name;

		/// <summary>
		/// 16-byte unique identifier
		/// </summary>
		public cudaUUID_t uuid;

		/// <summary>
		/// 8-byte locally unique identifier.
		/// Value is undefined on TCC and non-Windows platforms
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
		public byte[] luid;

		/// <summary>
		/// LUID device node mask.
		/// Value is undefined on TCC and non-Windows platforms
		/// </summary>
		public uint luidDeviceNodeMask;

		/// <summary>
		/// Global memory available on device in bytes
		/// </summary>
		public size_t totalGlobalMem;

		/// <summary>
		/// Shared memory available per block in bytes
		/// </summary>
		public size_t sharedMemPerBlock;

		/// <summary>
		/// 32-bit registers available per block
		/// </summary>
		public int regsPerBlock;

		/// <summary>
		/// Warp size in threads
		/// </summary>
		public int warpSize;

		/// <summary>
		/// Maximum pitch in bytes allowed by memory copies
		/// </summary>
		public size_t memPitch;

		/// <summary>
		/// Maximum number of threads per block
		/// </summary>
		public int maxThreadsPerBlock;

		/// <summary>
		/// Maximum size of each dimension of a block
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxThreadsDim;

		/// <summary>
		/// Maximum size of each dimension of a grid
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxGridSize;

		/// <summary>
		/// Clock frequency in kilohertz
		/// </summary>
		public int clockRate;

		/// <summary>
		/// Constant memory available on device in bytes
		/// </summary>
		public size_t totalConstMem;

		/// <summary>
		/// Major compute capability
		/// </summary>
		public int major;

		/// <summary>
		/// Minor compute capability
		/// </summary>
		public int minor;

		/// <summary>
		/// Alignment requirement for textures
		/// </summary>
		public size_t textureAlignment;

		/// <summary>
		/// Pitch alignment requirement for texture references bound to pitched memory
		/// </summary>
		public size_t texturePitchAlignment;

		/// <summary>
		/// Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount
		/// </summary>
		public int deviceOverlap;

		/// <summary>
		/// Number of multiprocessors on device
		/// </summary>
		public int multiProcessorCount;

		/// <summary>
		/// Specified whether there is a run time limit on kernels
		/// </summary>
		public int kernelExecTimeoutEnabled;

		/// <summary>
		/// Device is integrated as opposed to discrete
		/// </summary>
		public int integrated;

		/// <summary>
		/// Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
		/// </summary>
		public int canMapHostMemory;

		/// <summary>
		/// Compute mode (See ::cudaComputeMode)
		/// </summary>
		public int computeMode;

		/// <summary>
		/// Maximum 1D texture size
		/// </summary>
		public int maxTexture1D;

		/// <summary>
		/// Maximum 1D mipmapped texture size
		/// </summary>
		public int maxTexture1DMipmap;

		/// <summary>
		/// Maximum size for 1D textures bound to linear memory
		/// </summary>
		public int maxTexture1DLinear;

		/// <summary>
		/// Maximum 2D texture dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxTexture2D;

		/// <summary>
		/// Maximum 2D mipmapped texture dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxTexture2DMipmap;

		/// <summary>
		/// Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxTexture2DLinear;

		/// <summary>
		/// Maximum 2D texture dimensions if texture gather operations have to be performed
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxTexture2DGather;

		/// <summary>
		/// Maximum 3D texture dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxTexture3D;

		/// <summary>
		/// Maximum alternate 3D texture dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxTexture3DAlt;

		/// <summary>
		/// Maximum Cubemap texture dimensions
		/// </summary>
		public int maxTextureCubemap;

		/// <summary>
		/// Maximum 1D layered texture dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxTexture1DLayered;

		/// <summary>
		/// Maximum 2D layered texture dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxTexture2DLayered;

		/// <summary>
		/// Maximum Cubemap layered texture dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxTextureCubemapLayered;

		/// <summary>
		/// Maximum 1D surface size
		/// </summary>
		public int maxSurface1D;

		/// <summary>
		/// Maximum 2D surface dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxSurface2D;

		/// <summary>
		/// Maximum 3D surface dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxSurface3D;

		/// <summary>
		/// Maximum 1D layered surface dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxSurface1DLayered;

		/// <summary>
		/// Maximum 2D layered surface dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		public int[] maxSurface2DLayered;

		/// <summary>
		/// Maximum Cubemap surface dimensions
		/// </summary>
		public int maxSurfaceCubemap;

		/// <summary>
		/// Maximum Cubemap layered surface dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		public int[] maxSurfaceCubemapLayered;

		/// <summary>
		/// Alignment requirements for surfaces
		/// </summary>
		public size_t surfaceAlignment;

		/// <summary>
		/// Device can possibly execute multiple kernels concurrently
		/// </summary>
		public int concurrentKernels;

		/// <summary>
		/// Device has ECC support enabled
		/// </summary>
		public int ECCEnabled;

		/// <summary>
		/// PCI bus ID of the device
		/// </summary>
		public int pciBusID;

		/// <summary>
		/// PCI device ID of the device
		/// </summary>
		public int pciDeviceID;

		/// <summary>
		/// PCI domain ID of the device
		/// </summary>
		public int pciDomainID;

		/// <summary>
		/// 1 if device is a Tesla device using TCC driver, 0 otherwise
		/// </summary>
		public int tccDriver;

		/// <summary>
		/// Number of asynchronous engines
		/// </summary>
		public int asyncEngineCount;

		/// <summary>
		/// Device shares a unified address space with the host
		/// </summary>
		public int unifiedAddressing;

		/// <summary>
		/// Peak memory clock frequency in kilohertz
		/// </summary>
		public int memoryClockRate;

		/// <summary>
		/// Global memory bus width in bits
		/// </summary>
		public int memoryBusWidth;

		/// <summary>
		/// Size of L2 cache in bytes
		/// </summary>
		public int l2CacheSize;

		/// <summary>
		/// Maximum resident threads per multiprocessor
		/// </summary>
		public int maxThreadsPerMultiProcessor;

		/// <summary>
		/// Device supports stream priorities
		/// </summary>
		public int streamPrioritiesSupported;

		/// <summary>
		/// Device supports caching globals in L1
		/// </summary>
		public int globalL1CacheSupported;

		/// <summary>
		/// Device supports caching locals in L1
		/// </summary>
		public int localL1CacheSupported;

		/// <summary>
		/// Shared memory available per multiprocessor in bytes
		/// </summary>
		public size_t sharedMemPerMultiprocessor;

		/// <summary>
		/// 32-bit registers available per multiprocessor
		/// </summary>
		public int regsPerMultiprocessor;

		/// <summary>
		/// Device supports allocating managed memory on this system
		/// </summary>
		public int managedMemory;

		/// <summary>
		/// Device is on a multi-GPU board
		/// </summary>
		public int isMultiGpuBoard;

		/// <summary>
		/// Unique identifier for a group of devices on the same multi-GPU board
		/// </summary>
		public int multiGpuBoardGroupID;

		/// <summary>
		/// Link between the device and the host supports native atomic operations
		/// </summary>
		public int hostNativeAtomicSupported;

		/// <summary>
		/// Ratio of single precision performance (in floating-point operations per second) to double precision performance
		/// </summary>
		public int singleToDoublePrecisionPerfRatio;

		/// <summary>
		/// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
		/// </summary>
		public int pageableMemoryAccess;

		/// <summary>
		/// Device can coherently access managed memory concurrently with the CPU
		/// </summary>
		public int concurrentManagedAccess;

		/// <summary>
		/// Device supports Compute Preemption
		/// </summary>
		public int computePreemptionSupported;

		/// <summary>
		/// Device can access host registered memory at the same virtual address as the CPU
		/// </summary>
		public int canUseHostPointerForRegisteredMem;

		/// <summary>
		/// Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel
		/// </summary>
		public int cooperativeLaunch;

		/// <summary>
		/// Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice
		/// </summary>
		public int cooperativeMultiDeviceLaunch;

		/// <summary>
		/// Per device maximum shared memory per block usable by special opt in
		/// </summary>
		public size_t sharedMemPerBlockOptin;

		/// <summary>
		/// Device accesses pageable memory via the host's page tables
		/// </summary>
		public int pageableMemoryAccessUsesHostPageTables;

		/// <summary>
		/// Host can directly access managed memory on the device without migration.
		/// </summary>
		public int directManagedMemAccessFromHost;
	}

	/// <summary>
	/// (Runtime API) 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaEglFrame {
		public uint width;
		public uint height;
		public uint depth;
		public uint pitch;
		public uint numChannels;
		public cudaChannelFormatDesc channelDesc;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
		public uint[] reserved;
	}

	/// <summary>
	/// (Runtime API) 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaEglPlaneDesc {
	}

	/// <summary>
	/// (Runtime API) CUDA function attributes.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaFuncAttributes {
		/// <summary>
		/// The size in bytes of statically-allocated shared memory per block
		/// required by this function.This does not include dynamically-allocated
		/// shared memory requested by the user at runtime.
		/// </summary>
		public size_t sharedSizeBytes;

		/// <summary>
		/// The size in bytes of user-allocated constant memory required by this
		/// function.
		/// </summary>
		public size_t constSizeBytes;

		/// <summary>
		/// The size in bytes of local memory used by each thread of this function.
		/// </summary>
		public size_t localSizeBytes;

		/// <summary>
		/// The maximum number of threads per block, beyond which a launch of the
		/// function would fail.This number depends on both the function and the
		/// device on which the function is currently loaded.
		/// </summary>
		public int maxThreadsPerBlock;

		/// <summary>
		/// The number of registers used by each thread of this function.
		/// </summary>
		public int numRegs;

		/// <summary>
		/// The PTX virtual architecture version for which the function was
		/// compiled.This value is the major PTX version * 10 + the minor PTX
		/// version, so a PTX version 1.3 function would return the value 13.
		/// </summary>
		public int ptxVersion;

		/// <summary>
		/// The binary architecture version for which the function was compiled.
		/// This value is the major binary version * 10 + the minor binary version,
		/// so a binary version 1.3 function would return the value 13.
		/// </summary>
		public int binaryVersion;

		/// <summary>
		/// The attribute to indicate whether the function has been compiled with
		/// user specified option "-Xptxas --dlcm=ca" set.
		/// </summary>
		public int cacheModeCA;
	}

	/// <summary>
	/// (Runtime API) CUDA host node parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaHostNodeParams {

	}

	/// <summary>
	/// (Runtime API) CUDA IPC event handle.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaIpcEventHandle {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = Defines.CUDA_IPC_HANDLE_SIZE)]
		public byte[] reserved;
	}

	/// <summary>
	/// (Runtime API) CUDA IPC memory handle.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaIpcMemHandle {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = Defines.CUDA_IPC_HANDLE_SIZE)]
		public byte[] reserved;
	}

	/// <summary>
	/// (Runtime API) CUDA GPU kernel node parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaKernelNodeParams {

	}

	/// <summary>
	/// (Runtime API) CUDA launch parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaLaunchParams {
		/// <summary>
		/// Device function symbol.
		/// </summary>
		public IntPtr func; // void*
		/// <summary>
		/// Grid dimentions.
		/// </summary>
		public dim3 gridDim;
		/// <summary>
		/// Block dimentions.
		/// </summary>
		public dim3 blockDim;
		/// <summary>
		/// Arguments.
		/// </summary>
		public IntPtr args; // void**
		/// <summary>
		/// Shared memory.
		/// </summary>
		public size_t sharedMem;
		/// <summary>
		/// Stream identifier.
		/// </summary>
		public cudaStream_t stream;
	}

	/// <summary>
	/// (Runtime API) CUDA 3D memory copying parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaMemcpy3DParms {
		/// <summary>
		/// Source memory address
		/// </summary>
		public cudaArray_t srcArray;

		/// <summary>
		/// Source position offset
		/// </summary>
		public cudaPos srcPos;

		/// <summary>
		/// Pitched source memory address
		/// </summary>
		public cudaPitchedPtr srcPtr;

		/// <summary>
		/// Destination memory address
		/// </summary>
		public cudaArray_t dstArray;

		/// <summary>
		/// Destination position offset
		/// </summary>
		public cudaPos dstPos;

		/// <summary>
		/// Pitched destination memory address
		/// </summary>
		public cudaPitchedPtr  dstPtr;

		/// <summary>
		/// Requested memory copy size
		/// </summary>
		public cudaExtent      extent;

		/// <summary>
		/// Type of transfer
		/// </summary>
		public cudaMemcpyKind    kind;
	}

	/// <summary>
	/// (Runtime API) CUDA 3D cross-device memory copying parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaMemcpy3DPeerParms {
		/// <summary>
		/// Source memory address
		/// </summary>
		public cudaArray_t srcArray;

		/// <summary>
		/// Source position offset
		/// </summary>
		public cudaPos srcPos;

		/// <summary>
		/// Pitched source memory address
		/// </summary>
		public cudaPitchedPtr srcPtr;

		/// <summary>
		/// Source device
		/// </summary>
		public int srcDevice;

		/// <summary>
		/// Destination memory address
		/// </summary>
		public cudaArray_t dstArray;

		/// <summary>
		/// Destination position offset
		/// </summary>
		public cudaPos dstPos;

		/// <summary>
		/// Pitched destination memory address
		/// </summary>
		public cudaPitchedPtr dstPtr;

		/// <summary>
		/// Destination device
		/// </summary>
		public int dstDevice;

		/// <summary>
		/// Requested memory copy size
		/// </summary>
		public cudaExtent extent;
	}

	/// <summary>
	/// (Runtime API) CUDA Memset node parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaMemsetParams {

	}

	/// <summary>
	/// (Runtime API) CUDA pointer attributes.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaPointerAttributes {
		/// <summary>
		/// The physical location of the memory, ::cudaMemoryTypeHost or 
		/// ::cudaMemoryTypeDevice.
		/// </summary>
		public cudaMemoryType memoryType;

		/// <summary>
		/// The device against which the memory was allocated or registered.
		/// If the memory type is ::cudaMemoryTypeDevice then this identifies
		/// the device on which the memory referred physically resides.  If
		/// the memory type is ::cudaMemoryTypeHost then this identifies the 
		/// device which was current when the memory was allocated or registered
		/// (and if that device is deinitialized then this allocation will vanish
		/// with that device's state).
		/// </summary>
		public int device;

		/// <summary>
		/// The address which may be dereferenced on the current device to access 
		/// the memory or NULL if no such address exists.
		/// </summary>
		public IntPtr devicePointer;

		/// <summary>
		/// The address which may be dereferenced on the host to access the
		/// memory or NULL if no such address exists.
		/// </summary>
		public IntPtr hostPointer;

		/// <summary>
		/// Indicates if this pointer points to managed memory
		/// </summary>
		public int isManaged;
	}

	/// <summary>
	/// (Runtime API) CUDA Pitched memory pointer.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaPitchedPtr {
		/// <summary>
		/// Pointer to allocated memory
		/// </summary>
		public IntPtr ptr;
		/// <summary>
		/// Pitch of allocated memory in bytes
		/// </summary>
		public size_t pitch;
		/// <summary>
		/// Logical width of allocation in elements
		/// </summary>
		public size_t xsize;
		/// <summary>
		/// Logical height of allocation in elements
		/// </summary>
		public size_t ysize;
	}

	/// <summary>
	/// (Runtime API) CUDA extent.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaExtent {
		/// <summary>
		/// Width in elements when referring to array memory, in bytes when referring to linear memory
		/// </summary>
		public size_t width;
		/// <summary>
		/// Height in elements
		/// </summary>
		public size_t height;
		/// <summary>
		/// Depth in elements
		/// </summary>
		public size_t depth;
	}

	/// <summary>
	/// (Runtime API) External memory buffer descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaExternalMemoryBufferDesc {
		/// <summary>
		/// Offset into the memory object where the buffer's base is.
		/// </summary>
		public UInt64 offset;
		/// <summary>
		/// Size of the buffer.
		/// </summary>
		public UInt64 size;
		/// <summary>
		/// Flags reserved for future use. Must be zero.
		/// </summary>
		public uint flags;
	}

	/// <summary>
	/// (Runtime API) External memory handle descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaExternalMemoryHandleDesc {
		/// <summary>
		/// Type of the handle.
		/// </summary>
		cudaExternalMemoryHandleType type;
		/// <summary>
		/// (Runtime API) External memory handle descriptor.
		/// </summary>
		[StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi)]
		public struct Handle {
			/// <summary>
			/// File descriptor referencing the memory object.
			/// Valid when type is cudaExternalMemoryHandleTypeOpaqueFd.
			/// </summary>
			[FieldOffset(0)]
			public int fd;
			/// <summary>
			/// Valid NT handle.
			/// Must be NULL if 'name' is non-NULL.
			/// </summary>
			[FieldOffset(0)]
			public IntPtr handle;
			/// <summary>
			/// Name of a valid memory object.
			/// Must be NULL if 'handle' is non-NULL.
			/// </summary>
			[FieldOffset(8)]
			[MarshalAs(UnmanagedType.LPStr)]
			public string name;
		}
		public Handle handle;
		/// <summary>
		/// Size of the memory allocation.
		/// </summary>
		public UInt64 size;
		/// <summary>
		/// Flags must either be zero or cudaExternalMemoryDedicated.
		/// </summary>
		public uint flags;
	}

	/// <summary>
	/// (Runtime API) External memory mipmap descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaExternalMemoryMipmappedArrayDesc {
		/// <summary>
		/// Offset into the memory object where the base level of the mipmap chain is.
		/// </summary>
		public UInt64 offset;
		/// <summary>
		/// Format of base level of the mipmap chain.
		/// </summary>
		public cudaChannelFormatDesc formatDesc;
		/// <summary>
		/// Dimensions of base level of the mipmap chain.
		/// </summary>
		public cudaExtent extent;
		/// <summary>
		/// Flags associated with CUDA mipmapped arrays. See cudaMallocMipmappedArray.
		/// </summary>
		public uint flags;
		/// <summary>
		/// Total number of levels in the mipmap chain.
		/// </summary>
		public uint numLevels;
	}

	/// <summary>
	/// (Runtime API) External semaphore handle descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaExternalSemaphoreHandleDesc {
		/// <summary>
		/// Type of the handle.
		/// </summary>
		cudaExternalSemaphoreHandleType type;
		/// <summary>
		/// (Runtime API) External memory handle descriptor.
		/// </summary>
		[StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi)]
		public struct Handle {
			/// <summary>
			/// File descriptor referencing the semaphore object.
			/// Valid when type is cudaExternalSemaphoreHandleTypeOpaqueFd.
			/// </summary>
			[FieldOffset(0)]
			public int fd;
			/// <summary>
			/// Valid NT handle.
			/// Must be NULL if 'name' is non-NULL.
			/// </summary>
			[FieldOffset(0)]
			public IntPtr handle;
			/// <summary>
			/// Name of a valid synchronization primitive.
			/// Must be NULL if 'handle' is non-NULL.
			/// </summary>
			[FieldOffset(8)]
			[MarshalAs(UnmanagedType.LPStr)]
			public string name;
		}
		public Handle handle;
		/// <summary>
		/// Flags reserved for the future. Must be zero.
		/// </summary>
		public uint flags;
	}

	/// <summary>
	/// (Runtime API) External semaphore signal parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaExternalSemaphoreSignalParams {

	}

	/// <summary>
	/// (Runtime API) External semaphore wait parameters.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaExternalSemaphoreWaitParams {

	}

	/// <summary>
	/// (Runtime API) CUDA 3D position.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaPos {
		/// <summary>x</summary>
		public size_t x;
		/// <summary>y</summary>
		public size_t y;
		/// <summary>z</summary>
		public size_t z;
	}

	/// <summary>
	/// (Runtime API) CUDA resource descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaResourceDesc {
		/// <summary>
		/// Resource type
		/// </summary>
		public cudaResourceType resType;

		/// <summary>
		/// (Runtime API) CUDA resource descriptor.
		/// </summary>
		[StructLayout(LayoutKind.Explicit)]
		public struct Resource {
			[FieldOffset(0)]
			public cudaArray_t array;

			[FieldOffset(0)]
			public cudaMipmappedArray_t mipmap;

			/// <summary>
			/// Device pointer
			/// </summary>
			[FieldOffset(0)]
			IntPtr devPtr;

			/// <summary>
			/// Channel descriptor
			/// </summary>
			[FieldOffset(8)]
			cudaChannelFormatDesc desc;

			/// <summary>
			/// Size in bytes
			/// </summary>
			[FieldOffset(28)]
			size_t sizeInBytes;

			/// <summary>
			/// Width of the array in elements
			/// </summary>
			[FieldOffset(36)]
			size_t width;

			/// <summary>
			/// Height of the array in elements
			/// </summary>
			[FieldOffset(44)]
			size_t height;

			/// <summary>
			/// Pitch between two rows in bytes
			/// </summary>
			[FieldOffset(52)]
			size_t pitchInBytes;
		}
		public Resource res;
	}

	/// <summary>
	/// (Runtime API) CUDA resource view descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaResourceViewDesc {
		/// <summary>
		/// Resource view format
		/// </summary>
		public cudaResourceViewFormat format;

		/// <summary>
		/// Width of the resource view
		/// </summary>
		public size_t width;

		/// <summary>
		/// Height of the resource view
		/// </summary>
		public size_t height;

		/// <summary>
		/// Depth of the resource view
		/// </summary>
		public size_t depth;

		/// <summary>
		/// First defined mipmap level
		/// </summary>
		public uint firstMipmapLevel;

		/// <summary>
		/// Last defined mipmap level
		/// </summary>
		public uint lastMipmapLevel;

		/// <summary>
		/// First layer index
		/// </summary>
		public uint firstLayer;

		/// <summary>
		/// Last layer index
		/// </summary>
		public uint lastLayer;
	}

	/// <summary>
	/// (Runtime API) CUDA texture reference.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct textureReference {
		/// <summary>
		/// Indicates whether texture reads are normalized or not
		/// </summary>
		int normalized;

		/// <summary>
		/// Texture filter mode
		/// </summary>
		cudaTextureFilterMode filterMode;

		/// <summary>
		/// Texture address mode for up to 3 dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		cudaTextureAddressMode[] addressMode;

		/// <summary>
		/// Channel descriptor for the texture reference
		/// </summary>
		cudaChannelFormatDesc channelDesc;

		/// <summary>
		/// Perform sRGB->linear conversion during texture read
		/// </summary>
		int sRGB;

		/// <summary>
		/// Limit to the anisotropy ratio
		/// </summary>
		uint maxAnisotropy;

		/// <summary>
		/// Mipmap filter mode
		/// </summary>
		cudaTextureFilterMode mipmapFilterMode;

		/// <summary>
		/// Offset applied to the supplied mipmap level
		/// </summary>
		float mipmapLevelBias;

		/// <summary>
		/// Lower end of the mipmap level range to clamp access to
		/// </summary>
		float minMipmapLevelClamp;

		/// <summary>
		/// Upper end of the mipmap level range to clamp access to
		/// </summary>
		float maxMipmapLevelClamp;

		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 15)]
		int[] __cudaReserved;
	}

	/// <summary>
	/// (Runtime API) CUDA texture descriptor.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaTextureDesc {
		/// <summary>
		/// Texture address mode for up to 3 dimensions
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		cudaTextureAddressMode[] addressMode;

		/// <summary>
		/// Texture filter mode
		/// </summary>
		cudaTextureFilterMode filterMode;

		/// <summary>
		/// Texture read mode
		/// </summary>
		cudaTextureReadMode readMode;

		/// <summary>
		/// Perform sRGB->linear conversion during texture read
		/// </summary>
		int sRGB;

		/// <summary>
		/// Texture Border Color
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
		float[] borderColor;

		/// <summary>
		/// Indicates whether texture reads are normalized or not
		/// </summary>
		int normalizedCoords;

		/// <summary>
		/// Limit to the anisotropy ratio
		/// </summary>
		uint maxAnisotropy;

		/// <summary>
		/// Mipmap filter mode
		/// </summary>
		cudaTextureFilterMode mipmapFilterMode;

		/// <summary>
		/// Offset applied to the supplied mipmap level
		/// </summary>
		float mipmapLevelBias;

		/// <summary>
		/// Lower end of the mipmap level range to clamp access to
		/// </summary>
		float minMipmapLevelClamp;

		/// <summary>
		/// Upper end of the mipmap level range to clamp access to
		/// </summary>
		float maxMipmapLevelClamp;
	}

	/// <summary>
	/// (Runtime API) CUDA UUID types.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUuuid_st {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
		public byte[] bytes;
	}

	/// <summary>
	/// (Runtime API) CUDA Surface reference.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct surfaceReference {
		/// <summary>
		/// Channel descriptor for surface reference
		/// </summary>
		cudaChannelFormatDesc channelDesc;
	}


	/// <summary>
	/// (Runtime API) CUDA cooperative group scope.
	/// </summary>
	public enum cudaCGScope {
		/// <summary>
		/// Invalid cooperative group scope.
		/// </summary>
		cudaCGScopeInvalid = 0,

		/// <summary>
		/// Scope represented by a grid_group.
		/// </summary>
		cudaCGScopeGrid = 1,

		/// <summary>
		/// Scope represented by a multi_grid_group.
		/// </summary>
		cudaCGScopeMultiGrid = 2
	}

	/// <summary>
	/// (Runtime API) Channel format kind.
	/// </summary>
	public enum cudaChannelFormatKind {
		/// <summary>
		/// Signed channel format.
		/// </summary>
		cudaChannelFormatKindSigned = 0,

		/// <summary>
		/// Unsigned channel format.
		/// </summary>
		cudaChannelFormatKindUnsigned = 1,

		/// <summary>
		/// Float channel format.
		/// </summary>
		cudaChannelFormatKindFloat = 2,

		/// <summary>
		/// No channel format.
		/// </summary>
		cudaChannelFormatKindNone = 3
	}

	/// <summary>
	/// (Runtime API) CUDA device compute modes.
	/// </summary>
	public enum cudaComputeMode {
		/// <summary>
		/// Default compute mode (Multiple threads can use cudaSetDevice() with this device)
		/// </summary>
		cudaComputeModeDefault = 0,

		/// <summary>
		/// Compute-exclusive-thread mode (Only one thread in one process will be able to use cudaSetDevice() with this device)
		/// </summary>
		cudaComputeModeExclusive = 1,

		/// <summary>
		/// Compute-prohibited mode (No threads can use cudaSetDevice() with this device)
		/// </summary>
		cudaComputeModeProhibited = 2,

		/// <summary>
		/// Compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice() with this device)
		/// </summary>
		cudaComputeModeExclusiveProcess = 3
	}

	/// <summary>
	/// (Runtime API) CUDA device attributes.
	/// </summary>
	public enum cudaDeviceAttr {
		/// <summary>
		/// Maximum number of threads per block
		/// </summary>
		cudaDevAttrMaxThreadsPerBlock = 1,

		/// <summary>
		/// Maximum block dimension X
		/// </summary>
		cudaDevAttrMaxBlockDimX = 2,

		/// <summary>
		/// Maximum block dimension Y
		/// </summary>
		cudaDevAttrMaxBlockDimY = 3,

		/// <summary>
		/// Maximum block dimension Z
		/// </summary>
		cudaDevAttrMaxBlockDimZ = 4,

		/// <summary>
		/// Maximum grid dimension X
		/// </summary>
		cudaDevAttrMaxGridDimX = 5,

		/// <summary>
		/// Maximum grid dimension Y
		/// </summary>
		cudaDevAttrMaxGridDimY = 6,

		/// <summary>
		/// Maximum grid dimension Z
		/// </summary>
		cudaDevAttrMaxGridDimZ = 7,

		/// <summary>
		/// Maximum shared memory available per block in bytes
		/// </summary>
		cudaDevAttrMaxSharedMemoryPerBlock = 8,

		/// <summary>
		/// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
		/// </summary>
		cudaDevAttrTotalConstantMemory = 9,

		/// <summary>
		/// Warp size in threads
		/// </summary>
		cudaDevAttrWarpSize = 10,

		/// <summary>
		/// Maximum pitch in bytes allowed by memory copies
		/// </summary>
		cudaDevAttrMaxPitch = 11,

		/// <summary>
		/// Maximum number of 32-bit registers available per block
		/// </summary>
		cudaDevAttrMaxRegistersPerBlock = 12,

		/// <summary>
		/// Peak clock frequency in kilohertz
		/// </summary>
		cudaDevAttrClockRate = 13,

		/// <summary>
		/// Alignment requirement for textures
		/// </summary>
		cudaDevAttrTextureAlignment = 14,

		/// <summary>
		/// Device can possibly copy memory and execute a kernel concurrently
		/// </summary>
		cudaDevAttrGpuOverlap = 15,

		/// <summary>
		/// Number of multiprocessors on device
		/// </summary>
		cudaDevAttrMultiProcessorCount = 16,

		/// <summary>
		/// Specifies whether there is a run time limit on kernels
		/// </summary>
		cudaDevAttrKernelExecTimeout = 17,

		/// <summary>
		/// Device is integrated with host memory
		/// </summary>
		cudaDevAttrIntegrated = 18,

		/// <summary>
		/// Device can map host memory into CUDA address space
		/// </summary>
		cudaDevAttrCanMapHostMemory = 19,

		/// <summary>
		/// Compute mode (See cudaComputeMode for details)
		/// </summary>
		cudaDevAttrComputeMode = 20,

		/// <summary>
		/// Maximum 1D texture width
		/// </summary>
		cudaDevAttrMaxTexture1DWidth = 21,

		/// <summary>
		/// Maximum 2D texture width
		/// </summary>
		cudaDevAttrMaxTexture2DWidth = 22,

		/// <summary>
		/// Maximum 2D texture height
		/// </summary>
		cudaDevAttrMaxTexture2DHeight = 23,

		/// <summary>
		/// Maximum 3D texture width
		/// </summary>
		cudaDevAttrMaxTexture3DWidth = 24,

		/// <summary>
		/// Maximum 3D texture height
		/// </summary>
		cudaDevAttrMaxTexture3DHeight = 25,

		/// <summary>
		/// Maximum 3D texture depth
		/// </summary>
		cudaDevAttrMaxTexture3DDepth = 26,

		/// <summary>
		/// Maximum 2D layered texture width
		/// </summary>
		cudaDevAttrMaxTexture2DLayeredWidth = 27,

		/// <summary>
		/// Maximum 2D layered texture height
		/// </summary>
		cudaDevAttrMaxTexture2DLayeredHeight = 28,

		/// <summary>
		/// Maximum layers in a 2D layered texture
		/// </summary>
		cudaDevAttrMaxTexture2DLayeredLayers = 29,

		/// <summary>
		/// Alignment requirement for surfaces
		/// </summary>
		cudaDevAttrSurfaceAlignment = 30,

		/// <summary>
		/// Device can possibly execute multiple kernels concurrently
		/// </summary>
		cudaDevAttrConcurrentKernels = 31,

		/// <summary>
		/// Device has ECC support enabled
		/// </summary>
		cudaDevAttrEccEnabled = 32,

		/// <summary>
		/// PCI bus ID of the device
		/// </summary>
		cudaDevAttrPciBusId = 33,

		/// <summary>
		/// PCI device ID of the device
		/// </summary>
		cudaDevAttrPciDeviceId = 34,

		/// <summary>
		/// Device is using TCC driver model
		/// </summary>
		cudaDevAttrTccDriver = 35,

		/// <summary>
		/// Peak memory clock frequency in kilohertz
		/// </summary>
		cudaDevAttrMemoryClockRate = 36,

		/// <summary>
		/// Global memory bus width in bits
		/// </summary>
		cudaDevAttrGlobalMemoryBusWidth = 37,

		/// <summary>
		/// Size of L2 cache in bytes
		/// </summary>
		cudaDevAttrL2CacheSize = 38,

		/// <summary>
		/// Maximum resident threads per multiprocessor
		/// </summary>
		cudaDevAttrMaxThreadsPerMultiProcessor = 39,

		/// <summary>
		/// Number of asynchronous engines
		/// </summary>
		cudaDevAttrAsyncEngineCount = 40,

		/// <summary>
		/// Device shares a unified address space with the host
		/// </summary>
		cudaDevAttrUnifiedAddressing = 41,

		/// <summary>
		/// Maximum 1D layered texture width
		/// </summary>
		cudaDevAttrMaxTexture1DLayeredWidth = 42,

		/// <summary>
		/// Maximum layers in a 1D layered texture
		/// </summary>
		cudaDevAttrMaxTexture1DLayeredLayers = 43,

		/// <summary>
		/// Maximum 2D texture width if cudaArrayTextureGather is set
		/// </summary>
		cudaDevAttrMaxTexture2DGatherWidth = 45,

		/// <summary>
		/// Maximum 2D texture height if cudaArrayTextureGather is set
		/// </summary>
		cudaDevAttrMaxTexture2DGatherHeight = 46,

		/// <summary>
		/// Alternate maximum 3D texture width
		/// </summary>
		cudaDevAttrMaxTexture3DWidthAlt = 47,

		/// <summary>
		/// Alternate maximum 3D texture height
		/// </summary>
		cudaDevAttrMaxTexture3DHeightAlt = 48,

		/// <summary>
		/// Alternate maximum 3D texture depth
		/// </summary>
		cudaDevAttrMaxTexture3DDepthAlt = 49,

		/// <summary>
		/// PCI domain ID of the device
		/// </summary>
		cudaDevAttrPciDomainId = 50,

		/// <summary>
		/// Pitch alignment requirement for textures
		/// </summary>
		cudaDevAttrTexturePitchAlignment = 51,

		/// <summary>
		/// Maximum cubemap texture width/height
		/// </summary>
		cudaDevAttrMaxTextureCubemapWidth = 52,

		/// <summary>
		/// Maximum cubemap layered texture width/height
		/// </summary>
		cudaDevAttrMaxTextureCubemapLayeredWidth = 53,

		/// <summary>
		/// Maximum layers in a cubemap layered texture
		/// </summary>
		cudaDevAttrMaxTextureCubemapLayeredLayers = 54,

		/// <summary>
		/// Maximum 1D surface width
		/// </summary>
		cudaDevAttrMaxSurface1DWidth = 55,

		/// <summary>
		/// Maximum 2D surface width
		/// </summary>
		cudaDevAttrMaxSurface2DWidth = 56,

		/// <summary>
		/// Maximum 2D surface height
		/// </summary>
		cudaDevAttrMaxSurface2DHeight = 57,

		/// <summary>
		/// Maximum 3D surface width
		/// </summary>
		cudaDevAttrMaxSurface3DWidth = 58,

		/// <summary>
		/// Maximum 3D surface height
		/// </summary>
		cudaDevAttrMaxSurface3DHeight = 59,

		/// <summary>
		/// Maximum 3D surface depth
		/// </summary>
		cudaDevAttrMaxSurface3DDepth = 60,

		/// <summary>
		/// Maximum 1D layered surface width
		/// </summary>
		cudaDevAttrMaxSurface1DLayeredWidth = 61,

		/// <summary>
		/// Maximum layers in a 1D layered surface
		/// </summary>
		cudaDevAttrMaxSurface1DLayeredLayers = 62,

		/// <summary>
		/// Maximum 2D layered surface width
		/// </summary>
		cudaDevAttrMaxSurface2DLayeredWidth = 63,

		/// <summary>
		/// Maximum 2D layered surface height
		/// </summary>
		cudaDevAttrMaxSurface2DLayeredHeight = 64,

		/// <summary>
		/// Maximum layers in a 2D layered surface
		/// </summary>
		cudaDevAttrMaxSurface2DLayeredLayers = 65,

		/// <summary>
		/// Maximum cubemap surface width
		/// </summary>
		cudaDevAttrMaxSurfaceCubemapWidth = 66,

		/// <summary>
		/// Maximum cubemap layered surface width
		/// </summary>
		cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,

		/// <summary>
		/// Maximum layers in a cubemap layered surface
		/// </summary>
		cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,

		/// <summary>
		/// Maximum 1D linear texture width
		/// </summary>
		cudaDevAttrMaxTexture1DLinearWidth = 69,

		/// <summary>
		/// Maximum 2D linear texture width
		/// </summary>
		cudaDevAttrMaxTexture2DLinearWidth = 70,

		/// <summary>
		/// Maximum 2D linear texture height
		/// </summary>
		cudaDevAttrMaxTexture2DLinearHeight = 71,

		/// <summary>
		/// Maximum 2D linear texture pitch in bytes
		/// </summary>
		cudaDevAttrMaxTexture2DLinearPitch = 72,

		/// <summary>
		/// Maximum mipmapped 2D texture width
		/// </summary>
		cudaDevAttrMaxTexture2DMipmappedWidth = 73,

		/// <summary>
		/// Maximum mipmapped 2D texture height
		/// </summary>
		cudaDevAttrMaxTexture2DMipmappedHeight = 74,

		/// <summary>
		/// Major compute capability version number
		/// </summary>
		cudaDevAttrComputeCapabilityMajor = 75,

		/// <summary>
		/// Minor compute capability version number
		/// </summary>
		cudaDevAttrComputeCapabilityMinor = 76,

		/// <summary>
		/// Maximum mipmapped 1D texture width
		/// </summary>
		cudaDevAttrMaxTexture1DMipmappedWidth = 77,

		/// <summary>
		/// Device supports stream priorities
		/// </summary>
		cudaDevAttrStreamPrioritiesSupported = 78,

		/// <summary>
		/// Device supports caching globals in L1
		/// </summary>
		cudaDevAttrGlobalL1CacheSupported = 79,

		/// <summary>
		/// Device supports caching locals in L1
		/// </summary>
		cudaDevAttrLocalL1CacheSupported = 80,

		/// <summary>
		/// Maximum shared memory available per multiprocessor in bytes
		/// </summary>
		cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,

		/// <summary>
		/// Maximum number of 32-bit registers available per multiprocessor
		/// </summary>
		cudaDevAttrMaxRegistersPerMultiprocessor = 82,

		/// <summary>
		/// Device can allocate managed memory on this system
		/// </summary>
		cudaDevAttrManagedMemory = 83,

		/// <summary>
		/// Device is on a multi-GPU board
		/// </summary>
		cudaDevAttrIsMultiGpuBoard = 84,

		/// <summary>
		/// Unique identifier for a group of devices on the same multi-GPU board
		/// </summary>
		cudaDevAttrMultiGpuBoardGroupID = 85,

		/// <summary>
		/// Link between the device and the host supports native atomic operations
		/// </summary>
		cudaDevAttrHostNativeAtomicSupported = 86,

		/// <summary>
		/// Ratio of single precision performance (in floating-point operations per second) to double precision performance
		/// </summary>
		cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,

		/// <summary>
		/// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
		/// </summary>
		cudaDevAttrPageableMemoryAccess = 88,

		/// <summary>
		/// Device can coherently access managed memory concurrently with the CPU
		/// </summary>
		cudaDevAttrConcurrentManagedAccess = 89,

		/// <summary>
		/// Device supports Compute Preemption
		/// </summary>
		cudaDevAttrComputePreemptionSupported = 90,

		/// <summary>
		/// Device can access host registered memory at the same virtual address as the CPU
		/// </summary>
		cudaDevAttrCanUseHostPointerForRegisteredMem = 91,

		cudaDevAttrReserved92 = 92,
		cudaDevAttrReserved93 = 93,
		cudaDevAttrReserved94 = 94,

		/// <summary>
		/// Device supports launching cooperative kernels via cudaLaunchCooperativeKernel
		/// </summary>
		cudaDevAttrCooperativeLaunch = 95,

		/// <summary>
		/// Device can participate in cooperative kernels launched via cudaLaunchCooperativeKernelMultiDevice
		/// </summary>
		cudaDevAttrCooperativeMultiDeviceLaunch = 96,

		/// <summary>
		/// The maximum optin shared memory per block. This value may vary by chip. See cudaFuncSetAttribute
		/// </summary>
		cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,

		/// <summary>
		/// Device supports flushing of outstanding remote writes.
		/// </summary>
		cudaDevAttrCanFlushRemoteWrites = 98,

		/// <summary>
		/// Device supports host memory registration via cudaHostRegister.
		/// </summary>
		cudaDevAttrHostRegisterSupported = 99,

		/// <summary>
		/// Device accesses pageable memory via the host's page tables.
		/// </summary>
		cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,

		/// <summary>
		/// Host can directly access managed memory on the device without migration.
		/// </summary>
		cudaDevAttrDirectManagedMemAccessFromHost = 101
	}

	/// <summary>
	/// (Runtime API) CUDA device P2P attributes.
	/// </summary>
	public enum cudaDeviceP2PAttr {
		/// <summary>
		/// A relative value indicating the performance of the link between two devices
		/// </summary>
		cudaDevP2PAttrPerformanceRank = 1,

		/// <summary>
		/// Peer access is enabled
		/// </summary>
		cudaDevP2PAttrAccessSupported = 2,

		/// <summary>
		/// Native atomic operation over the link supported
		/// </summary>
		cudaDevP2PAttrNativeAtomicSupported = 3,

		/// <summary>
		/// Accessing CUDA arrays over the link supported
		/// </summary>
		cudaDevP2PAttrCudaArrayAccessSupported = 4
	}

	/// <summary>
	/// (Runtime API) CUDA EGL Color Format - The different planar and multiplanar formats currently supported for CUDA_EGL interops.
	/// </summary>
	public enum cudaEglColorFormat {
		/// <summary>
		/// Y, U, V in three surfaces, each in a separate surface, U/V width = 1 / 2 Y width, U/V height = 1 / 2 Y height.
		/// </summary>
		cudaEglColorFormatYUV420Planar = 0,

		/// <summary>
		/// Y, UV in two surfaces (UV as one surface), width, height ratio same as YUV420Planar.
		/// </summary>
		cudaEglColorFormatYUV420SemiPlanar = 1,

		/// <summary>
		/// Y, U, V each in a separate surface, U/V width = 1 / 2 Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYUV422Planar = 2,

		/// <summary>
		/// Y, UV in two surfaces, width, height ratio same as YUV422Planar.
		/// </summary>
		cudaEglColorFormatYUV422SemiPlanar = 3,

		/// <summary>
		/// R/G/B three channels in one surface with RGB byte ordering.
		/// </summary>
		cudaEglColorFormatRGB = 4,

		/// <summary>
		/// R/G/B three channels in one surface with BGR byte ordering.
		/// </summary>
		cudaEglColorFormatBGR = 5,

		/// <summary>
		/// R/G/B/A four channels in one surface with ARGB byte ordering.
		/// </summary>
		cudaEglColorFormatARGB = 6,

		/// <summary>
		/// R/G/B/A four channels in one surface with RGBA byte ordering.
		/// </summary>
		cudaEglColorFormatRGBA = 7,

		/// <summary>
		/// single luminance channel in one surface.
		/// </summary>
		cudaEglColorFormatL = 8,

		/// <summary>
		/// single color channel in one surface.
		/// </summary>
		cudaEglColorFormatR = 9,

		/// <summary>
		/// Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYUV444Planar = 10,

		/// <summary>
		/// Y, UV in two surfaces (UV as one surface), width, height ratio same as YUV444Planar.
		/// </summary>
		cudaEglColorFormatYUV444SemiPlanar = 11,

		/// <summary>
		/// Y, U, V in one surface, interleaved as YUYV.
		/// </summary>
		cudaEglColorFormatYUYV422 = 12,

		/// <summary>
		/// Y, U, V in one surface, interleaved as UYVY.
		/// </summary>
		cudaEglColorFormatUYVY422 = 13,

		/// <summary>
		/// R/G/B/A four channels in one surface with RGBA byte ordering.
		/// </summary>
		cudaEglColorFormatABGR = 14,

		/// <summary>
		/// R/G/B/A four channels in one surface with ARGB byte ordering.
		/// </summary>
		cudaEglColorFormatBGRA = 15,

		/// <summary>
		/// Alpha color format - one channel in one surface.
		/// </summary>
		cudaEglColorFormatA = 16,

		/// <summary>
		/// R/G color format - two channels in one surface with GR byte ordering.
		/// </summary>
		cudaEglColorFormatRG = 17,

		/// <summary>
		/// Y, U, V, A four channels in one surface, interleaved as VUYA.
		/// </summary>
		cudaEglColorFormatAYUV = 18,

		/// <summary>
		/// Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU444SemiPlanar = 19,

		/// <summary>
		/// Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU422SemiPlanar = 20,

		/// <summary>
		/// Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatYVU420SemiPlanar = 21,

		/// <summary>
		/// Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatY10V10U10_444SemiPlanar = 22,

		/// <summary>
		/// Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatY10V10U10_420SemiPlanar = 23,

		/// <summary>
		/// Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatY12V12U12_444SemiPlanar = 24,

		/// <summary>
		/// Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatY12V12U12_420SemiPlanar = 25,

		/// <summary>
		/// Extended Range Y, U, V in one surface, interleaved as YVYU.
		/// </summary>
		cudaEglColorFormatVYUY_ER = 26,

		/// <summary>
		/// Extended Range Y, U, V in one surface, interleaved as YUYV.
		/// </summary>
		cudaEglColorFormatUYVY_ER = 27,

		/// <summary>
		/// Extended Range Y, U, V in one surface, interleaved as UYVY.
		/// </summary>
		cudaEglColorFormatYUYV_ER = 28,

		/// <summary>
		/// Extended Range Y, U, V in one surface, interleaved as VYUY.
		/// </summary>
		cudaEglColorFormatYVYU_ER = 29,

		/// <summary>
		/// Extended Range Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported.
		/// </summary>
		cudaEglColorFormatYUV_ER = 30,

		/// <summary>
		/// Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY.
		/// </summary>
		cudaEglColorFormatYUVA_ER = 31,

		/// <summary>
		/// Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA.
		/// </summary>
		cudaEglColorFormatAYUV_ER = 32,

		/// <summary>
		/// Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYUV444Planar_ER = 33,

		/// <summary>
		/// Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYUV422Planar_ER = 34,

		/// <summary>
		/// Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatYUV420Planar_ER = 35,

		/// <summary>
		/// Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYUV444SemiPlanar_ER = 36,

		/// <summary>
		/// Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYUV422SemiPlanar_ER = 37,

		/// <summary>
		/// Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatYUV420SemiPlanar_ER = 38,

		/// <summary>
		/// Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU444Planar_ER = 39,

		/// <summary>
		/// Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU422Planar_ER = 40,

		/// <summary>
		/// Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatYVU420Planar_ER = 41,

		/// <summary>
		/// Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU444SemiPlanar_ER = 42,

		/// <summary>
		/// Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU422SemiPlanar_ER = 43,

		/// <summary>
		/// Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatYVU420SemiPlanar_ER = 44,

		/// <summary>
		/// Bayer format - one channel in one surface with interleaved RGGB ordering.
		/// </summary>
		cudaEglColorFormatBayerRGGB = 45,

		/// <summary>
		/// Bayer format - one channel in one surface with interleaved BGGR ordering.
		/// </summary>
		cudaEglColorFormatBayerBGGR = 46,

		/// <summary>
		/// Bayer format - one channel in one surface with interleaved GRBG ordering.
		/// </summary>
		cudaEglColorFormatBayerGRBG = 47,

		/// <summary>
		/// Bayer format - one channel in one surface with interleaved GBRG ordering.
		/// </summary>
		cudaEglColorFormatBayerGBRG = 48,

		/// <summary>
		/// Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer10RGGB = 49,

		/// <summary>
		/// Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer10BGGR = 50,

		/// <summary>
		/// Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer10GRBG = 51,

		/// <summary>
		/// Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer10GBRG = 52,

		/// <summary>
		/// Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer12RGGB = 53,

		/// <summary>
		/// Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer12BGGR = 54,

		/// <summary>
		/// Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer12GRBG = 55,

		/// <summary>
		/// Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer12GBRG = 56,

		/// <summary>
		/// Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer14RGGB = 57,

		/// <summary>
		/// Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer14BGGR = 58,

		/// <summary>
		/// Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer14GRBG = 59,

		/// <summary>
		/// Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer14GBRG = 60,

		/// <summary>
		/// Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer20RGGB = 61,

		/// <summary>
		/// Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer20BGGR = 62,

		/// <summary>
		/// Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer20GRBG = 63,

		/// <summary>
		/// Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op.
		/// </summary>
		cudaEglColorFormatBayer20GBRG = 64,

		/// <summary>
		/// Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU444Planar = 65,

		/// <summary>
		/// Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.
		/// </summary>
		cudaEglColorFormatYVU422Planar = 66,

		/// <summary>
		/// Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.
		/// </summary>
		cudaEglColorFormatYVU420Planar = 67,

		/// <summary>
		/// Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype.
		/// </summary>
		cudaEglColorFormatBayerIspRGGB = 68,

		/// <summary>
		/// Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype.
		/// </summary>
		cudaEglColorFormatBayerIspBGGR = 69,

		/// <summary>
		/// Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype.
		/// </summary>
		cudaEglColorFormatBayerIspGRBG = 70,

		/// <summary>
		/// Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype.
		/// </summary>
		cudaEglColorFormatBayerIspGBRG = 71
	}

	/// <summary>
	/// (Runtime API) CUDA EglFrame type - array or pointer.
	/// </summary>
	public enum cudaEglFrameType {
		/// <summary>
		/// Frame type CUDA array
		/// </summary>
		cudaEglFrameTypeArray = 0,

		/// <summary>
		/// Frame type CUDA pointer
		/// </summary>
		cudaEglFrameTypePitch = 1
	}

	/// <summary>
	/// (Runtime API) Resource location flags- sysmem or vidmem.
	/// </summary>
	public enum cudaEglResourceLocationFlags {
		/// <summary>
		/// Resource location sysmem
		/// </summary>
		cudaEglResourceLocationSysmem = 0x00,

		/// <summary>
		/// Resource location vidmem
		/// </summary>
		cudaEglResourceLocationVidmem = 0x01
	}

	/// <summary>
	/// (Runtime API) CUDA error types.
	/// </summary>
	public enum cudaError {
		/// <summary>
		/// The API call returned with no errors.
		/// In the case of query calls, this can also mean that the operation being queried is complete
		/// (see cudaEventQuery() and cudaStreamQuery()).
		/// </summary>
		cudaSuccess = 0,

		/// <summary>
		/// This indicates that one or more of the parameters passed to the API call
		/// is not within an acceptable range of values.
		/// </summary>
		cudaErrorInvalidValue = 1,

		/// <summary>
		/// The API call failed because it was unable to allocate enough memory to perform the requested operation.
		/// </summary>
		cudaErrorMemoryAllocation = 2,

		/// <summary>
		/// The API call failed because the CUDA driver and runtime could not be initialized.
		/// </summary>
		cudaErrorInitializationError = 3,

		/// <summary>
		/// This indicates that a CUDA Runtime API call cannot be executed because
		/// it is being called during process shut down,
		/// at a point in time after CUDA driver has been unloaded.
		/// </summary>
		cudaErrorCudartUnloading = 4,

		/// <summary>
		/// This indicates profiler is not initialized for this run.
		/// This can happen when the application is running with external
		/// profiling tools like visual profiler.
		/// </summary>
		cudaErrorProfilerDisabled = 5,

		/// <summary>
		/// It is no longer an error to attempt to enable/disable the profiling
		/// via cudaProfilerStart or cudaProfilerStop without initialization.
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 5.0.")]
		cudaErrorProfilerNotInitialized = 6,

		/// <summary>
		/// It is no longer an error to call cudaProfilerStart()
		/// when profiling is already enabled.
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 5.0.")]
		cudaErrorProfilerAlreadyStarted = 7,

		/// <summary>
		///  It is no longer an error to call cudaProfilerStop()
		///  when profiling is already disabled.
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 5.0.")]
		cudaErrorProfilerAlreadyStopped = 8,

		/// <summary>
		/// This indicates that a kernel launch is requesting resources that can never be satisfied by the current device.
		/// </summary>
		/// <remarks>
		/// Requesting more shared memory per block than the device supports will trigger this error,
		/// as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations.
		/// </remarks>
		cudaErrorInvalidConfiguration = 9,

		/// <summary>
		/// This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.
		/// </summary>
		cudaErrorInvalidPitchValue = 12,

		/// <summary>
		/// This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.
		/// </summary>
		cudaErrorInvalidSymbol = 13,

		/// <summary>
		/// This indicates that at least one host pointer passed to the API call is not a valid host pointer.
		/// </summary>
		cudaErrorInvalidHostPointer = 16,

		/// <summary>
		/// This indicates that at least one device pointer passed to the API call is not a valid device pointer.
		/// </summary>
		cudaErrorInvalidDevicePointer = 17,

		/// <summary>
		/// This indicates that the texture passed to the API call is not a valid texture.
		/// </summary>
		cudaErrorInvalidTexture = 18,

		/// <summary>
		/// This indicates that the texture binding is not valid.
		/// This occurs if you call cudaGetTextureAlignmentOffset() with an unbound texture.
		/// </summary>
		cudaErrorInvalidTextureBinding = 19,

		/// <summary>
		/// This indicates that the channel descriptor passed to the API call is not valid.
		/// </summary>
		/// <remarks>
		/// This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of the dimensions is invalid.
		/// </remarks>
		cudaErrorInvalidChannelDescriptor = 20,

		/// <summary>
		/// This indicates that the direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind.
		/// </summary>
		cudaErrorInvalidMemcpyDirection = 21,

		[Obsolete("This error return is deprecated as of CUDA 3.1")]
		cudaErrorAddressOfConstant = 22,

		[Obsolete("This error return is deprecated as of CUDA 3.1")]
		cudaErrorTextureFetchFailed = 23,

		[Obsolete("This error return is deprecated as of CUDA 3.1")]
		cudaErrorTextureNotBound = 24,

		[Obsolete("This error return is deprecated as of CUDA 3.1")]
		cudaErrorSynchronizationError = 25,

		/// <summary>
		/// This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.
		/// </summary>
		cudaErrorInvalidFilterSetting = 26,

		/// <summary>
		/// This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.
		/// </summary>
		cudaErrorInvalidNormSetting = 27,

		[Obsolete("This error return is deprecated as of CUDA 3.1")]
		cudaErrorMixedDeviceExecution = 28,

		[Obsolete("This error return is deprecated as of CUDA 4.1")]
		cudaErrorNotYetImplemented = 31,

		[Obsolete("This error return is deprecated as of CUDA 3.1")]
		cudaErrorMemoryValueTooLarge = 32,

		/// <summary>
		/// This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library.
		/// This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.
		/// </summary>
		cudaErrorInsufficientDriver = 35,

		/// <summary>
		/// This indicates that the surface passed to the API call is not a valid surface.
		/// </summary>
		cudaErrorInvalidSurface = 37,

		/// <summary>
		/// This indicates that multiple global or constant variables 
		/// (across separate CUDA source files in the application) share the same string name.
		/// </summary>
		cudaErrorDuplicateVariableName = 43,

		/// <summary>
		/// This indicates that multiple textures (across separate CUDA source files in the application) share the same string name.
		/// </summary>
		cudaErrorDuplicateTextureName = 44,

		/// <summary>
		/// This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name.
		/// </summary>
		cudaErrorDuplicateSurfaceName = 45,

		/// <summary>
		/// This indicates that all CUDA devices are busy or unavailable at the current time.
		/// </summary>
		/// <remarks>
		/// Devices are often busy/unavailable due to use of cudaComputeModeExclusive,
		/// cudaComputeModeProhibited or when long running CUDA kernels have filled
		/// up the GPU and are blocking new work from starting.
		/// They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed.
		/// </remarks>
		cudaErrorDevicesUnavailable = 46,

		/// <summary>
		/// This indicates that the current context is not compatible with this the CUDA Runtime.
		/// </summary>
		/// <remarks>
		/// This can only occur if you are using CUDA Runtime/Driver interoperability and have created
		/// an existing Driver context using the driver API.
		/// The Driver context may be incompatible either because the Driver context was created
		/// using an older version of the API, because the Runtime API call expects a primary driver
		/// context and the Driver context is not primary, or because the Driver context has been destroyed.
		/// Please see Interactions with the CUDA Driver API" for more information.
		/// </remarks>
		cudaErrorIncompatibleDriverContext = 49,

		/// <summary>
		/// The device function being invoked (usually via cudaLaunchKernel())
		/// was not previously configured via the cudaConfigureCall() function.
		/// </summary>
		cudaErrorMissingConfiguration = 52,

		/// <summary>
		/// 
		/// </summary>
		/// [Obsolete("This error return is deprecated as of CUDA 3.1")]
		cudaErrorPriorLaunchFailure = 53,

		/// <summary>
		/// This error indicates that a device runtime grid launch did not occur because 
		/// the depth of the child grid would exceed the maximum supported number of nested grid launches.
		/// </summary>
		cudaErrorLaunchMaxDepthExceeded = 65,

		/// <summary>
		/// This error indicates that a grid launch did not occur because the kernel uses file-scoped textures
		/// which are unsupported by the device runtime.
		/// Kernels launched via the device runtime only support textures created with the Texture Object API's.
		/// </summary>
		cudaErrorLaunchFileScopedTex = 66,

		/// <summary>
		/// This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces
		/// which are unsupported by the device runtime.
		/// Kernels launched via the device runtime only support surfaces created with the Surface Object API's.
		/// </summary>
		cudaErrorLaunchFileScopedSurf = 67,

		/// <summary>
		/// This error indicates that a call to cudaDeviceSynchronize made from the device runtime failed
		/// because the call was made at grid depth greater than than either the default (2 levels of grids)
		/// or user specified device limit cudaLimitDevRuntimeSyncDepth.
		/// </summary>
		/// <remarks>
		/// To be able to synchronize on launched grids at a greater depth successfully,
		/// the maximum nested depth at which cudaDeviceSynchronize will be called must be specified
		/// with the cudaLimitDevRuntimeSyncDepth limit to the cudaDeviceSetLimit api before the host-side
		/// launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth
		/// require the runtime to reserve large amounts of device memory that cannot be used for user allocations.
		/// </remarks>
		cudaErrorSyncDepthExceeded = 68,

		/// <summary>
		/// This error indicates that a device runtime grid launch failed because the launch would exceed
		/// the limit cudaLimitDevRuntimePendingLaunchCount.
		/// </summary>
		/// <remarks>
		/// For this launch to proceed successfully, cudaDeviceSetLimit must be called to
		/// set the cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding
		/// launches that can be issued to the device runtime.
		/// Keep in mind that raising the limit of pending device runtime launches will require the runtime
		/// to reserve device memory that cannot be used for user allocations.
		/// </remarks>
		cudaErrorLaunchPendingCountExceeded = 69,

		/// <summary>
		/// The requested device function does not exist or is not compiled for the proper device architecture.
		/// </summary>
		cudaErrorInvalidDeviceFunction = 98,

		/// <summary>
		/// This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
		/// </summary>
		cudaErrorNoDevice = 100,

		/// <summary>
		/// This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device.
		/// </summary>
		cudaErrorInvalidDevice = 101,

		/// <summary>
		/// This indicates an internal startup failure in the CUDA runtime.
		/// </summary>
		cudaErrorStartupFailure = 127,

		/// <summary>
		/// This indicates that the device kernel image is invalid.
		/// </summary>
		cudaErrorInvalidKernelImage = 200,

		/// <summary>
		/// This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.
		/// </summary>
		cudaErrorDeviceUninitialized = 201,

		/// <summary>
		/// This indicates that the buffer object could not be mapped.
		/// </summary>
		cudaErrorMapBufferObjectFailed = 205,

		/// <summary>
		/// This indicates that the buffer object could not be unmapped.
		/// </summary>
		cudaErrorUnmapBufferObjectFailed = 206,

		/// <summary>
		/// This indicates that the specified array is currently mapped and thus cannot be destroyed.
		/// </summary>
		cudaErrorArrayIsMapped = 207,

		/// <summary>
		/// This indicates that the resource is already mapped.
		/// </summary>
		cudaErrorAlreadyMapped = 208,

		/// <summary>
		/// This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
		/// </summary>
		cudaErrorNoKernelImageForDevice = 209,

		/// <summary>
		/// This indicates that a resource has already been acquired.
		/// </summary>
		cudaErrorAlreadyAcquired = 210,

		/// <summary>
		/// This indicates that a resource is not mapped.
		/// </summary>
		cudaErrorNotMapped = 211,

		/// <summary>
		/// This indicates that a mapped resource is not available for access as an array.
		/// </summary>
		cudaErrorNotMappedAsArray = 212,

		/// <summary>
		/// This indicates that a mapped resource is not available for access as a pointer.
		/// </summary>
		cudaErrorNotMappedAsPointer = 213,

		/// <summary>
		/// This indicates that an uncorrectable ECC error was detected during execution.
		/// </summary>
		cudaErrorECCUncorrectable = 214,

		/// <summary>
		/// This indicates that the cudaLimit passed to the API call is not supported by the active device.
		/// </summary>
		cudaErrorUnsupportedLimit = 215,

		/// <summary>
		/// This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.
		/// </summary>
		cudaErrorDeviceAlreadyInUse = 216,

		/// <summary>
		/// This error indicates that P2P access is not supported across the given devices.
		/// </summary>
		cudaErrorPeerAccessUnsupported = 217,

		/// <summary>
		/// A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
		/// </summary>
		cudaErrorInvalidPtx = 218,

		/// <summary>
		/// This indicates an error with the OpenGL or DirectX context.
		/// </summary>
		cudaErrorInvalidGraphicsContext = 219,

		/// <summary>
		/// This indicates that an uncorrectable NVLink error was detected during the execution.
		/// </summary>
		cudaErrorNvlinkUncorrectable = 220,

		/// <summary>
		/// This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.
		/// </summary>
		cudaErrorJitCompilerNotFound = 221,

		/// <summary>
		/// This indicates that the device kernel source is invalid.
		/// </summary>
		cudaErrorInvalidSource = 300,

		/// <summary>
		/// This indicates that the file specified was not found.
		/// </summary>
		cudaErrorFileNotFound = 301,

		/// <summary>
		/// This indicates that a link to a shared object failed to resolve.
		/// </summary>
		cudaErrorSharedObjectSymbolNotFound = 302,

		/// <summary>
		/// This indicates that initialization of a shared object failed.
		/// </summary>
		cudaErrorSharedObjectInitFailed = 303,

		/// <summary>
		/// This error indicates that an OS call failed.
		/// </summary>
		cudaErrorOperatingSystem = 304,

		/// <summary>
		/// This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like cudaStream_t and cudaEvent_t.
		/// </summary>
		cudaErrorInvalidResourceHandle = 400,

		/// <summary>
		/// This indicates that a resource required by the API call is not in a valid state to perform the requested operation.
		/// </summary>
		cudaErrorIllegalState = 401,

		/// <summary>
		/// This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names.
		/// </summary>
		cudaErrorSymbolNotFound = 500,

		/// <summary>
		/// This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates completion). Calls that may return this value include cudaEventQuery() and cudaStreamQuery().
		/// </summary>
		cudaErrorNotReady = 600,

		/// <summary>
		/// The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorIllegalAddress = 700,

		/// <summary>
		/// This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.
		/// </summary>
		cudaErrorLaunchOutOfResources = 701,

		/// <summary>
		/// This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property kernelExecTimeoutEnabled for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorLaunchTimeout = 702,

		/// <summary>
		/// This error indicates a kernel launch that uses an incompatible texturing mode.
		/// </summary>
		cudaErrorLaunchIncompatibleTexturing = 703,

		/// <summary>
		/// This error indicates that a call to cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.
		/// </summary>
		cudaErrorPeerAccessAlreadyEnabled = 704,

		/// <summary>
		/// This error indicates that cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().
		/// </summary>
		cudaErrorPeerAccessNotEnabled = 705,

		/// <summary>
		/// This indicates that the user has called cudaSetValidDevices(), cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(), or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing CUcontext active on the host thread.
		/// </summary>
		cudaErrorSetOnActiveProcess = 708,

		/// <summary>
		/// This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.
		/// </summary>
		cudaErrorContextIsDestroyed = 709,

		/// <summary>
		/// An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorAssert = 710,

		/// <summary>
		/// This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cudaEnablePeerAccess().
		/// </summary>
		cudaErrorTooManyPeers = 711,

		/// <summary>
		/// This error indicates that the memory range passed to cudaHostRegister() has already been registered.
		/// </summary>
		cudaErrorHostMemoryAlreadyRegistered = 712,

		/// <summary>
		/// This error indicates that the pointer passed to cudaHostUnregister() does not correspond to any currently registered memory region.
		/// </summary>
		cudaErrorHostMemoryNotRegistered = 713,

		/// <summary>
		/// Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorHardwareStackError = 714,

		/// <summary>
		/// The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorIllegalInstruction = 715,

		/// <summary>
		/// The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorMisalignedAddress = 716,

		/// <summary>
		/// While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorInvalidAddressSpace = 717,

		/// <summary>
		/// The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorInvalidPc = 718,

		/// <summary>
		/// An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
		/// </summary>
		cudaErrorLaunchFailure = 719,

		/// <summary>
		/// This error indicates that the number of blocks launched per grid for a kernel that was launched via either cudaLaunchCooperativeKernel or cudaLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cudaOccupancyMaxActiveBlocksPerMultiprocessor or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount.
		/// </summary>
		cudaErrorCooperativeLaunchTooLarge = 720,

		/// <summary>
		/// This error indicates the attempted operation is not permitted.
		/// </summary>
		cudaErrorNotPermitted = 800,

		/// <summary>
		/// This error indicates the attempted operation is not supported on the current system or device.
		/// </summary>
		cudaErrorNotSupported = 801,

		/// <summary>
		/// This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
		/// </summary>
		cudaErrorSystemNotReady = 802,

		/// <summary>
		/// This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.
		/// </summary>
		cudaErrorSystemDriverMismatch = 803,

		/// <summary>
		/// This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
		/// </summary>
		cudaErrorCompatNotSupportedOnDevice = 804,

		/// <summary>
		/// The operation is not permitted when the stream is capturing.
		/// </summary>
		cudaErrorStreamCaptureUnsupported = 900,

		/// <summary>
		/// The current capture sequence on the stream has been invalidated due to a previous error.
		/// </summary>
		cudaErrorStreamCaptureInvalidated = 901,

		/// <summary>
		/// The operation would have resulted in a merge of two independent capture sequences.
		/// </summary>
		cudaErrorStreamCaptureMerge = 902,

		/// <summary>
		/// The capture was not initiated in this stream.
		/// </summary>
		cudaErrorStreamCaptureUnmatched = 903,

		/// <summary>
		/// The capture sequence contains a fork that was not joined to the primary stream.
		/// </summary>
		cudaErrorStreamCaptureUnjoined = 904,

		/// <summary>
		/// A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
		/// </summary>
		cudaErrorStreamCaptureIsolation = 905,

		/// <summary>
		/// The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.
		/// </summary>
		cudaErrorStreamCaptureImplicit = 906,

		/// <summary>
		/// The operation is not permitted on an event which was last recorded in a capturing stream.
		/// </summary>
		cudaErrorCapturedEvent = 907,

		/// <summary>
		/// A stream capture sequence not initiated with the cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture in a different thread.
		/// </summary>
		cudaErrorStreamCaptureWrongThread = 908,

		/// <summary>
		/// This indicates that the wait operation has timed out.
		/// </summary>
		cudaErrorTimeout = 909,

		/// <summary>
		/// This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.
		/// </summary>
		cudaErrorGraphExecUpdateFailure = 910,

		/// <summary>
		/// This indicates that an unknown internal error has occurred.
		/// </summary>
		cudaErrorUnknown = 999,

		/// <summary>
		/// Any unhandled CUDA driver error is added to this value and returned via the runtime.
		/// Production releases of CUDA should not return such errors.
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 4.1")]
		cudaErrorApiFailureBase = 10000
	}

	/// <summary>
	/// (Runtime API) External memory handle types.
	/// </summary>
	public enum cudaExternalMemoryHandleType {
		/// <summary>
		/// Handle is an opaque file descriptor.
		/// </summary>
		cudaExternalMemoryHandleTypeOpaqueFd = 1,
		/// <summary>
		/// Handle is an opaque shared NT handle.
		/// </summary>
		cudaExternalMemoryHandleTypeOpaqueWin32 = 2,
		/// <summary>
		/// Handle is an opaque, globally shared handle.
		/// </summary>
		cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
		/// <summary>
		/// Handle is a D3D12 heap object.
		/// </summary>
		cudaExternalMemoryHandleTypeD3D12Heap = 4,
		/// <summary>
		/// Handle is a D3D12 committed resource.
		/// </summary>
		cudaExternalMemoryHandleTypeD3D12Resource = 5,
		/// <summary>
		/// Handle is a shared NT handle to a D3D11 resource
		/// </summary>
		cudaExternalMemoryHandleTypeD3D11Resource = 6,
		/// <summary>
		/// Handle is a globally shared handle to a D3D11 resource
		/// </summary>
		cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
		/// <summary>
		/// Handle is an NvSciBuf object
		/// </summary>
		cudaExternalMemoryHandleTypeNvSciBuf = 8
	}

	/// <summary>
	/// (Runtime API) External semaphore handle types.
	/// </summary>
	public enum cudaExternalSemaphoreHandleType {
		/// <summary>
		/// Handle is an opaque file descriptor.
		/// </summary>
		cudaExternalSemaphoreHandleTypeOpaqueFd = 1,
		/// <summary>
		/// Handle is an opaque shared NT handle.
		/// </summary>
		cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,
		/// <summary>
		/// Handle is an opaque, globally shared handle.
		/// </summary>
		cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
		/// <summary>
		/// Handle is a shared NT handle referencing a D3D12 fence object.
		/// </summary>
		cudaExternalSemaphoreHandleTypeD3D12Fence = 4,
		/// <summary>
		/// Handle is a shared NT handle referencing a D3D11 fence object
		/// </summary>
		cudaExternalSemaphoreHandleTypeD3D11Fence = 5,
		/// <summary>
		/// Opaque handle to NvSciSync Object
		/// </summary>
		cudaExternalSemaphoreHandleTypeNvSciSync = 6,
		/// <summary>
		/// Handle is a shared NT handle referencing a D3D11 keyed mutex object
		/// </summary>
		cudaExternalSemaphoreHandleTypeKeyedMutex = 7,
		/// <summary>
		/// Handle is a shared KMT handle referencing a D3D11 keyed mutex object
		/// </summary>
		cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8
	}

	/// <summary>
	/// (Runtime API) CUDA function attributes that can be set using cudaFuncSetAttribute.
	/// </summary>
	public enum cudaFuncAttribute {
		/// <summary>
		/// Maximum dynamic shared memory size
		/// </summary>
		cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
		/// <summary>
		/// Preferred shared memory-L1 cache split
		/// </summary>
		cudaFuncAttributePreferredSharedMemoryCarveout = 9,
		cudaFuncAttributeMax
	}

	/// <summary>
	/// (Runtime API) CUDA function cache configurations.
	/// </summary>
	public enum cudaFuncCache {
		/// <summary>
		/// Default function cache configuration, no preference
		/// </summary>
		cudaFuncCachePreferNone = 0,

		/// <summary>
		/// Prefer larger shared memory and smaller L1 cache
		/// </summary>
		cudaFuncCachePreferShared = 1,

		/// <summary>
		/// Prefer larger L1 cache and smaller shared memory
		/// </summary>
		cudaFuncCachePreferL1 = 2,

		/// <summary>
		/// Prefer equal size L1 cache and shared memory
		/// </summary>
		cudaFuncCachePreferEqual = 3
	}

	/// <summary>
	/// (Runtime API) CUDA Graph Update error types.
	/// </summary>
	public enum cudaGraphExecUpdateResult {
		/// <summary>
		/// The update succeeded
		/// </summary>
		cudaGraphExecUpdateSuccess = 0x0,

		/// <summary>
		/// The update failed for an unexpected reason which is described in the return value of the function
		/// </summary>
		cudaGraphExecUpdateError = 0x1,

		/// <summary>
		/// The update failed because the topology changed
		/// </summary>
		cudaGraphExecUpdateErrorTopologyChanged = 0x2,

		/// <summary>
		/// The update failed because a node type changed
		/// </summary>
		cudaGraphExecUpdateErrorNodeTypeChanged = 0x3,

		/// <summary>
		/// The update failed because the function of a kernel node changed
		/// </summary>
		cudaGraphExecUpdateErrorFunctionChanged = 0x4,

		/// <summary>
		/// The update failed because the parameters changed in a way that is not supported
		/// </summary>
		cudaGraphExecUpdateErrorParametersChanged = 0x5,

		/// <summary>
		/// The update failed because something about the node is not supported
		/// </summary>
		cudaGraphExecUpdateErrorNotSupported = 0x6
	}

	/// <summary>
	/// (Runtime API) CUDA Graph node types.
	/// </summary>
	public enum cudaGraphNodeType {
		/// <summary>
		/// GPU kernel node
		/// </summary>
		cudaGraphNodeTypeKernel = 0x00,

		/// <summary>
		/// Memcpy node
		/// </summary>
		cudaGraphNodeTypeMemcpy = 0x01,

		/// <summary>
		/// Memset node
		/// </summary>
		cudaGraphNodeTypeMemset = 0x02,

		/// <summary>
		/// Host (executable) node
		/// </summary>
		cudaGraphNodeTypeHost = 0x03,

		/// <summary>
		/// Node which executes an embedded graph
		/// </summary>
		cudaGraphNodeTypeGraph = 0x04,

		/// <summary>
		/// Empty (no-op) node
		/// </summary>
		cudaGraphNodeTypeEmpty = 0x05,
		cudaGraphNodeTypeCount
	}

	/// <summary>
	/// (Runtime API) CUDA graphics interop array indices for cube maps.
	/// </summary>
	public enum cudaGraphicsCubeFace {
		/// <summary>
		/// Positive X face of cubemap
		/// </summary>
		cudaGraphicsCubeFacePositiveX = 0x00,

		/// <summary>
		/// Negative X face of cubemap
		/// </summary>
		cudaGraphicsCubeFaceNegativeX = 0x01,

		/// <summary>
		/// Positive Y face of cubemap
		/// </summary>
		cudaGraphicsCubeFacePositiveY = 0x02,

		/// <summary>
		/// Negative Y face of cubemap
		/// </summary>
		cudaGraphicsCubeFaceNegativeY = 0x03,

		/// <summary>
		/// Positive Z face of cubemap
		/// </summary>
		cudaGraphicsCubeFacePositiveZ = 0x04,

		/// <summary>
		/// Negative Z face of cubemap
		/// </summary>
		cudaGraphicsCubeFaceNegativeZ = 0x05
	}

	/// <summary>
	/// (Runtime API) CUDA graphics interop map flags.
	/// </summary>
	public enum cudaGraphicsMapFlags {
		/// <summary>
		/// Default; Assume resource can be read/written
		/// </summary>
		cudaGraphicsMapFlagsNone = 0,

		/// <summary>
		/// CUDA will not write to this resource
		/// </summary>
		cudaGraphicsMapFlagsReadOnly = 1,

		/// <summary>
		/// CUDA will only write to and will not read from this resource
		/// </summary>
		cudaGraphicsMapFlagsWriteDiscard = 2
	}

	/// <summary>
	/// (Runtime API) CUDA graphics interop register flags.
	/// </summary>
	[Flags]
	public enum cudaGraphicsRegisterFlags {
		/// <summary>
		/// Default
		/// </summary>
		cudaGraphicsRegisterFlagsNone = 0,

		/// <summary>
		/// CUDA will not write to this resource
		/// </summary>
		cudaGraphicsRegisterFlagsReadOnly = 1,

		/// <summary>
		/// CUDA will only write to and will not read from this resource
		/// </summary>
		cudaGraphicsRegisterFlagsWriteDiscard = 2,

		/// <summary>
		/// CUDA will bind this resource to a surface reference
		/// </summary>
		cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,

		/// <summary>
		/// CUDA will perform texture gather operations on this resource
		/// </summary>
		cudaGraphicsRegisterFlagsTextureGather = 8
	}

	/// <summary>
	/// (Runtime API) CUDA Limits.
	/// </summary>
	public enum cudaLimit {
		/// <summary>
		/// GPU thread stack size
		/// </summary>
		cudaLimitStackSize = 0x00,

		/// <summary>
		/// GPU printf/fprintf FIFO size
		/// </summary>
		cudaLimitPrintfFifoSize = 0x01,

		/// <summary>
		/// GPU malloc heap size
		/// </summary>
		cudaLimitMallocHeapSize = 0x02,

		/// <summary>
		/// GPU device runtime synchronize depth
		/// </summary>
		cudaLimitDevRuntimeSyncDepth = 0x03,

		/// <summary>
		/// GPU device runtime pending launch count
		/// </summary>
		cudaLimitDevRuntimePendingLaunchCount = 0x04,

		/// <summary>
		/// A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint.
		/// </summary>
		cudaLimitMaxL2FetchGranularity = 0x05
	}

	/// <summary>
	/// (Runtime API) CUDA range attributes.
	/// </summary>
	public enum cudaMemRangeAttribute {
		/// <summary>
		/// Whether the range will mostly be read and only occassionally be written to
		/// </summary>
		cudaMemRangeAttributeReadMostly = 1,

		/// <summary>
		/// The preferred location of the range
		/// </summary>
		cudaMemRangeAttributePreferredLocation = 2,

		/// <summary>
		/// Memory range has cudaMemAdviseSetAccessedBy set for specified device
		/// </summary>
		cudaMemRangeAttributeAccessedBy = 3,

		/// <summary>
		/// The last location to which the range was prefetched
		/// </summary>
		cudaMemRangeAttributeLastPrefetchLocation = 4
	}

	/// <summary>
	/// (Runtime API) CUDA memory copy types.
	/// </summary>
	public enum cudaMemcpyKind {
		/// <summary>
		/// Host -> Host
		/// </summary>
		cudaMemcpyHostToHost = 0,

		/// <summary>
		/// Host -> Device
		/// </summary>
		cudaMemcpyHostToDevice = 1,

		/// <summary>
		/// Device -> Host
		/// </summary>
		cudaMemcpyDeviceToHost = 2,

		/// <summary>
		/// Device -> Device
		/// </summary>
		cudaMemcpyDeviceToDevice = 3,

		/// <summary>
		/// Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing
		/// </summary>
		cudaMemcpyDefault = 4
	}

	/// <summary>
	/// (Runtime API) CUDA Memory Advise values.
	/// </summary>
	public enum cudaMemoryAdvise {
		/// <summary>
		/// Data will mostly be read and only occassionally be written to
		/// </summary>
		cudaMemAdviseSetReadMostly = 1,

		/// <summary>
		/// Undo the effect of cudaMemAdviseSetReadMostly
		/// </summary>
		cudaMemAdviseUnsetReadMostly = 2,

		/// <summary>
		/// Set the preferred location for the data as the specified device
		/// </summary>
		cudaMemAdviseSetPreferredLocation = 3,

		/// <summary>
		/// Clear the preferred location for the data
		/// </summary>
		cudaMemAdviseUnsetPreferredLocation = 4,

		/// <summary>
		/// Data will be accessed by the specified device, so prevent page faults as much as possible
		/// </summary>
		cudaMemAdviseSetAccessedBy = 5,

		/// <summary>
		/// Let the Unified Memory subsystem decide on the page faulting policy for the specified device
		/// </summary>
		cudaMemAdviseUnsetAccessedBy = 6
	}

	/// <summary>
	/// (Runtime API) CUDA memory types.
	/// </summary>
	public enum cudaMemoryType {
		/// <summary>
		/// Unregistered memory
		/// </summary>
		cudaMemoryTypeUnregistered = 0,

		/// <summary>
		/// Host memory
		/// </summary>
		cudaMemoryTypeHost = 1,

		/// <summary>
		/// Device memory
		/// </summary>
		cudaMemoryTypeDevice = 2,

		/// <summary>
		/// Managed memory
		/// </summary>
		cudaMemoryTypeManaged = 3
	}

	/// <summary>
	/// (Runtime API) CUDA Profiler Output modes.
	/// </summary>
	public enum cudaOutputMode {
		/// <summary>
		/// Output mode Key-Value pair format.
		/// </summary>
		cudaKeyValuePair = 0x00,

		/// <summary>
		/// Output mode Comma separated values format.
		/// </summary>
		cudaCSV = 0x01
	}

	/// <summary>
	/// (Runtime API) CUDA resource types.
	/// </summary>
	public enum cudaResourceType {
		/// <summary>
		/// Array resource
		/// </summary>
		cudaResourceTypeArray = 0x00,

		/// <summary>
		/// Mipmapped array resource
		/// </summary>
		cudaResourceTypeMipmappedArray = 0x01,

		/// <summary>
		/// Linear resource
		/// </summary>
		cudaResourceTypeLinear = 0x02,

		/// <summary>
		/// Pitch 2D resource
		/// </summary>
		cudaResourceTypePitch2D = 0x03
	}

	/// <summary>
	/// (Runtime API) CUDA texture resource view formats.
	/// </summary>
	public enum cudaResourceViewFormat {
		/// <summary>
		/// No resource view format (use underlying resource format)
		/// </summary>
		cudaResViewFormatNone = 0x00,

		/// <summary>
		/// 1 channel unsigned 8-bit integers
		/// </summary>
		cudaResViewFormatUnsignedChar1 = 0x01,

		/// <summary>
		/// 2 channel unsigned 8-bit integers
		/// </summary>
		cudaResViewFormatUnsignedChar2 = 0x02,

		/// <summary>
		/// 4 channel unsigned 8-bit integers
		/// </summary>
		cudaResViewFormatUnsignedChar4 = 0x03,

		/// <summary>
		/// 1 channel signed 8-bit integers
		/// </summary>
		cudaResViewFormatSignedChar1 = 0x04,

		/// <summary>
		/// 2 channel signed 8-bit integers
		/// </summary>
		cudaResViewFormatSignedChar2 = 0x05,

		/// <summary>
		/// 4 channel signed 8-bit integers
		/// </summary>
		cudaResViewFormatSignedChar4 = 0x06,

		/// <summary>
		/// 1 channel unsigned 16-bit integers
		/// </summary>
		cudaResViewFormatUnsignedShort1 = 0x07,

		/// <summary>
		/// 2 channel unsigned 16-bit integers
		/// </summary>
		cudaResViewFormatUnsignedShort2 = 0x08,

		/// <summary>
		/// 4 channel unsigned 16-bit integers
		/// </summary>
		cudaResViewFormatUnsignedShort4 = 0x09,

		/// <summary>
		/// 1 channel signed 16-bit integers
		/// </summary>
		cudaResViewFormatSignedShort1 = 0x0a,

		/// <summary>
		/// 2 channel signed 16-bit integers
		/// </summary>
		cudaResViewFormatSignedShort2 = 0x0b,

		/// <summary>
		/// 4 channel signed 16-bit integers
		/// </summary>
		cudaResViewFormatSignedShort4 = 0x0c,

		/// <summary>
		/// 1 channel unsigned 32-bit integers
		/// </summary>
		cudaResViewFormatUnsignedInt1 = 0x0d,

		/// <summary>
		/// 2 channel unsigned 32-bit integers
		/// </summary>
		cudaResViewFormatUnsignedInt2 = 0x0e,

		/// <summary>
		/// 4 channel unsigned 32-bit integers
		/// </summary>
		cudaResViewFormatUnsignedInt4 = 0x0f,

		/// <summary>
		/// 1 channel signed 32-bit integers
		/// </summary>
		cudaResViewFormatSignedInt1 = 0x10,

		/// <summary>
		/// 2 channel signed 32-bit integers
		/// </summary>
		cudaResViewFormatSignedInt2 = 0x11,

		/// <summary>
		/// 4 channel signed 32-bit integers
		/// </summary>
		cudaResViewFormatSignedInt4 = 0x12,

		/// <summary>
		/// 1 channel 16-bit floating point
		/// </summary>
		cudaResViewFormatHalf1 = 0x13,

		/// <summary>
		/// 2 channel 16-bit floating point
		/// </summary>
		cudaResViewFormatHalf2 = 0x14,

		/// <summary>
		/// 4 channel 16-bit floating point
		/// </summary>
		cudaResViewFormatHalf4 = 0x15,

		/// <summary>
		/// 1 channel 32-bit floating point
		/// </summary>
		cudaResViewFormatFloat1 = 0x16,

		/// <summary>
		/// 2 channel 32-bit floating point
		/// </summary>
		cudaResViewFormatFloat2 = 0x17,

		/// <summary>
		/// 4 channel 32-bit floating point
		/// </summary>
		cudaResViewFormatFloat4 = 0x18,

		/// <summary>
		/// Block compressed 1
		/// </summary>
		cudaResViewFormatUnsignedBlockCompressed1 = 0x19,

		/// <summary>
		/// Block compressed 2
		/// </summary>
		cudaResViewFormatUnsignedBlockCompressed2 = 0x1a,

		/// <summary>
		/// Block compressed 3
		/// </summary>
		cudaResViewFormatUnsignedBlockCompressed3 = 0x1b,

		/// <summary>
		/// Block compressed 4 unsigned
		/// </summary>
		cudaResViewFormatUnsignedBlockCompressed4 = 0x1c,

		/// <summary>
		/// Block compressed 4 signed
		/// </summary>
		cudaResViewFormatSignedBlockCompressed4 = 0x1d,

		/// <summary>
		/// Block compressed 5 unsigned
		/// </summary>
		cudaResViewFormatUnsignedBlockCompressed5 = 0x1e,

		/// <summary>
		/// Block compressed 5 signed
		/// </summary>
		cudaResViewFormatSignedBlockCompressed5 = 0x1f,

		/// <summary>
		/// Block compressed 6 unsigned half-float
		/// </summary>
		cudaResViewFormatUnsignedBlockCompressed6H = 0x20,

		/// <summary>
		/// Block compressed 6 signed half-float
		/// </summary>
		cudaResViewFormatSignedBlockCompressed6H = 0x21,

		/// <summary>
		/// Block compressed 7
		/// </summary>
		cudaResViewFormatUnsignedBlockCompressed7 = 0x22
	}

	/// <summary>
	/// (Runtime API) Shared memory carveout configurations.
	/// These may be passed to cudaFuncSetAttribute.
	/// </summary>
	public enum cudaSharedCarveout {
		/// <summary>
		/// No preference for shared memory or L1 (default)
		/// </summary>
		cudaSharedmemCarveoutDefault = -1,

		/// <summary>
		/// Prefer maximum available shared memory, minimum L1 cache
		/// </summary>
		cudaSharedmemCarveoutMaxShared = 100,

		/// <summary>
		/// Prefer maximum available L1 cache, minimum shared memory
		/// </summary>
		cudaSharedmemCarveoutMaxL1 = 0
	}

	/// <summary>
	/// (Runtime API) CUDA shared memory configuration.
	/// </summary>
	public enum cudaSharedMemConfig {
		cudaSharedMemBankSizeDefault = 0,
		cudaSharedMemBankSizeFourByte = 1,
		cudaSharedMemBankSizeEightByte = 2
	}

	/// <summary>
	/// (Runtime API) Possible modes for stream capture thread interactions.
	/// For more details see cudaStreamBeginCapture and cudaThreadExchangeStreamCaptureMode
	/// </summary>
	public enum cudaStreamCaptureMode {
		cudaStreamCaptureModeGlobal = 0,
		cudaStreamCaptureModeThreadLocal = 1,
		cudaStreamCaptureModeRelaxed = 2
	}

	/// <summary>
	/// (Runtime API) Possible stream capture statuses returned by cudaStreamIsCapturing.
	/// </summary>
	public enum cudaStreamCaptureStatus {
		/// <summary>
		/// Stream is not capturing
		/// </summary>
		cudaStreamCaptureStatusNone = 0,

		/// <summary>
		/// Stream is actively capturing
		/// </summary>
		cudaStreamCaptureStatusActive = 1,

		/// <summary>
		/// Stream is part of a capture sequence that has been invalidated, but not terminated
		/// </summary>
		cudaStreamCaptureStatusInvalidated = 2
	}

	/// <summary>
	/// (Runtime API) CUDA Surface boundary modes.
	/// </summary>
	public enum cudaSurfaceBoundaryMode {
		/// <summary>
		/// Zero boundary mode
		/// </summary>
		cudaBoundaryModeZero = 0,

		/// <summary>
		/// Clamp boundary mode
		/// </summary>
		cudaBoundaryModeClamp = 1,

		/// <summary>
		/// Trap boundary mode
		/// </summary>
		cudaBoundaryModeTrap = 2
	}

	/// <summary>
	/// (Runtime API) CUDA Surface format modes.
	/// </summary>
	public enum cudaSurfaceFormatMode {
		/// <summary>
		/// Forced format mode
		/// </summary>
		cudaFormatModeForced = 0,

		/// <summary>
		/// Auto format mode
		/// </summary>
		cudaFormatModeAuto = 1
	}

	/// <summary>
	/// (Runtime API) CUDA texture address modes.
	/// </summary>
	public enum cudaTextureAddressMode {
		/// <summary>
		/// Wrapping address mode
		/// </summary>
		cudaAddressModeWrap = 0,

		/// <summary>
		/// Clamp to edge address mode
		/// </summary>
		cudaAddressModeClamp = 1,

		/// <summary>
		/// Mirror address mode
		/// </summary>
		cudaAddressModeMirror = 2,

		/// <summary>
		/// Border address mode
		/// </summary>
		cudaAddressModeBorder = 3
	}

	/// <summary>
	/// (Runtime API) CUDA texture filter modes.
	/// </summary>
	public enum cudaTextureFilterMode {
		/// <summary>
		/// Point filter mode
		/// </summary>
		cudaFilterModePoint = 0,

		/// <summary>
		/// Linear filter mode
		/// </summary>
		cudaFilterModeLinear = 1
	}

	/// <summary>
	/// (Runtime API) CUDA texture read modes.
	/// </summary>
	public enum cudaTextureReadMode {
		/// <summary>
		/// Read texture as specified element type
		/// </summary>
		cudaReadModeElementType = 0,

		/// <summary>
		/// Read texture as normalized float
		/// </summary>
		cudaReadModeNormalizedFloat = 1
	}

	/// <summary>
	/// (Runtime API) CUDA devices corresponding to the current OpenGL context.
	/// </summary>
	public enum cudaGLDeviceList {
		/// <summary>
		/// The CUDA devices for all GPUs used by the current OpenGL context.
		/// </summary>
		cudaGLDeviceListAll = 1,
		/// <summary>
		/// The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame.
		/// </summary>
		cudaGLDeviceListCurrentFrame = 2,
		/// <summary>
		/// The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame.
		/// </summary>
		cudaGLDeviceListNextFrame = 3
	}

	/// <summary>
	/// (Runtime API) CUDA devices corresponding to a D3D9 device.
	/// </summary>
	public enum cudaD3D9DeviceList {
		/// <summary>
		/// The CUDA devices for all GPUs used by a D3D9 device.
		/// </summary>
		cudaD3D9DeviceListAll = 1,
		/// <summary>
		/// The CUDA devices for the GPUs used by a D3D9 device in its currently rendering frame.
		/// </summary>
		cudaD3D9DeviceListCurrentFrame = 2,
		/// <summary>
		/// The CUDA devices for the GPUs to be used by a D3D9 device in the next frame.
		/// </summary>
		cudaD3D9DeviceListNextFrame = 3
	}
}
