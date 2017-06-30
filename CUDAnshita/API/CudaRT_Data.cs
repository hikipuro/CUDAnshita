using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using System.Text;
	using cudaArray_t = IntPtr;
	using cudaMipmappedArray_t = IntPtr;
	using size_t = Int64;

	public partial class Defines {
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
	/// CUDA device properties
	/// </summary>
	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
	public struct cudaDeviceProp {
		/// <summary>
		/// ASCII string identifying device
		/// </summary>
		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
		public string name;

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
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct cudaEglFrame {
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct cudaEglPlaneDesc {
	}

	/// <summary>
	/// CUDA function attributes
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
	/// CUDA IPC event handle
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaIpcEventHandle {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = Defines.CUDA_IPC_HANDLE_SIZE)]
		public byte[] reserved;
	}

	/// <summary>
	/// CUDA IPC memory handle
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaIpcMemHandle {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = Defines.CUDA_IPC_HANDLE_SIZE)]
		public byte[] reserved;
	}

	/// <summary>
	/// CUDA 3D memory copying parameters
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
	/// CUDA 3D cross-device memory copying parameters
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
	/// CUDA pointer attributes
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
	/// CUDA Pitched memory pointer
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
	/// CUDA extent
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
	/// CUDA 3D position
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
	/// CUDA resource descriptor
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaResourceDesc {
		/// <summary>
		/// Resource type
		/// </summary>
		public cudaResourceType resType;

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
	/// CUDA resource view descriptor
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
	/// CUDA texture reference
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
	/// CUDA texture descriptor
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
	/// CUDA Surface reference
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct surfaceReference {
		/// <summary>
		/// Channel descriptor for surface reference
		/// </summary>
		cudaChannelFormatDesc channelDesc;
	}


	/// <summary>
	/// Channel format kind
	/// </summary>
	public enum cudaChannelFormatKind {
		/// <summary>
		/// Signed channel format
		/// </summary>
		cudaChannelFormatKindSigned = 0,

		/// <summary>
		/// Unsigned channel format
		/// </summary>
		cudaChannelFormatKindUnsigned = 1,

		/// <summary>
		/// Float channel format
		/// </summary>
		cudaChannelFormatKindFloat = 2,

		/// <summary>
		/// No channel format
		/// </summary>
		cudaChannelFormatKindNone = 3
	}

	/// <summary>
	/// CUDA device compute modes
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
	/// CUDA device attributes
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
		cudaDevAttrCanUseHostPointerForRegisteredMem = 91
	}

	/// <summary>
	/// CUDA device P2P attributes
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
		cudaDevP2PAttrNativeAtomicSupported = 3
	}

	/// <summary>
	/// CUDA EGL Color Format - The different planar and multiplanar formats currently supported for CUDA_EGL interops.
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
		cudaEglColorFormatUYVY422 = 13
	}

	/// <summary>
	/// CUDA EglFrame type - array or pointer
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
	/// Resource location flags- sysmem or vidmem
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
	/// CUDA graphics interop array indices for cube maps
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
	/// CUDA graphics interop register flags
	/// </summary>
	public enum cudaGraphicsMapFlags {
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
	/// CUDA Limits
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
		cudaLimitDevRuntimePendingLaunchCount = 0x04
	}

	/// <summary>
	/// CUDA range attributes
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
	/// CUDA memory copy types
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
	/// CUDA Memory Advise values
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
	/// CUDA memory types
	/// </summary>
	public enum cudaMemoryType {
		/// <summary>
		/// Host memory
		/// </summary>
		cudaMemoryTypeHost = 1,

		/// <summary>
		/// Device memory
		/// </summary>
		cudaMemoryTypeDevice = 2
	}

	/// <summary>
	/// CUDA Profiler Output modes
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
	/// CUDA resource types
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
	/// CUDA texture resource view formats
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
	/// CUDA Surface format modes
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
	/// CUDA texture address modes
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
	/// CUDA texture filter modes
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
	/// CUDA texture read modes
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
}
