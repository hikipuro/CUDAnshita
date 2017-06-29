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


	public partial class Defines {
		public const int CU_IPC_HANDLE_SIZE = 64;
	}

	/// <summary>
	/// CUDA IPC event handle
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUipcEventHandle {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = Defines.CU_IPC_HANDLE_SIZE)]
		byte[] reserved;
	}

	/// <summary>
	/// CUDA IPC mem handle
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUipcMemHandle {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = Defines.CU_IPC_HANDLE_SIZE)]
		byte[] reserved;
	}

	/// <summary>
	/// CUDA Ipc Mem Flags
	/// </summary>
	public enum CUipcMem_flags {
		CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1 /**< Automatically enable peer access between remote devices as needed */
	}

	/// <summary>
	/// CUDA Mem Attach Flags
	/// </summary>
	public enum CUmemAttach_flags {
		CU_MEM_ATTACH_GLOBAL = 0x1, /**< Memory can be accessed by any stream on any device */
		CU_MEM_ATTACH_HOST = 0x2, /**< Memory cannot be accessed by any stream on any device */
		CU_MEM_ATTACH_SINGLE = 0x4  /**< Memory can only be accessed by a single stream on the associated device */
	}

	/// <summary>
	/// Context creation flags
	/// </summary>
	public enum CUctx_flags {
		CU_CTX_SCHED_AUTO = 0x00, /**< Automatic scheduling */
		CU_CTX_SCHED_SPIN = 0x01, /**< Set spin as default scheduling */
		CU_CTX_SCHED_YIELD = 0x02, /**< Set yield as default scheduling */
		CU_CTX_SCHED_BLOCKING_SYNC = 0x04, /**< Set blocking synchronization as default scheduling */
		CU_CTX_BLOCKING_SYNC = 0x04, /**< Set blocking synchronization as default scheduling
                                         *  \deprecated This flag was deprecated as of CUDA 4.0
                                         *  and was replaced with ::CU_CTX_SCHED_BLOCKING_SYNC. */
		CU_CTX_SCHED_MASK = 0x07,
		CU_CTX_MAP_HOST = 0x08, /**< Support mapped pinned allocations */
		CU_CTX_LMEM_RESIZE_TO_MAX = 0x10, /**< Keep local memory allocation after launch */
		CU_CTX_FLAGS_MASK = 0x1f
	}

	/// <summary>
	/// Stream creation flags
	/// </summary>
	public enum CUstream_flags {
		CU_STREAM_DEFAULT = 0x0, /**< Default stream flag */
		CU_STREAM_NON_BLOCKING = 0x1  /**< Stream does not synchronize with stream 0 (the NULL stream) */
	}

	/// <summary>
	/// Event creation flags
	/// </summary>
	public enum CUevent_flags {
		CU_EVENT_DEFAULT = 0x0, /**< Default event flag */
		CU_EVENT_BLOCKING_SYNC = 0x1, /**< Event uses blocking synchronization */
		CU_EVENT_DISABLE_TIMING = 0x2, /**< Event will not record timing data */
		CU_EVENT_INTERPROCESS = 0x4  /**< Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set */
	}

	/// <summary>
	/// Flags for ::cuStreamWaitValue32
	/// </summary>
	public enum CUstreamWaitValue_flags {
		/// <summary>
		/// Wait until (int32_t)(*addr - value) >= 0. Note this is a
		/// cyclic comparison which ignores wraparound. (Default behavior.)
		/// </summary>
		CU_STREAM_WAIT_VALUE_GEQ = 0x0,
		/// <summary>
		/// Wait until *addr == value.
		/// </summary>
		CU_STREAM_WAIT_VALUE_EQ = 0x1,
		/// <summary>
		/// Wait until (*addr & value) != 0.
		/// </summary>
		CU_STREAM_WAIT_VALUE_AND = 0x2,
		/// <summary>
		/// Follow the wait operation with a flush of outstanding remote writes. This
		/// means that, if a remote write operation is guaranteed to have reached the
		/// device before the wait can be satisfied, that write is guaranteed to be
		/// visible to downstream device work. The device is permitted to reorder
		/// remote writes internally. For example, this flag would be required if
		/// two remote writes arrive in a defined order, the wait is satisfied by the
		/// second write, and downstream work needs to observe the first write.
		/// </summary>
		CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30
	}

	/// <summary>
	/// Flags for ::cuStreamWriteValue32
	/// </summary>
	public enum CUstreamWriteValue_flags {
		/// <summary>
		/// Default behavior
		/// </summary>
		CU_STREAM_WRITE_VALUE_DEFAULT = 0x0,
		/// <summary>
		/// Permits the write to be reordered with writes which were issued
		/// before it, as a performance optimization. Normally,
		/// ::cuStreamWriteValue32 will provide a memory fence before the
		/// write, which has similar semantics to
		/// __threadfence_system() but is scoped to the stream
		/// rather than a CUDA thread.
		/// </summary>
		CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1
	}

	/// <summary>
	/// Operations for ::cuStreamBatchMemOp
	/// </summary>
	public enum CUstreamBatchMemOpType {
		CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,     /**< Represents a ::cuStreamWaitValue32 operation */
		CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,     /**< Represents a ::cuStreamWriteValue32 operation */
		CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3 /**< This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a standalone operation. */
	}

	/// <summary>
	/// Occupancy calculator flag
	/// </summary>
	public enum CUoccupancy_flags {
		CU_OCCUPANCY_DEFAULT = 0x0, /**< Default behavior */
		CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1  /**< Assume global caching is enabled and cannot be automatically turned off */
	}

	/// <summary>
	/// Array formats
	/// </summary>
	public enum CUarray_format {
		CU_AD_FORMAT_UNSIGNED_INT8 = 0x01, /**< Unsigned 8-bit integers */
		CU_AD_FORMAT_UNSIGNED_INT16 = 0x02, /**< Unsigned 16-bit integers */
		CU_AD_FORMAT_UNSIGNED_INT32 = 0x03, /**< Unsigned 32-bit integers */
		CU_AD_FORMAT_SIGNED_INT8 = 0x08, /**< Signed 8-bit integers */
		CU_AD_FORMAT_SIGNED_INT16 = 0x09, /**< Signed 16-bit integers */
		CU_AD_FORMAT_SIGNED_INT32 = 0x0a, /**< Signed 32-bit integers */
		CU_AD_FORMAT_HALF = 0x10, /**< 16-bit floating point */
		CU_AD_FORMAT_FLOAT = 0x20  /**< 32-bit floating point */
	}

	/// <summary>
	/// Texture reference addressing modes
	/// </summary>
	public enum CUaddress_mode {
		CU_TR_ADDRESS_MODE_WRAP = 0, /**< Wrapping address mode */
		CU_TR_ADDRESS_MODE_CLAMP = 1, /**< Clamp to edge address mode */
		CU_TR_ADDRESS_MODE_MIRROR = 2, /**< Mirror address mode */
		CU_TR_ADDRESS_MODE_BORDER = 3  /**< Border address mode */
	}

	/// <summary>
	/// Texture reference filtering modes
	/// </summary>
	public enum CUfilter_mode {
		CU_TR_FILTER_MODE_POINT = 0, /**< Point filter mode */
		CU_TR_FILTER_MODE_LINEAR = 1  /**< Linear filter mode */
	}

	/// <summary>
	/// Device properties
	/// </summary>
	public enum CUdevice_attribute {
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,              /**< Maximum number of threads per block */
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,                    /**< Maximum block dimension X */
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,                    /**< Maximum block dimension Y */
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,                    /**< Maximum block dimension Z */
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,                     /**< Maximum grid dimension X */
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,                     /**< Maximum grid dimension Y */
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,                     /**< Maximum grid dimension Z */
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,        /**< Maximum shared memory available per block in bytes */
		CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,            /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
		CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,              /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
		CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                         /**< Warp size in threads */
		CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                         /**< Maximum pitch in bytes allowed by memory copies */
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,           /**< Maximum number of 32-bit registers available per block */
		CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,               /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
		CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        /**< Typical clock frequency in kilohertz */
		CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                 /**< Alignment requirement for textures */
		CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,              /**< Number of multiprocessors on device */
		CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,               /**< Specifies whether there is a run time limit on kernels */
		CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,                        /**< Device is integrated with host memory */
		CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,               /**< Device can map host memory into CUDA address space */
		CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                      /**< Compute mode (See ::CUcomputemode for details) */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,           /**< Maximum 1D texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,           /**< Maximum 2D texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,          /**< Maximum 2D texture height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,           /**< Maximum 3D texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,          /**< Maximum 3D texture height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,           /**< Maximum 3D texture depth */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,   /**< Maximum 2D layered texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,  /**< Maximum 2D layered texture height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,  /**< Maximum layers in a 2D layered texture */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,     /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,    /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, /**< Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
		CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                 /**< Alignment requirement for surfaces */
		CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                /**< Device can possibly execute multiple kernels concurrently */
		CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       /**< Device has ECC support enabled */
		CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        /**< PCI bus ID of the device */
		CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     /**< PCI device ID of the device */
		CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                        /**< Device is using TCC driver model */
		CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 /**< Peak memory clock frequency in kilohertz */
		CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           /**< Global memory bus width in bits */
		CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                     /**< Size of L2 cache in bytes */
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    /**< Maximum resident threads per multiprocessor */
		CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                /**< Number of asynchronous engines */
		CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                /**< Device shares a unified address space with the host */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,   /**< Maximum 1D layered texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,  /**< Maximum layers in a 1D layered texture */
		CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                  /**< Deprecated, do not use. */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,    /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,   /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47, /**< Alternate maximum 3D texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,/**< Alternate maximum 3D texture height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49, /**< Alternate maximum 3D texture depth */
		CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                     /**< PCI domain ID of the device */
		CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,           /**< Pitch alignment requirement for textures */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,      /**< Maximum cubemap texture width/height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,  /**< Maximum cubemap layered texture width/height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54, /**< Maximum layers in a cubemap layered texture */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,           /**< Maximum 1D surface width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,           /**< Maximum 2D surface width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,          /**< Maximum 2D surface height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,           /**< Maximum 3D surface width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,          /**< Maximum 3D surface height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,           /**< Maximum 3D surface depth */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,   /**< Maximum 1D layered surface width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,  /**< Maximum layers in a 1D layered surface */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,   /**< Maximum 2D layered surface width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,  /**< Maximum 2D layered surface height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,  /**< Maximum layers in a 2D layered surface */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,      /**< Maximum cubemap surface width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,  /**< Maximum cubemap layered surface width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68, /**< Maximum layers in a cubemap layered surface */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,    /**< Maximum 1D linear texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,    /**< Maximum 2D linear texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,   /**< Maximum 2D linear texture height */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,    /**< Maximum 2D linear texture pitch in bytes */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73, /**< Maximum mipmapped 2D texture width */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,/**< Maximum mipmapped 2D texture height */
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,          /**< Major compute capability version number */
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,          /**< Minor compute capability version number */
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77, /**< Maximum mipmapped 1D texture width */
		CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,       /**< Device supports stream priorities */
		CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,         /**< Device supports caching globals in L1 */
		CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,          /**< Device supports caching locals in L1 */
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,  /**< Maximum shared memory available per multiprocessor in bytes */
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,  /**< Maximum number of 32-bit registers available per multiprocessor */
		CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,                    /**< Device can allocate managed memory on this system */
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,                    /**< Device is on a multi-GPU board */
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,           /**< Unique id for a group of devices on the same multi-GPU board */
		CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,       /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
		CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,  /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
		CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,            /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
		CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,         /**< Device can coherently access managed memory concurrently with the CPU */
		CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,      /**< Device supports compute preemption. */
		CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
		CU_DEVICE_ATTRIBUTE_MAX
	}

	/// <summary>
	/// Legacy device properties
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUdevprop {
		int maxThreadsPerBlock;     /**< Maximum number of threads per block */
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		int[] maxThreadsDim;       /**< Maximum size of each dimension of a block */
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		int[] maxGridSize;         /**< Maximum size of each dimension of a grid */
		int sharedMemPerBlock;      /**< Shared memory available per block in bytes */
		int totalConstantMemory;    /**< Constant memory available on device in bytes */
		int SIMDWidth;              /**< Warp size in threads */
		int memPitch;               /**< Maximum pitch in bytes allowed by memory copies */
		int regsPerBlock;           /**< 32-bit registers available per block */
		int clockRate;              /**< Clock frequency in kilohertz */
		int textureAlign;           /**< Alignment requirement for textures */
	}

	/// <summary>
	/// Pointer information
	/// </summary>
	public enum CUpointer_attribute {
		CU_POINTER_ATTRIBUTE_CONTEXT = 1,        /**< The ::CUcontext on which a pointer was allocated or registered */
		CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,    /**< The ::CUmemorytype describing the physical location of a pointer */
		CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3, /**< The address at which a pointer's memory may be accessed on the device */
		CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,   /**< The address at which a pointer's memory may be accessed on the host */
		CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,     /**< A pair of tokens for use with the nv-p2p.h Linux kernel interface */
		CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,    /**< Synchronize every synchronous memory operation initiated on this region */
		CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,      /**< A process-wide unique ID for an allocated memory region*/
		CU_POINTER_ATTRIBUTE_IS_MANAGED = 8      /**< Indicates if the pointer points to managed memory */
	}

	/// <summary>
	/// Function properties
	/// </summary>
	public enum CUfunction_attribute {
		/**
		 * The maximum number of threads per block, beyond which a launch of the
		 * function would fail. This number depends on both the function and the
		 * device on which the function is currently loaded.
		 */
		CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

		/**
		 * The size in bytes of statically-allocated shared memory required by
		 * this function. This does not include dynamically-allocated shared
		 * memory requested by the user at runtime.
		 */
		CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

		/**
		 * The size in bytes of user-allocated constant memory required by this
		 * function.
		 */
		CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

		/**
		 * The size in bytes of local memory used by each thread of this function.
		 */
		CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

		/**
		 * The number of registers used by each thread of this function.
		 */
		CU_FUNC_ATTRIBUTE_NUM_REGS = 4,

		/**
		 * The PTX virtual architecture version for which the function was
		 * compiled. This value is the major PTX version * 10 + the minor PTX
		 * version, so a PTX version 1.3 function would return the value 13.
		 * Note that this may return the undefined value of 0 for cubins
		 * compiled prior to CUDA 3.0.
		 */
		CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,

		/**
		 * The binary architecture version for which the function was compiled.
		 * This value is the major binary version * 10 + the minor binary version,
		 * so a binary version 1.3 function would return the value 13. Note that
		 * this will return a value of 10 for legacy cubins that do not have a
		 * properly-encoded binary architecture version.
		 */
		CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,

		/**
		 * The attribute to indicate whether the function has been compiled with 
		 * user specified option "-Xptxas --dlcm=ca" set .
		 */
		CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,

		CU_FUNC_ATTRIBUTE_MAX
	}

	/// <summary>
	/// Function cache configurations
	/// </summary>
	public enum CUfunc_cache {
		CU_FUNC_CACHE_PREFER_NONE = 0x00, /**< no preference for shared memory or L1 (default) */
		CU_FUNC_CACHE_PREFER_SHARED = 0x01, /**< prefer larger shared memory and smaller L1 cache */
		CU_FUNC_CACHE_PREFER_L1 = 0x02, /**< prefer larger L1 cache and smaller shared memory */
		CU_FUNC_CACHE_PREFER_EQUAL = 0x03  /**< prefer equal sized L1 cache and shared memory */
	}

	/// <summary>
	/// Shared memory configurations
	/// </summary>
	public enum CUsharedconfig {
		CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00, /**< set default shared memory bank size */
		CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01, /**< set shared memory bank width to four bytes */
		CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02  /**< set shared memory bank width to eight bytes */
	}

	/// <summary>
	/// Memory types
	/// </summary>
	public enum CUmemorytype {
		CU_MEMORYTYPE_HOST = 0x01,    /**< Host memory */
		CU_MEMORYTYPE_DEVICE = 0x02,    /**< Device memory */
		CU_MEMORYTYPE_ARRAY = 0x03,    /**< Array memory */
		CU_MEMORYTYPE_UNIFIED = 0x04     /**< Unified device or host memory */
	}

	/// <summary>
	/// Compute Modes
	/// </summary>
	public enum CUcomputemode {
		CU_COMPUTEMODE_DEFAULT = 0, /**< Default compute mode (Multiple contexts allowed per device) */
		CU_COMPUTEMODE_EXCLUSIVE = 1,
		CU_COMPUTEMODE_PROHIBITED = 2, /**< Compute-prohibited mode (No contexts can be created on this device at this time) */
		CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3  /**< Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time) */
	}

	/// <summary>
	/// Memory advise values
	/// </summary>
	public enum CUmem_advise {
		CU_MEM_ADVISE_SET_READ_MOSTLY = 1, /**< Data will mostly be read and only occassionally be written to */
		CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2, /**< Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY */
		CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3, /**< Set the preferred location for the data as the specified device */
		CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4, /**< Clear the preferred location for the data */
		CU_MEM_ADVISE_SET_ACCESSED_BY = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
		CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
	}

	/// <summary>
	/// 
	/// </summary>
	public enum CUmem_range_attribute {
		CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1, /**< Whether the range will mostly be read and only occassionally be written to */
		CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2, /**< The preferred location of the range */
		CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3, /**< Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device */
		CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4  /**< The last location to which the range was prefetched */
	}

	/// <summary>
	/// Online compiler and linker options
	/// </summary>
	public enum CUjit_option {
		/**
		 * Max number of registers that a thread may use.\n
		 * Option type: unsigned int\n
		 * Applies to: compiler only
		 */
		CU_JIT_MAX_REGISTERS = 0,

		/**
		 * IN: Specifies minimum number of threads per block to target compilation
		 * for\n
		 * OUT: Returns the number of threads the compiler actually targeted.
		 * This restricts the resource utilization fo the compiler (e.g. max
		 * registers) such that a block with the given number of threads should be
		 * able to launch based on register limitations. Note, this option does not
		 * currently take into account any other resource limitations, such as
		 * shared memory utilization.\n
		 * Cannot be combined with ::CU_JIT_TARGET.\n
		 * Option type: unsigned int\n
		 * Applies to: compiler only
		 */
		CU_JIT_THREADS_PER_BLOCK,

		/**
		 * Overwrites the option value with the total wall clock time, in
		 * milliseconds, spent in the compiler and linker\n
		 * Option type: float\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_WALL_TIME,

		/**
		 * Pointer to a buffer in which to print any log messages
		 * that are informational in nature (the buffer size is specified via
		 * option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
		 * Option type: char *\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_INFO_LOG_BUFFER,

		/**
		 * IN: Log buffer size in bytes.  Log messages will be capped at this size
		 * (including null terminator)\n
		 * OUT: Amount of log buffer filled with messages\n
		 * Option type: unsigned int\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

		/**
		 * Pointer to a buffer in which to print any log messages that
		 * reflect errors (the buffer size is specified via option
		 * ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
		 * Option type: char *\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_ERROR_LOG_BUFFER,

		/**
		 * IN: Log buffer size in bytes.  Log messages will be capped at this size
		 * (including null terminator)\n
		 * OUT: Amount of log buffer filled with messages\n
		 * Option type: unsigned int\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

		/**
		 * Level of optimizations to apply to generated code (0 - 4), with 4
		 * being the default and highest level of optimizations.\n
		 * Option type: unsigned int\n
		 * Applies to: compiler only
		 */
		CU_JIT_OPTIMIZATION_LEVEL,

		/**
		 * No option value required. Determines the target based on the current
		 * attached context (default)\n
		 * Option type: No option value needed\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_TARGET_FROM_CUCONTEXT,

		/**
		 * Target is chosen based on supplied ::CUjit_target.  Cannot be
		 * combined with ::CU_JIT_THREADS_PER_BLOCK.\n
		 * Option type: unsigned int for enumerated type ::CUjit_target\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_TARGET,

		/**
		 * Specifies choice of fallback strategy if matching cubin is not found.
		 * Choice is based on supplied ::CUjit_fallback.  This option cannot be
		 * used with cuLink* APIs as the linker requires exact matches.\n
		 * Option type: unsigned int for enumerated type ::CUjit_fallback\n
		 * Applies to: compiler only
		 */
		CU_JIT_FALLBACK_STRATEGY,

		/**
		 * Specifies whether to create debug information in output (-g)
		 * (0: false, default)\n
		 * Option type: int\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_GENERATE_DEBUG_INFO,

		/**
		 * Generate verbose log messages (0: false, default)\n
		 * Option type: int\n
		 * Applies to: compiler and linker
		 */
		CU_JIT_LOG_VERBOSE,

		/**
		 * Generate line number information (-lineinfo) (0: false, default)\n
		 * Option type: int\n
		 * Applies to: compiler only
		 */
		CU_JIT_GENERATE_LINE_INFO,

		/**
		 * Specifies whether to enable caching explicitly (-dlcm) \n
		 * Choice is based on supplied ::CUjit_cacheMode_enum.\n
		 * Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum\n
		 * Applies to: compiler only
		 */
		CU_JIT_CACHE_MODE,

		/**
		 * The below jit options are used for internal purposes only, in this version of CUDA
		 */
		CU_JIT_NEW_SM3X_OPT,
		CU_JIT_FAST_COMPILE,

		CU_JIT_NUM_OPTIONS
	}

	/// <summary>
	/// Online compilation targets
	/// </summary>
	public enum CUjit_target {
		CU_TARGET_COMPUTE_10 = 10,       /**< Compute device class 1.0 */
		CU_TARGET_COMPUTE_11 = 11,       /**< Compute device class 1.1 */
		CU_TARGET_COMPUTE_12 = 12,       /**< Compute device class 1.2 */
		CU_TARGET_COMPUTE_13 = 13,       /**< Compute device class 1.3 */
		CU_TARGET_COMPUTE_20 = 20,       /**< Compute device class 2.0 */
		CU_TARGET_COMPUTE_21 = 21,       /**< Compute device class 2.1 */
		CU_TARGET_COMPUTE_30 = 30,       /**< Compute device class 3.0 */
		CU_TARGET_COMPUTE_32 = 32,       /**< Compute device class 3.2 */
		CU_TARGET_COMPUTE_35 = 35,       /**< Compute device class 3.5 */
		CU_TARGET_COMPUTE_37 = 37,       /**< Compute device class 3.7 */
		CU_TARGET_COMPUTE_50 = 50,       /**< Compute device class 5.0 */
		CU_TARGET_COMPUTE_52 = 52,       /**< Compute device class 5.2 */
		CU_TARGET_COMPUTE_53 = 53,       /**< Compute device class 5.3 */
		CU_TARGET_COMPUTE_60 = 60,       /**< Compute device class 6.0. This must be removed for CUDA 7.0 toolkit. See bug 1518217. */
		CU_TARGET_COMPUTE_61 = 61,       /**< Compute device class 6.1. This must be removed for CUDA 7.0 toolkit.*/
		CU_TARGET_COMPUTE_62 = 62        /**< Compute device class 6.2. This must be removed for CUDA 7.0 toolkit.*/
	}

	/// <summary>
	/// Cubin matching fallback strategies
	/// </summary>
	public enum CUjit_fallback {
		CU_PREFER_PTX = 0,  /**< Prefer to compile ptx if exact binary match not found */
		CU_PREFER_BINARY    /**< Prefer to fall back to compatible binary code if exact match not found */
	}

	/// <summary>
	/// Caching modes for dlcm 
	/// </summary>
	public enum CUjit_cacheMode {
		CU_JIT_CACHE_OPTION_NONE = 0, /**< Compile with no -dlcm flag specified */
		CU_JIT_CACHE_OPTION_CG,       /**< Compile with L1 cache disabled */
		CU_JIT_CACHE_OPTION_CA        /**< Compile with L1 cache enabled */
	}

	/// <summary>
	/// Device code formats
	/// </summary>
	public enum CUjitInputType {
		/**
		 * Compiled device-class-specific device code\n
		 * Applicable options: none
		 */
		CU_JIT_INPUT_CUBIN = 0,

		/**
		 * PTX source code\n
		 * Applicable options: PTX compiler options
		 */
		CU_JIT_INPUT_PTX,

		/**
		 * Bundle of multiple cubins and/or PTX of some device code\n
		 * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
		 */
		CU_JIT_INPUT_FATBINARY,

		/**
		 * Host object with embedded device code\n
		 * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
		 */
		CU_JIT_INPUT_OBJECT,

		/**
		 * Archive of host objects with embedded device code\n
		 * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
		 */
		CU_JIT_INPUT_LIBRARY,

		CU_JIT_NUM_INPUT_TYPES
	}

	/// <summary>
	/// Flags to register a graphics resource
	/// </summary>
	public enum CUgraphicsRegisterFlags {
		CU_GRAPHICS_REGISTER_FLAGS_NONE = 0x00,
		CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 0x01,
		CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02,
		CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04,
		CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08
	}

	/// <summary>
	/// Flags for mapping and unmapping interop resources
	/// </summary>
	public enum CUgraphicsMapResourceFlags {
		CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00,
		CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01,
		CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
	}

	/// <summary>
	/// Array indices for cube faces
	/// </summary>
	public enum CUarray_cubemap_face {
		CU_CUBEMAP_FACE_POSITIVE_X = 0x00, /**< Positive X face of cubemap */
		CU_CUBEMAP_FACE_NEGATIVE_X = 0x01, /**< Negative X face of cubemap */
		CU_CUBEMAP_FACE_POSITIVE_Y = 0x02, /**< Positive Y face of cubemap */
		CU_CUBEMAP_FACE_NEGATIVE_Y = 0x03, /**< Negative Y face of cubemap */
		CU_CUBEMAP_FACE_POSITIVE_Z = 0x04, /**< Positive Z face of cubemap */
		CU_CUBEMAP_FACE_NEGATIVE_Z = 0x05  /**< Negative Z face of cubemap */
	}

	/// <summary>
	/// Limits
	/// </summary>
	public enum CUlimit {
		CU_LIMIT_STACK_SIZE = 0x00, /**< GPU thread stack size */
		CU_LIMIT_PRINTF_FIFO_SIZE = 0x01, /**< GPU printf FIFO size */
		CU_LIMIT_MALLOC_HEAP_SIZE = 0x02, /**< GPU malloc heap size */
		CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03, /**< GPU device runtime launch synchronize depth */
		CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04, /**< GPU device runtime pending launch count */
		CU_LIMIT_MAX
	}

	/// <summary>
	/// Resource types
	/// </summary>
	public enum CUresourcetype {
		CU_RESOURCE_TYPE_ARRAY = 0x00, /**< Array resoure */
		CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
		CU_RESOURCE_TYPE_LINEAR = 0x02, /**< Linear resource */
		CU_RESOURCE_TYPE_PITCH2D = 0x03  /**< Pitch 2D resource */
	}

	/// <summary>
	/// Error codes
	/// </summary>
	public enum CUresult {
		/// <summary>
		/// The API call returned with no errors. In the case of query calls, this
		/// can also mean that the operation being queried is complete (see
		/// ::cuEventQuery() and ::cuStreamQuery()).
		/// </summary>
		CUDA_SUCCESS = 0,

		/**
		 * This indicates that one or more of the parameters passed to the API call
		 * is not within an acceptable range of values.
		 */
		CUDA_ERROR_INVALID_VALUE = 1,

		/**
		 * The API call failed because it was unable to allocate enough memory to
		 * perform the requested operation.
		 */
		CUDA_ERROR_OUT_OF_MEMORY = 2,

		/**
		 * This indicates that the CUDA driver has not been initialized with
		 * ::cuInit() or that initialization has failed.
		 */
		CUDA_ERROR_NOT_INITIALIZED = 3,

		/**
		 * This indicates that the CUDA driver is in the process of shutting down.
		 */
		CUDA_ERROR_DEINITIALIZED = 4,

		/**
		 * This indicates profiler is not initialized for this run. This can
		 * happen when the application is running with external profiling tools
		 * like visual profiler.
		 */
		CUDA_ERROR_PROFILER_DISABLED = 5,

		/**
		 * \deprecated
		 * This error return is deprecated as of CUDA 5.0. It is no longer an error
		 * to attempt to enable/disable the profiling via ::cuProfilerStart or
		 * ::cuProfilerStop without initialization.
		 */
		CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,

		/**
		 * \deprecated
		 * This error return is deprecated as of CUDA 5.0. It is no longer an error
		 * to call cuProfilerStart() when profiling is already enabled.
		 */
		CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,

		/**
		 * \deprecated
		 * This error return is deprecated as of CUDA 5.0. It is no longer an error
		 * to call cuProfilerStop() when profiling is already disabled.
		 */
		CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,

		/**
		 * This indicates that no CUDA-capable devices were detected by the installed
		 * CUDA driver.
		 */
		CUDA_ERROR_NO_DEVICE = 100,

		/**
		 * This indicates that the device ordinal supplied by the user does not
		 * correspond to a valid CUDA device.
		 */
		CUDA_ERROR_INVALID_DEVICE = 101,


		/**
		 * This indicates that the device kernel image is invalid. This can also
		 * indicate an invalid CUDA module.
		 */
		CUDA_ERROR_INVALID_IMAGE = 200,

		/**
		 * This most frequently indicates that there is no context bound to the
		 * current thread. This can also be returned if the context passed to an
		 * API call is not a valid handle (such as a context that has had
		 * ::cuCtxDestroy() invoked on it). This can also be returned if a user
		 * mixes different API versions (i.e. 3010 context with 3020 API calls).
		 * See ::cuCtxGetApiVersion() for more details.
		 */
		CUDA_ERROR_INVALID_CONTEXT = 201,

		/**
		 * This indicated that the context being supplied as a parameter to the
		 * API call was already the active context.
		 * \deprecated
		 * This error return is deprecated as of CUDA 3.2. It is no longer an
		 * error to attempt to push the active context via ::cuCtxPushCurrent().
		 */
		CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,

		/**
		 * This indicates that a map or register operation has failed.
		 */
		CUDA_ERROR_MAP_FAILED = 205,

		/**
		 * This indicates that an unmap or unregister operation has failed.
		 */
		CUDA_ERROR_UNMAP_FAILED = 206,

		/**
		 * This indicates that the specified array is currently mapped and thus
		 * cannot be destroyed.
		 */
		CUDA_ERROR_ARRAY_IS_MAPPED = 207,

		/**
		 * This indicates that the resource is already mapped.
		 */
		CUDA_ERROR_ALREADY_MAPPED = 208,

		/**
		 * This indicates that there is no kernel image available that is suitable
		 * for the device. This can occur when a user specifies code generation
		 * options for a particular CUDA source file that do not include the
		 * corresponding device configuration.
		 */
		CUDA_ERROR_NO_BINARY_FOR_GPU = 209,

		/**
		 * This indicates that a resource has already been acquired.
		 */
		CUDA_ERROR_ALREADY_ACQUIRED = 210,

		/**
		 * This indicates that a resource is not mapped.
		 */
		CUDA_ERROR_NOT_MAPPED = 211,

		/**
		 * This indicates that a mapped resource is not available for access as an
		 * array.
		 */
		CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,

		/**
		 * This indicates that a mapped resource is not available for access as a
		 * pointer.
		 */
		CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,

		/**
		 * This indicates that an uncorrectable ECC error was detected during
		 * execution.
		 */
		CUDA_ERROR_ECC_UNCORRECTABLE = 214,

		/**
		 * This indicates that the ::CUlimit passed to the API call is not
		 * supported by the active device.
		 */
		CUDA_ERROR_UNSUPPORTED_LIMIT = 215,

		/**
		 * This indicates that the ::CUcontext passed to the API call can
		 * only be bound to a single CPU thread at a time but is already 
		 * bound to a CPU thread.
		 */
		CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,

		/**
		 * This indicates that peer access is not supported across the given
		 * devices.
		 */
		CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,

		/**
		 * This indicates that a PTX JIT compilation failed.
		 */
		CUDA_ERROR_INVALID_PTX = 218,

		/**
		 * This indicates an error with OpenGL or DirectX context.
		 */
		CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,

		/**
		* This indicates that an uncorrectable NVLink error was detected during the
		* execution.
		*/
		CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,

		/**
		 * This indicates that the device kernel source is invalid.
		 */
		CUDA_ERROR_INVALID_SOURCE = 300,

		/**
		 * This indicates that the file specified was not found.
		 */
		CUDA_ERROR_FILE_NOT_FOUND = 301,

		/**
		 * This indicates that a link to a shared object failed to resolve.
		 */
		CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

		/**
		 * This indicates that initialization of a shared object failed.
		 */
		CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,

		/**
		 * This indicates that an OS call failed.
		 */
		CUDA_ERROR_OPERATING_SYSTEM = 304,

		/**
		 * This indicates that a resource handle passed to the API call was not
		 * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
		 */
		CUDA_ERROR_INVALID_HANDLE = 400,

		/**
		 * This indicates that a named symbol was not found. Examples of symbols
		 * are global/constant variable names, texture names, and surface names.
		 */
		CUDA_ERROR_NOT_FOUND = 500,

		/**
		 * This indicates that asynchronous operations issued previously have not
		 * completed yet. This result is not actually an error, but must be indicated
		 * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
		 * may return this value include ::cuEventQuery() and ::cuStreamQuery().
		 */
		CUDA_ERROR_NOT_READY = 600,

		/**
		 * While executing a kernel, the device encountered a
		 * load or store instruction on an invalid memory address.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_ILLEGAL_ADDRESS = 700,

		/**
		 * This indicates that a launch did not occur because it did not have
		 * appropriate resources. This error usually indicates that the user has
		 * attempted to pass too many arguments to the device kernel, or the
		 * kernel launch specifies too many threads for the kernel's register
		 * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
		 * when a 32-bit int is expected) is equivalent to passing too many
		 * arguments and can also result in this error.
		 */
		CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,

		/**
		 * This indicates that the device kernel took too long to execute. This can
		 * only occur if timeouts are enabled - see the device attribute
		 * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_LAUNCH_TIMEOUT = 702,

		/**
		 * This error indicates a kernel launch that uses an incompatible texturing
		 * mode.
		 */
		CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,

		/**
		 * This error indicates that a call to ::cuCtxEnablePeerAccess() is
		 * trying to re-enable peer access to a context which has already
		 * had peer access to it enabled.
		 */
		CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

		/**
		 * This error indicates that ::cuCtxDisablePeerAccess() is 
		 * trying to disable peer access which has not been enabled yet 
		 * via ::cuCtxEnablePeerAccess(). 
		 */
		CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,

		/**
		 * This error indicates that the primary context for the specified device
		 * has already been initialized.
		 */
		CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,

		/**
		 * This error indicates that the context current to the calling thread
		 * has been destroyed using ::cuCtxDestroy, or is a primary context which
		 * has not yet been initialized.
		 */
		CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,

		/**
		 * A device-side assert triggered during kernel execution. The context
		 * cannot be used anymore, and must be destroyed. All existing device 
		 * memory allocations from this context are invalid and must be 
		 * reconstructed if the program is to continue using CUDA.
		 */
		CUDA_ERROR_ASSERT = 710,

		/**
		 * This error indicates that the hardware resources required to enable
		 * peer access have been exhausted for one or more of the devices 
		 * passed to ::cuCtxEnablePeerAccess().
		 */
		CUDA_ERROR_TOO_MANY_PEERS = 711,

		/**
		 * This error indicates that the memory range passed to ::cuMemHostRegister()
		 * has already been registered.
		 */
		CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

		/**
		 * This error indicates that the pointer passed to ::cuMemHostUnregister()
		 * does not correspond to any currently registered memory region.
		 */
		CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,

		/**
		 * While executing a kernel, the device encountered a stack error.
		 * This can be due to stack corruption or exceeding the stack size limit.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_HARDWARE_STACK_ERROR = 714,

		/**
		 * While executing a kernel, the device encountered an illegal instruction.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,

		/**
		 * While executing a kernel, the device encountered a load or store instruction
		 * on a memory address which is not aligned.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_MISALIGNED_ADDRESS = 716,

		/**
		 * While executing a kernel, the device encountered an instruction
		 * which can only operate on memory locations in certain address spaces
		 * (global, shared, or local), but was supplied a memory address not
		 * belonging to an allowed address space.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,

		/**
		 * While executing a kernel, the device program counter wrapped its address space.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_INVALID_PC = 718,

		/**
		 * An exception occurred on the device while executing a kernel. Common
		 * causes include dereferencing an invalid device pointer and accessing
		 * out of bounds shared memory.
		 * This leaves the process in an inconsistent state and any further CUDA work
		 * will return the same error. To continue using CUDA, the process must be terminated
		 * and relaunched.
		 */
		CUDA_ERROR_LAUNCH_FAILED = 719,


		/**
		 * This error indicates that the attempted operation is not permitted.
		 */
		CUDA_ERROR_NOT_PERMITTED = 800,

		/**
		 * This error indicates that the attempted operation is not supported
		 * on the current system or device.
		 */
		CUDA_ERROR_NOT_SUPPORTED = 801,

		/**
		 * This indicates that an unknown internal error has occurred.
		 */
		CUDA_ERROR_UNKNOWN = 999
	}

	/// <summary>
	/// P2P Attributes
	/// </summary>
	public enum CUdevice_P2PAttribute {
		CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01, /**< A relative value indicating the performance of the link between two devices */
		CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02, /**< P2P Access is enable */
		CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03  /**< Atomic operation over the link supported */
	}

	/// <summary>
	/// 2D memory copy parameters
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_MEMCPY2D {
		size_t srcXInBytes;         /**< Source X in bytes */
		size_t srcY;                /**< Source Y */

		CUmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
		IntPtr srcHost;             /**< Source host pointer */
		CUdeviceptr srcDevice;      /**< Source device pointer */
		CUarray srcArray;           /**< Source array reference */
		size_t srcPitch;            /**< Source pitch (ignored when src is array) */

		size_t dstXInBytes;         /**< Destination X in bytes */
		size_t dstY;                /**< Destination Y */

		CUmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
		IntPtr dstHost;             /**< Destination host pointer */
		CUdeviceptr dstDevice;      /**< Destination device pointer */
		CUarray dstArray;           /**< Destination array reference */
		size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */

		size_t WidthInBytes;        /**< Width of 2D memory copy in bytes */
		size_t Height;              /**< Height of 2D memory copy */
	}

	/// <summary>
	/// 3D memory copy parameters
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_MEMCPY3D {
		size_t srcXInBytes;         /**< Source X in bytes */
		size_t srcY;                /**< Source Y */
		size_t srcZ;                /**< Source Z */
		size_t srcLOD;              /**< Source LOD */
		CUmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
		IntPtr srcHost;             /**< Source host pointer */
		CUdeviceptr srcDevice;      /**< Source device pointer */
		CUarray srcArray;           /**< Source array reference */
		IntPtr reserved0;           /**< Must be NULL */
		size_t srcPitch;            /**< Source pitch (ignored when src is array) */
		size_t srcHeight;           /**< Source height (ignored when src is array; may be 0 if Depth==1) */

		size_t dstXInBytes;         /**< Destination X in bytes */
		size_t dstY;                /**< Destination Y */
		size_t dstZ;                /**< Destination Z */
		size_t dstLOD;              /**< Destination LOD */
		CUmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
		IntPtr dstHost;             /**< Destination host pointer */
		CUdeviceptr dstDevice;      /**< Destination device pointer */
		CUarray dstArray;           /**< Destination array reference */
		IntPtr reserved1;           /**< Must be NULL */
		size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */
		size_t dstHeight;           /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

		size_t WidthInBytes;        /**< Width of 3D memory copy in bytes */
		size_t Height;              /**< Height of 3D memory copy */
		size_t Depth;               /**< Depth of 3D memory copy */
	}

	/// <summary>
	/// 3D memory cross-context copy parameters
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_MEMCPY3D_PEER {
		size_t srcXInBytes;         /**< Source X in bytes */
		size_t srcY;                /**< Source Y */
		size_t srcZ;                /**< Source Z */
		size_t srcLOD;              /**< Source LOD */
		CUmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
		IntPtr srcHost;             /**< Source host pointer */
		CUdeviceptr srcDevice;      /**< Source device pointer */
		CUarray srcArray;           /**< Source array reference */
		CUcontext srcContext;       /**< Source context (ignored with srcMemoryType is ::CU_MEMORYTYPE_ARRAY) */
		size_t srcPitch;            /**< Source pitch (ignored when src is array) */
		size_t srcHeight;           /**< Source height (ignored when src is array; may be 0 if Depth==1) */

		size_t dstXInBytes;         /**< Destination X in bytes */
		size_t dstY;                /**< Destination Y */
		size_t dstZ;                /**< Destination Z */
		size_t dstLOD;              /**< Destination LOD */
		CUmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
		IntPtr dstHost;             /**< Destination host pointer */
		CUdeviceptr dstDevice;      /**< Destination device pointer */
		CUarray dstArray;           /**< Destination array reference */
		CUcontext dstContext;       /**< Destination context (ignored with dstMemoryType is ::CU_MEMORYTYPE_ARRAY) */
		size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */
		size_t dstHeight;           /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

		size_t WidthInBytes;        /**< Width of 3D memory copy in bytes */
		size_t Height;              /**< Height of 3D memory copy */
		size_t Depth;               /**< Depth of 3D memory copy */
	}

	/// <summary>
	/// Array descriptor
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_ARRAY_DESCRIPTOR {
		size_t Width;             /**< Width of array */
		size_t Height;            /**< Height of array */

		CUarray_format Format;    /**< Array format */
		uint NumChannels; /**< Channels per array element */
	}

	/// <summary>
	/// 3D array descriptor
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_ARRAY3D_DESCRIPTOR {
		size_t Width;             /**< Width of 3D array */
		size_t Height;            /**< Height of 3D array */
		size_t Depth;             /**< Depth of 3D array */

		CUarray_format Format;    /**< Array format */
		uint NumChannels; /**< Channels per array element */
		uint Flags;       /**< Flags */
	}

	/// <summary>
	/// Resource view format
	/// </summary>
	public enum CUresourceViewFormat {
		CU_RES_VIEW_FORMAT_NONE = 0x00, /**< No resource view format (use underlying resource format) */
		CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01, /**< 1 channel unsigned 8-bit integers */
		CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02, /**< 2 channel unsigned 8-bit integers */
		CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03, /**< 4 channel unsigned 8-bit integers */
		CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04, /**< 1 channel signed 8-bit integers */
		CU_RES_VIEW_FORMAT_SINT_2X8 = 0x05, /**< 2 channel signed 8-bit integers */
		CU_RES_VIEW_FORMAT_SINT_4X8 = 0x06, /**< 4 channel signed 8-bit integers */
		CU_RES_VIEW_FORMAT_UINT_1X16 = 0x07, /**< 1 channel unsigned 16-bit integers */
		CU_RES_VIEW_FORMAT_UINT_2X16 = 0x08, /**< 2 channel unsigned 16-bit integers */
		CU_RES_VIEW_FORMAT_UINT_4X16 = 0x09, /**< 4 channel unsigned 16-bit integers */
		CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0a, /**< 1 channel signed 16-bit integers */
		CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0b, /**< 2 channel signed 16-bit integers */
		CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0c, /**< 4 channel signed 16-bit integers */
		CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0d, /**< 1 channel unsigned 32-bit integers */
		CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0e, /**< 2 channel unsigned 32-bit integers */
		CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0f, /**< 4 channel unsigned 32-bit integers */
		CU_RES_VIEW_FORMAT_SINT_1X32 = 0x10, /**< 1 channel signed 32-bit integers */
		CU_RES_VIEW_FORMAT_SINT_2X32 = 0x11, /**< 2 channel signed 32-bit integers */
		CU_RES_VIEW_FORMAT_SINT_4X32 = 0x12, /**< 4 channel signed 32-bit integers */
		CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13, /**< 1 channel 16-bit floating point */
		CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14, /**< 2 channel 16-bit floating point */
		CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15, /**< 4 channel 16-bit floating point */
		CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16, /**< 1 channel 32-bit floating point */
		CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17, /**< 2 channel 32-bit floating point */
		CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18, /**< 4 channel 32-bit floating point */
		CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19, /**< Block compressed 1 */
		CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a, /**< Block compressed 2 */
		CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b, /**< Block compressed 3 */
		CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c, /**< Block compressed 4 unsigned */
		CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d, /**< Block compressed 4 signed */
		CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e, /**< Block compressed 5 unsigned */
		CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f, /**< Block compressed 5 signed */
		CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20, /**< Block compressed 6 unsigned half-float */
		CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21, /**< Block compressed 6 signed half-float */
		CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22  /**< Block compressed 7 */
	}
}
