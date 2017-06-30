using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using CUdevice = Int32;
	using CUdeviceptr = IntPtr;
	using CUcontext = IntPtr;
	using CUarray = IntPtr;
	using CUstream = IntPtr;
	using size_t = Int64;

	public partial class Defines {
		/// <summary>
		/// CUDA IPC handle size 
		/// </summary>
		public const int CU_IPC_HANDLE_SIZE = 64;

		/// <summary>
		/// Legacy stream handle
		/// 
		/// Stream handle that can be passed as a CUstream to use an implicit stream
		/// with legacy synchronization behavior.
		/// 
		/// See details of the \link_sync_behavior
		/// </summary>
		public readonly CUstream CU_STREAM_LEGACY = new CUstream(0x1);

		/// <summary>
		/// Per-thread stream handle
		/// 
		/// Stream handle that can be passed as a CUstream to use an implicit stream
		/// with per-thread synchronization behavior.
		/// 
		/// See details of the \link_sync_behavior
		/// </summary>
		public readonly CUstream CU_STREAM_PER_THREAD = new CUstream(0x2);

		/// <summary>
		/// If set, host memory is portable between CUDA contexts.
		/// Flag for ::cuMemHostAlloc()
		/// </summary>
		public const int CU_MEMHOSTALLOC_PORTABLE = 0x01;

		/// <summary>
		/// If set, host memory is mapped into CUDA address space and
		/// ::cuMemHostGetDevicePointer() may be called on the host pointer.
		/// Flag for ::cuMemHostAlloc()
		/// </summary>
		public const int CU_MEMHOSTALLOC_DEVICEMAP = 0x02;

		/// <summary>
		/// If set, host memory is portable between CUDA contexts.
		/// Flag for ::cuMemHostRegister()
		/// </summary>
		public const int CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;

		/// <summary>
		/// If set, host memory is portable between CUDA contexts.
		/// Flag for ::cuMemHostRegister()
		/// </summary>
		public const int CU_MEMHOSTREGISTER_PORTABLE = 0x01;

		/// <summary>
		/// If set, host memory is mapped into CUDA address space and
		/// ::cuMemHostGetDevicePointer() may be called on the host pointer.
		/// Flag for ::cuMemHostRegister()
		/// </summary>
		public const int CU_MEMHOSTREGISTER_DEVICEMAP = 0x02;

		/// <summary>
		/// If set, the passed memory pointer is treated as pointing to some
		/// memory-mapped I/O space, e.g.belonging to a third-party PCIe device.
		///
		/// On Windows the flag is a no-op.
		///
		/// On Linux that memory is marked as non cache-coherent for the GPU and
		/// is expected to be physically contiguous.It may return
		/// CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
		/// CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions.
		/// On all other platforms, it is not supported and CUDA_ERROR_NOT_SUPPORTED
		/// is returned.
		/// Flag for ::cuMemHostRegister()
		/// </summary>
		public const int CU_MEMHOSTREGISTER_IOMEMORY = 0x04;

		/// <summary>
		/// If set, the CUDA array is a collection of layers, where each layer is either a 1D
		/// or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number 
		/// of layers, not the depth of a 3D array.
		/// </summary>
		public const int CUDA_ARRAY3D_LAYERED = 0x01;

		/// <summary>
		/// Deprecated, use CUDA_ARRAY3D_LAYERED
		/// </summary>
		[Obsolete("use CUDA_ARRAY3D_LAYERED")]
		public const int CUDA_ARRAY3D_2DARRAY = 0x01;

		/// <summary>
		/// This flag must be set in order to bind a surface reference
		/// to the CUDA array
		/// </summary>
		public const int CUDA_ARRAY3D_SURFACE_LDST = 0x02;

		/// <summary>
		/// If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube.The
		/// width of such a CUDA array must be equal to its height, and Depth must be six.
		/// If ::CUDA_ARRAY3D_LAYERED flag is also set, then the CUDA array is a collection of cubemaps
		/// and Depth must be a multiple of six.
		/// </summary>
		public const int CUDA_ARRAY3D_CUBEMAP = 0x04;

		/// <summary>
		/// This flag must be set in order to perform texture gather operations
		/// on a CUDA array.
		/// </summary>
		public const int CUDA_ARRAY3D_TEXTURE_GATHER = 0x08;

		/// <summary>
		/// This flag if set indicates that the CUDA
		/// array is a DEPTH_TEXTURE.
		/// </summary>
		public const int CUDA_ARRAY3D_DEPTH_TEXTURE = 0x10;

		/// <summary>
		/// Override the texref format with a format inferred from the array.
		/// Flag for ::cuTexRefSetArray()
		/// </summary>
		public const int CU_TRSA_OVERRIDE_FORMAT = 0x01;

		/// <summary>
		/// Read the texture as integers rather than promoting the values to floats
		/// in the range[0, 1].
		/// Flag for ::cuTexRefSetFlags()
		/// </summary>
		public const int CU_TRSF_READ_AS_INTEGER = 0x01;

		/// <summary>
		/// Use normalized texture coordinates in the range[0, 1) instead of[0, dim).
		/// Flag for ::cuTexRefSetFlags()
		/// </summary>
		public const int CU_TRSF_NORMALIZED_COORDINATES = 0x02;

		/// <summary>
		/// Perform sRGB->linear conversion during texture read.
		/// Flag for ::cuTexRefSetFlags()
		/// </summary>
		public const int CU_TRSF_SRGB = 0x10;

		/// <summary>
		/// End of array terminator for the \p extra parameter to
		/// ::cuLaunchKernel
		/// </summary>
		public readonly IntPtr CU_LAUNCH_PARAM_END = IntPtr.Zero;

		/// <summary>
		/// Indicator that the next value in the \p extra parameter to
		/// ::cuLaunchKernel will be a pointer to a buffer containing all kernel
		/// parameters used for launching kernel \p f.This buffer needs to
		/// honor all alignment/padding requirements of the individual parameters.
		///  If::CU_LAUNCH_PARAM_BUFFER_SIZE is not also specified in the
		/// \p extra array, then::CU_LAUNCH_PARAM_BUFFER_POINTER will have no
		/// effect.
		/// </summary>
		public readonly IntPtr CU_LAUNCH_PARAM_BUFFER_POINTER = new IntPtr(0x01);

		/// <summary>
		/// Indicator that the next value in the \p extra parameter to
		/// ::cuLaunchKernel will be a pointer to a size_t which contains the
		/// size of the buffer specified with::CU_LAUNCH_PARAM_BUFFER_POINTER.
		/// It is required that ::CU_LAUNCH_PARAM_BUFFER_POINTER also be specified
		/// in the \p extra array if the value associated with
		/// ::CU_LAUNCH_PARAM_BUFFER_SIZE is not zero.
		/// </summary>
		public readonly IntPtr CU_LAUNCH_PARAM_BUFFER_SIZE = new IntPtr(0x02);

		/// <summary>
		/// For texture references loaded into the module, use default texunit from
		/// texture reference.
		/// </summary>
		public int CU_PARAM_TR_DEFAULT = -1;

		/// <summary>
		/// Device that represents the CPU
		/// </summary>
		public const CUdevice CU_DEVICE_CPU = -1;

		/// <summary>
		/// Device that represents an invalid device
		/// </summary>
		public const CUdevice CU_DEVICE_INVALID = -2;
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
		/// <summary>
		/// Automatically enable peer access between remote devices as needed
		/// </summary>
		CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1 
	}

	/// <summary>
	/// CUDA Mem Attach Flags
	/// </summary>
	public enum CUmemAttach_flags {
		/// <summary>
		/// Memory can be accessed by any stream on any device
		/// </summary>
		CU_MEM_ATTACH_GLOBAL = 0x1,

		/// <summary>
		/// Memory cannot be accessed by any stream on any device
		/// </summary>
		CU_MEM_ATTACH_HOST = 0x2,

		/// <summary>
		/// Memory can only be accessed by a single stream on the associated device
		/// </summary>
		CU_MEM_ATTACH_SINGLE = 0x4
	}

	/// <summary>
	/// Context creation flags
	/// </summary>
	public enum CUctx_flags {
		/// <summary>
		/// Automatic scheduling
		/// </summary>
		CU_CTX_SCHED_AUTO = 0x00, 

		/// <summary>
		/// Set spin as default scheduling
		/// </summary>
		CU_CTX_SCHED_SPIN = 0x01, 

		/// <summary>
		/// Set yield as default scheduling
		/// </summary>
		CU_CTX_SCHED_YIELD = 0x02, 

		/// <summary>
		/// Set blocking synchronization as default scheduling
		/// </summary>
		CU_CTX_SCHED_BLOCKING_SYNC = 0x04,

		/// <summary>
		/// Set blocking synchronization as default scheduling
		/// \deprecated This flag was deprecated as of CUDA 4.0
		/// and was replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
		/// </summary>
		[Obsolete("This flag was deprecated as of CUDA 4.0")]
		CU_CTX_BLOCKING_SYNC = 0x04,

		CU_CTX_SCHED_MASK = 0x07,

		/// <summary>
		/// Support mapped pinned allocations
		/// </summary>
		CU_CTX_MAP_HOST = 0x08, 

		/// <summary>
		/// Keep local memory allocation after launch
		/// </summary>
		CU_CTX_LMEM_RESIZE_TO_MAX = 0x10, 

		CU_CTX_FLAGS_MASK = 0x1f
	}

	/// <summary>
	/// Stream creation flags
	/// </summary>
	public enum CUstream_flags {
		/// <summary>
		/// Default stream flag
		/// </summary>
		CU_STREAM_DEFAULT = 0x0, 
		/// <summary>
		/// Stream does not synchronize with stream 0 (the NULL stream)
		/// </summary>
		CU_STREAM_NON_BLOCKING = 0x1  
	}

	/// <summary>
	/// Event creation flags
	/// </summary>
	public enum CUevent_flags {
		/// <summary>
		/// Default event flag
		/// </summary>
		CU_EVENT_DEFAULT = 0x0, 
		/// <summary>
		/// Event uses blocking synchronization
		/// </summary>
		CU_EVENT_BLOCKING_SYNC = 0x1, 
		/// <summary>
		/// Event will not record timing data
		/// </summary>
		CU_EVENT_DISABLE_TIMING = 0x2, 
		/// <summary>
		/// Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set
		/// </summary>
		CU_EVENT_INTERPROCESS = 0x4  
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
		/// <summary>
		/// Represents a ::cuStreamWaitValue32 operation
		/// </summary>
		CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,     
		/// <summary>
		/// Represents a ::cuStreamWriteValue32 operation
		/// </summary>
		CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,     
		/// <summary>
		/// This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a standalone operation.
		/// </summary>
		CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3 
	}

	/// <summary>
	/// Occupancy calculator flag
	/// </summary>
	public enum CUoccupancy_flags {
		/// <summary>
		/// Default behavior
		/// </summary>
		CU_OCCUPANCY_DEFAULT = 0x0, 
		/// <summary>
		/// Assume global caching is enabled and cannot be automatically turned off
		/// </summary>
		CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1  
	}

	/// <summary>
	/// Array formats
	/// </summary>
	public enum CUarray_format {
		/// <summary>
		/// Unsigned 8-bit integers
		/// </summary>
		CU_AD_FORMAT_UNSIGNED_INT8 = 0x01, 
		/// <summary>
		/// Unsigned 16-bit integers
		/// </summary>
		CU_AD_FORMAT_UNSIGNED_INT16 = 0x02, 
		/// <summary>
		/// Unsigned 32-bit integers
		/// </summary>
		CU_AD_FORMAT_UNSIGNED_INT32 = 0x03, 
		/// <summary>
		/// Signed 8-bit integers
		/// </summary>
		CU_AD_FORMAT_SIGNED_INT8 = 0x08, 
		/// <summary>
		/// Signed 16-bit integers
		/// </summary>
		CU_AD_FORMAT_SIGNED_INT16 = 0x09, 
		/// <summary>
		/// Signed 32-bit integers
		/// </summary>
		CU_AD_FORMAT_SIGNED_INT32 = 0x0a, 
		/// <summary>
		/// 16-bit floating point
		/// </summary>
		CU_AD_FORMAT_HALF = 0x10, 
		/// <summary>
		/// 32-bit floating point
		/// </summary>
		CU_AD_FORMAT_FLOAT = 0x20  
	}

	/// <summary>
	/// Texture reference addressing modes
	/// </summary>
	public enum CUaddress_mode {
		/// <summary>
		/// Wrapping address mode
		/// </summary>
		CU_TR_ADDRESS_MODE_WRAP = 0, 
		/// <summary>
		/// Clamp to edge address mode
		/// </summary>
		CU_TR_ADDRESS_MODE_CLAMP = 1, 
		/// <summary>
		/// Mirror address mode
		/// </summary>
		CU_TR_ADDRESS_MODE_MIRROR = 2, 
		/// <summary>
		/// Border address mode
		/// </summary>
		CU_TR_ADDRESS_MODE_BORDER = 3  
	}

	/// <summary>
	/// Texture reference filtering modes
	/// </summary>
	public enum CUfilter_mode {
		/// <summary>
		/// Point filter mode
		/// </summary>
		CU_TR_FILTER_MODE_POINT = 0, 
		/// <summary>
		/// Linear filter mode
		/// </summary>
		CU_TR_FILTER_MODE_LINEAR = 1  
	}

	/// <summary>
	/// Device properties
	/// </summary>
	public enum CUdevice_attribute {
		/// <summary>
		/// Maximum number of threads per block
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,              
		/// <summary>
		/// Maximum block dimension X
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,                    
		/// <summary>
		/// Maximum block dimension Y
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,                    
		/// <summary>
		/// Maximum block dimension Z
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,                    
		/// <summary>
		/// Maximum grid dimension X
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,                     
		/// <summary>
		/// Maximum grid dimension Y
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,                     
		/// <summary>
		/// Maximum grid dimension Z
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,                     
		/// <summary>
		/// Maximum shared memory available per block in bytes
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,        
		/// <summary>
		/// Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
		/// </summary>
		[Obsolete("use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK")]
		CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,            
		/// <summary>
		/// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
		/// </summary>
		CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,              
		/// <summary>
		/// Warp size in threads
		/// </summary>
		CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                         
		/// <summary>
		/// Maximum pitch in bytes allowed by memory copies
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                         
		/// <summary>
		/// Maximum number of 32-bit registers available per block
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,           
		/// <summary>
		/// Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
		/// </summary>
		[Obsolete("use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK")]
		CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,               
		/// <summary>
		/// Typical clock frequency in kilohertz
		/// </summary>
		CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        
		/// <summary>
		/// Alignment requirement for textures
		/// </summary>
		CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                 
		/// <summary>
		/// Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.
		/// </summary>
		[Obsolete("Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT")]
		CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       
		/// <summary>
		/// Number of multiprocessors on device
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,              
		/// <summary>
		/// Specifies whether there is a run time limit on kernels
		/// </summary>
		CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,               
		/// <summary>
		/// Device is integrated with host memory
		/// </summary>
		CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,                        
		/// <summary>
		/// Device can map host memory into CUDA address space
		/// </summary>
		CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,               
		/// <summary>
		/// Compute mode (See ::CUcomputemode for details)
		/// </summary>
		CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                      
		/// <summary>
		/// Maximum 1D texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,           
		/// <summary>
		/// Maximum 2D texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,           
		/// <summary>
		/// Maximum 2D texture height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,          
		/// <summary>
		/// Maximum 3D texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,           
		/// <summary>
		/// Maximum 3D texture height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,          
		/// <summary>
		/// Maximum 3D texture depth
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,           
		/// <summary>
		/// Maximum 2D layered texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,   
		/// <summary>
		/// Maximum 2D layered texture height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,  
		/// <summary>
		/// Maximum layers in a 2D layered texture
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,  
		/// <summary>
		/// Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
		/// </summary>
		[Obsolete("use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH")]
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,     
		/// <summary>
		/// Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
		/// </summary>
		[Obsolete("use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT")]
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,    
		/// <summary>
		/// Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
		/// </summary>
		[Obsolete("use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS")]
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, 
		/// <summary>
		/// Alignment requirement for surfaces
		/// </summary>
		CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                 
		/// <summary>
		/// Device can possibly execute multiple kernels concurrently
		/// </summary>
		CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,                
		/// <summary>
		/// Device has ECC support enabled
		/// </summary>
		CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       
		/// <summary>
		/// PCI bus ID of the device
		/// </summary>
		CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        
		/// <summary>
		/// PCI device ID of the device
		/// </summary>
		CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     
		/// <summary>
		/// Device is using TCC driver model
		/// </summary>
		CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                        
		/// <summary>
		/// Peak memory clock frequency in kilohertz
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 
		/// <summary>
		/// Global memory bus width in bits
		/// </summary>
		CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           
		/// <summary>
		/// Size of L2 cache in bytes
		/// </summary>
		CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                     
		/// <summary>
		/// Maximum resident threads per multiprocessor
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    
		/// <summary>
		/// Number of asynchronous engines
		/// </summary>
		CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                
		/// <summary>
		/// Device shares a unified address space with the host
		/// </summary>
		CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                
		/// <summary>
		/// Maximum 1D layered texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,   
		/// <summary>
		/// Maximum layers in a 1D layered texture
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,  
		/// <summary>
		/// Deprecated, do not use.
		/// </summary>
		[Obsolete("Deprecated, do not use")]
		CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                  
		/// <summary>
		/// Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,    
		/// <summary>
		/// Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,   
		/// <summary>
		/// Alternate maximum 3D texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47, 
		/// <summary>
		/// Alternate maximum 3D texture height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
		/// <summary>
		/// Alternate maximum 3D texture depth
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49, 
		/// <summary>
		/// PCI domain ID of the device
		/// </summary>
		CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                     
		/// <summary>
		/// Pitch alignment requirement for textures
		/// </summary>
		CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,           
		/// <summary>
		/// Maximum cubemap texture width/height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,      
		/// <summary>
		/// Maximum cubemap layered texture width/height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,  
		/// <summary>
		/// Maximum layers in a cubemap layered texture
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54, 
		/// <summary>
		/// Maximum 1D surface width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,           
		/// <summary>
		/// Maximum 2D surface width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,           
		/// <summary>
		/// Maximum 2D surface height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,          
		/// <summary>
		/// Maximum 3D surface width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,           
		/// <summary>
		/// Maximum 3D surface height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,          
		/// <summary>
		/// Maximum 3D surface depth
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,           
		/// <summary>
		/// Maximum 1D layered surface width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,   
		/// <summary>
		/// Maximum layers in a 1D layered surface
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,  
		/// <summary>
		/// Maximum 2D layered surface width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,   
		/// <summary>
		/// Maximum 2D layered surface height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,  
		/// <summary>
		/// Maximum layers in a 2D layered surface
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,  
		/// <summary>
		/// Maximum cubemap surface width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,      
		/// <summary>
		/// Maximum cubemap layered surface width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,  
		/// <summary>
		/// Maximum layers in a cubemap layered surface
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68, 
		/// <summary>
		/// Maximum 1D linear texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,    
		/// <summary>
		/// Maximum 2D linear texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,    
		/// <summary>
		/// Maximum 2D linear texture height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,   
		/// <summary>
		/// Maximum 2D linear texture pitch in bytes
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,    
		/// <summary>
		/// Maximum mipmapped 2D texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73, 
		/// <summary>
		/// Maximum mipmapped 2D texture height
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
		/// <summary>
		/// Major compute capability version number
		/// </summary>
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,          
		/// <summary>
		/// Minor compute capability version number
		/// </summary>
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,          
		/// <summary>
		/// Maximum mipmapped 1D texture width
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77, 
		/// <summary>
		/// Device supports stream priorities
		/// </summary>
		CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,       
		/// <summary>
		/// Device supports caching globals in L1
		/// </summary>
		CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,         
		/// <summary>
		/// Device supports caching locals in L1
		/// </summary>
		CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,          
		/// <summary>
		/// Maximum shared memory available per multiprocessor in bytes
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,  
		/// <summary>
		/// Maximum number of 32-bit registers available per multiprocessor
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,  
		/// <summary>
		/// Device can allocate managed memory on this system
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,                    
		/// <summary>
		/// Device is on a multi-GPU board
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,                    
		/// <summary>
		/// Unique id for a group of devices on the same multi-GPU board
		/// </summary>
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
		/// <summary>
		/// Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
		/// </summary>
		CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,       
		/// <summary>
		/// Ratio of single precision performance (in floating-point operations per second) to double precision performance
		/// </summary>
		CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,  
		/// <summary>
		/// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
		/// </summary>
		CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,            
		/// <summary>
		/// Device can coherently access managed memory concurrently with the CPU
		/// </summary>
		CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,         
		/// <summary>
		/// Device supports compute preemption.
		/// </summary>
		CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,      
		/// <summary>
		/// Device can access host registered memory at the same virtual address as the CPU
		/// </summary>
		CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91, 
		CU_DEVICE_ATTRIBUTE_MAX
	}

	/// <summary>
	/// Legacy device properties
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUdevprop {
		/// <summary>
		/// Maximum number of threads per block
		/// </summary>
		int maxThreadsPerBlock;     
		/// <summary>
		/// Maximum size of each dimension of a block
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		int[] maxThreadsDim;       
		/// <summary>
		/// Maximum size of each dimension of a grid
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
		int[] maxGridSize;         
		/// <summary>
		/// Shared memory available per block in bytes
		/// </summary>
		int sharedMemPerBlock;      
		/// <summary>
		/// Constant memory available on device in bytes
		/// </summary>
		int totalConstantMemory;    
		/// <summary>
		/// Warp size in threads
		/// </summary>
		int SIMDWidth;              
		/// <summary>
		/// Maximum pitch in bytes allowed by memory copies
		/// </summary>
		int memPitch;               
		/// <summary>
		/// 32-bit registers available per block
		/// </summary>
		int regsPerBlock;           
		/// <summary>
		/// Clock frequency in kilohertz
		/// </summary>
		int clockRate;              
		/// <summary>
		/// Alignment requirement for textures
		/// </summary>
		
		int textureAlign;           
	}

	/// <summary>
	/// Pointer information
	/// </summary>
	public enum CUpointer_attribute {
		/// <summary>
		/// The ::CUcontext on which a pointer was allocated or registered
		/// </summary>
		CU_POINTER_ATTRIBUTE_CONTEXT = 1,        
		/// <summary>
		/// The ::CUmemorytype describing the physical location of a pointer
		/// </summary>
		CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,    
		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device
		/// </summary>
		CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3, 
		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host
		/// </summary>
		CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,   
		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,     
		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,      
		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		CU_POINTER_ATTRIBUTE_IS_MANAGED = 8      
	}

	/// <summary>
	/// Function properties
	/// </summary>
	public enum CUfunction_attribute {
		/// <summary>
		/// The maximum number of threads per block, beyond which a launch of the
		/// function would fail. This number depends on both the function and the
		/// device on which the function is currently loaded.
		/// </summary>
		CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

		/// <summary>
		/// The size in bytes of statically-allocated shared memory required by
		/// this function. This does not include dynamically-allocated shared
		/// memory requested by the user at runtime.
		/// </summary>
		CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

		/// <summary>
		/// The size in bytes of user-allocated constant memory required by this
		/// function.
		/// </summary>
		CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

		/// <summary>
		/// The size in bytes of local memory used by each thread of this function.
		/// </summary>
		CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

		/// <summary>
		/// The number of registers used by each thread of this function.
		/// </summary>
		CU_FUNC_ATTRIBUTE_NUM_REGS = 4,

		/// <summary>
		/// The PTX virtual architecture version for which the function was
		/// compiled. This value is the major PTX version * 10 + the minor PTX
		/// version, so a PTX version 1.3 function would return the value 13.
		/// Note that this may return the undefined value of 0 for cubins
		/// compiled prior to CUDA 3.0.
		/// </summary>
		CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,

		/// <summary>
		/// The binary architecture version for which the function was compiled.
		/// This value is the major binary version * 10 + the minor binary version,
		/// so a binary version 1.3 function would return the value 13. Note that
		/// this will return a value of 10 for legacy cubins that do not have a
		/// properly-encoded binary architecture version.
		/// </summary>
		CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,

		/// <summary>
		/// The attribute to indicate whether the function has been compiled with 
		/// user specified option "-Xptxas --dlcm=ca" set .
		/// </summary>
		CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,

		CU_FUNC_ATTRIBUTE_MAX
	}

	/// <summary>
	/// Function cache configurations
	/// </summary>
	public enum CUfunc_cache {
		/// <summary>
		/// no preference for shared memory or L1 (default)
		/// </summary>
		CU_FUNC_CACHE_PREFER_NONE = 0x00, 
		/// <summary>
		/// prefer larger shared memory and smaller L1 cache
		/// </summary>
		CU_FUNC_CACHE_PREFER_SHARED = 0x01, 
		/// <summary>
		/// prefer larger L1 cache and smaller shared memory
		/// </summary>
		CU_FUNC_CACHE_PREFER_L1 = 0x02, 
		/// <summary>
		/// prefer equal sized L1 cache and shared memory
		/// </summary>
		CU_FUNC_CACHE_PREFER_EQUAL = 0x03  
	}

	/// <summary>
	/// Shared memory configurations
	/// </summary>
	public enum CUsharedconfig {
		/// <summary>
		/// set default shared memory bank size
		/// </summary>
		CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00, 
		/// <summary>
		/// set shared memory bank width to four bytes
		/// </summary>
		CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01, 
		/// <summary>
		/// set shared memory bank width to eight bytes
		/// </summary>
		CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02  
	}

	/// <summary>
	/// Memory types
	/// </summary>
	public enum CUmemorytype {
		/// <summary>
		/// Host memory
		/// </summary>
		CU_MEMORYTYPE_HOST = 0x01,    
		/// <summary>
		/// Device memory
		/// </summary>
		CU_MEMORYTYPE_DEVICE = 0x02,    
		/// <summary>
		/// Array memory
		/// </summary>
		CU_MEMORYTYPE_ARRAY = 0x03,    
		/// <summary>
		/// Unified device or host memory
		/// </summary>
		CU_MEMORYTYPE_UNIFIED = 0x04     
	}

	/// <summary>
	/// Compute Modes
	/// </summary>
	public enum CUcomputemode {
		/// <summary>
		/// Default compute mode (Multiple contexts allowed per device)
		/// </summary>
		CU_COMPUTEMODE_DEFAULT = 0, 
		/// <summary>
		/// Compute-prohibited mode (No contexts can be created on this device at this time)
		/// </summary>
		CU_COMPUTEMODE_EXCLUSIVE = 1,
		CU_COMPUTEMODE_PROHIBITED = 2, 
		/// <summary>
		/// Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time)
		/// </summary>
		CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3  
	}

	/// <summary>
	/// Memory advise values
	/// </summary>
	public enum CUmem_advise {
		/// <summary>
		/// Data will mostly be read and only occassionally be written to
		/// </summary>
		CU_MEM_ADVISE_SET_READ_MOSTLY = 1, 
		/// <summary>
		/// Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY
		/// </summary>
		CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2, 
		/// <summary>
		/// Set the preferred location for the data as the specified device
		/// </summary>
		CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3, 
		/// <summary>
		/// Clear the preferred location for the data
		/// </summary>
		CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4, 
		/// <summary>
		/// Data will be accessed by the specified device, so prevent page faults as much as possible
		/// </summary>
		CU_MEM_ADVISE_SET_ACCESSED_BY = 5, 
		/// <summary>
		/// Let the Unified Memory subsystem decide on the page faulting policy for the specified device
		/// </summary>
		CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6  
	}

	/// <summary>
	/// 
	/// </summary>
	public enum CUmem_range_attribute {
		/// <summary>
		/// Whether the range will mostly be read and only occassionally be written to
		/// </summary>
		CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1, 
		/// <summary>
		/// The preferred location of the range
		/// </summary>
		CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2, 
		/// <summary>
		/// Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device
		/// </summary>
		CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3, 
		/// <summary>
		/// The last location to which the range was prefetched
		/// </summary>
		CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4  
	}

	/// <summary>
	/// Online compiler and linker options
	/// </summary>
	public enum CUjit_option {
		/// <summary>
		/// Max number of registers that a thread may use.\n
		/// Option type: unsigned int\n
		/// Applies to: compiler only
		/// </summary>
		CU_JIT_MAX_REGISTERS = 0,

		/// <summary>
		/// IN: Specifies minimum number of threads per block to target compilation
		/// for\n
		/// OUT: Returns the number of threads the compiler actually targeted.
		/// This restricts the resource utilization fo the compiler (e.g. max
		/// registers) such that a block with the given number of threads should be
		/// able to launch based on register limitations. Note, this option does not
		/// currently take into account any other resource limitations, such as
		/// shared memory utilization.\n
		/// Cannot be combined with ::CU_JIT_TARGET.\n
		/// Option type: unsigned int\n
		/// Applies to: compiler only
		/// </summary>
		CU_JIT_THREADS_PER_BLOCK,

		/// <summary>
		/// Overwrites the option value with the total wall clock time, in
		/// milliseconds, spent in the compiler and linker\n
		/// Option type: float\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_WALL_TIME,

		/// <summary>
		/// Pointer to a buffer in which to print any log messages
		/// that are informational in nature (the buffer size is specified via
		/// option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
		/// Option type: char *\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_INFO_LOG_BUFFER,

		/// <summary>
		/// IN: Log buffer size in bytes.  Log messages will be capped at this size
		/// (including null terminator)\n
		/// OUT: Amount of log buffer filled with messages\n
		/// Option type: unsigned int\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

		/// <summary>
		/// Pointer to a buffer in which to print any log messages that
		/// reflect errors (the buffer size is specified via option
		/// ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
		/// Option type: char *\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_ERROR_LOG_BUFFER,

		/// <summary>
		/// IN: Log buffer size in bytes.  Log messages will be capped at this size
		/// (including null terminator)\n
		/// OUT: Amount of log buffer filled with messages\n
		/// Option type: unsigned int\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

		/// <summary>
		/// Level of optimizations to apply to generated code (0 - 4), with 4
		/// being the default and highest level of optimizations.\n
		/// Option type: unsigned int\n
		/// Applies to: compiler only
		/// </summary>
		CU_JIT_OPTIMIZATION_LEVEL,

		/// <summary>
		/// No option value required. Determines the target based on the current
		/// attached context (default)\n
		/// Option type: No option value needed\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_TARGET_FROM_CUCONTEXT,

		/// <summary>
		/// Target is chosen based on supplied ::CUjit_target.  Cannot be
		/// combined with ::CU_JIT_THREADS_PER_BLOCK.\n
		/// Option type: unsigned int for enumerated type ::CUjit_target\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_TARGET,

		/// <summary>
		/// Specifies choice of fallback strategy if matching cubin is not found.
		/// Choice is based on supplied ::CUjit_fallback.  This option cannot be
		/// used with cuLink* APIs as the linker requires exact matches.\n
		/// Option type: unsigned int for enumerated type ::CUjit_fallback\n
		/// Applies to: compiler only
		/// </summary>
		CU_JIT_FALLBACK_STRATEGY,

		/// <summary>
		/// Specifies whether to create debug information in output (-g)
		/// (0: false, default)\n
		/// Option type: int\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_GENERATE_DEBUG_INFO,

		/// <summary>
		/// Generate verbose log messages (0: false, default)\n
		/// Option type: int\n
		/// Applies to: compiler and linker
		/// </summary>
		CU_JIT_LOG_VERBOSE,

		/// <summary>
		/// Generate line number information (-lineinfo) (0: false, default)\n
		/// Option type: int\n
		/// Applies to: compiler only
		/// </summary>
		CU_JIT_GENERATE_LINE_INFO,

		/// <summary>
		/// Specifies whether to enable caching explicitly (-dlcm) \n
		/// Choice is based on supplied ::CUjit_cacheMode_enum.\n
		/// Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum\n
		/// Applies to: compiler only
		/// </summary>
		CU_JIT_CACHE_MODE,

		/// <summary>
		/// The below jit options are used for internal purposes only, in this version of CUDA
		/// </summary>
		CU_JIT_NEW_SM3X_OPT,
		CU_JIT_FAST_COMPILE,

		CU_JIT_NUM_OPTIONS
	}

	/// <summary>
	/// Online compilation targets
	/// </summary>
	public enum CUjit_target {
		/// <summary>
		/// Compute device class 1.0
		/// </summary>
		CU_TARGET_COMPUTE_10 = 10,       
		/// <summary>
		/// Compute device class 1.1
		/// </summary>
		CU_TARGET_COMPUTE_11 = 11,       
		/// <summary>
		/// Compute device class 1.2
		/// </summary>
		CU_TARGET_COMPUTE_12 = 12,       
		/// <summary>
		/// Compute device class 1.3
		/// </summary>
		CU_TARGET_COMPUTE_13 = 13,       
		/// <summary>
		/// Compute device class 2.0
		/// </summary>
		CU_TARGET_COMPUTE_20 = 20,       
		/// <summary>
		/// Compute device class 2.1
		/// </summary>
		CU_TARGET_COMPUTE_21 = 21,       
		/// <summary>
		/// Compute device class 3.0
		/// </summary>
		CU_TARGET_COMPUTE_30 = 30,       
		/// <summary>
		/// Compute device class 3.2
		/// </summary>
		CU_TARGET_COMPUTE_32 = 32,       
		/// <summary>
		/// Compute device class 3.5
		/// </summary>
		CU_TARGET_COMPUTE_35 = 35,       
		/// <summary>
		/// Compute device class 3.7
		/// </summary>
		CU_TARGET_COMPUTE_37 = 37,       
		/// <summary>
		/// Compute device class 5.0
		/// </summary>
		CU_TARGET_COMPUTE_50 = 50,       
		/// <summary>
		/// Compute device class 5.2
		/// </summary>
		CU_TARGET_COMPUTE_52 = 52,       
		/// <summary>
		/// Compute device class 5.3
		/// </summary>
		CU_TARGET_COMPUTE_53 = 53,       
		/// <summary>
		/// Compute device class 6.0. This must be removed for CUDA 7.0 toolkit. See bug 1518217.
		/// </summary>
		CU_TARGET_COMPUTE_60 = 60,
		/// <summary>
		/// Compute device class 6.1. This must be removed for CUDA 7.0 toolkit.
		/// </summary>
		CU_TARGET_COMPUTE_61 = 61,
		/// <summary>
		/// Compute device class 6.2. This must be removed for CUDA 7.0 toolkit.
		/// </summary>
		CU_TARGET_COMPUTE_62 = 62        
	}

	/// <summary>
	/// Cubin matching fallback strategies
	/// </summary>
	public enum CUjit_fallback {
		/// <summary>
		/// Prefer to compile ptx if exact binary match not found
		/// </summary>
		CU_PREFER_PTX = 0,  
		/// <summary>
		/// Prefer to fall back to compatible binary code if exact match not found
		/// </summary>
		CU_PREFER_BINARY    
	}

	/// <summary>
	/// Caching modes for dlcm 
	/// </summary>
	public enum CUjit_cacheMode {
		/// <summary>
		/// Compile with no -dlcm flag specified
		/// </summary>
		CU_JIT_CACHE_OPTION_NONE = 0, 
		/// <summary>
		/// Compile with L1 cache disabled
		/// </summary>
		CU_JIT_CACHE_OPTION_CG,       
		/// <summary>
		/// Compile with L1 cache enabled
		/// </summary>
		CU_JIT_CACHE_OPTION_CA        
	}

	/// <summary>
	/// Device code formats
	/// </summary>
	public enum CUjitInputType {
		/// <summary>
		/// Compiled device-class-specific device code\n
		/// Applicable options: none
		/// </summary>
		CU_JIT_INPUT_CUBIN = 0,

		/// <summary>
		/// PTX source code\n
		/// Applicable options: PTX compiler options
		/// </summary>
		CU_JIT_INPUT_PTX,

		/// <summary>
		/// Bundle of multiple cubins and/or PTX of some device code\n
		/// Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
		/// </summary>
		CU_JIT_INPUT_FATBINARY,

		/// <summary>
		/// Host object with embedded device code\n
		/// Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
		/// </summary>
		CU_JIT_INPUT_OBJECT,

		/// <summary>
		/// Archive of host objects with embedded device code\n
		/// Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
		/// </summary>
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
		/// <summary>
		/// Positive X face of cubemap
		/// </summary>
		CU_CUBEMAP_FACE_POSITIVE_X = 0x00, 
		/// <summary>
		/// Negative X face of cubemap
		/// </summary>
		CU_CUBEMAP_FACE_NEGATIVE_X = 0x01, 
		/// <summary>
		/// Positive Y face of cubemap
		/// </summary>
		CU_CUBEMAP_FACE_POSITIVE_Y = 0x02, 
		/// <summary>
		/// Negative Y face of cubemap
		/// </summary>
		CU_CUBEMAP_FACE_NEGATIVE_Y = 0x03, 
		/// <summary>
		/// Positive Z face of cubemap
		/// </summary>
		CU_CUBEMAP_FACE_POSITIVE_Z = 0x04, 
		/// <summary>
		/// Negative Z face of cubemap
		/// </summary>
		CU_CUBEMAP_FACE_NEGATIVE_Z = 0x05  
	}

	/// <summary>
	/// Limits
	/// </summary>
	public enum CUlimit {
		/// <summary>
		/// GPU thread stack size
		/// </summary>
		CU_LIMIT_STACK_SIZE = 0x00, 
		/// <summary>
		/// GPU printf FIFO size
		/// </summary>
		CU_LIMIT_PRINTF_FIFO_SIZE = 0x01, 
		/// <summary>
		/// GPU malloc heap size
		/// </summary>
		CU_LIMIT_MALLOC_HEAP_SIZE = 0x02, 
		/// <summary>
		/// GPU device runtime launch synchronize depth
		/// </summary>
		CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03, 
		/// <summary>
		/// GPU device runtime pending launch count
		/// </summary>
		CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04, 
		CU_LIMIT_MAX
	}

	/// <summary>
	/// Resource types
	/// </summary>
	public enum CUresourcetype {
		/// <summary>
		/// Array resoure
		/// </summary>
		CU_RESOURCE_TYPE_ARRAY = 0x00, 
		/// <summary>
		/// Mipmapped array resource
		/// </summary>
		CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, 
		/// <summary>
		/// Linear resource
		/// </summary>
		CU_RESOURCE_TYPE_LINEAR = 0x02, 
		/// <summary>
		/// Pitch 2D resource
		/// </summary>
		CU_RESOURCE_TYPE_PITCH2D = 0x03  
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

		/// <summary>
		/// This indicates that one or more of the parameters passed to the API call
		/// is not within an acceptable range of values.
		/// </summary>
		CUDA_ERROR_INVALID_VALUE = 1,

		/// <summary>
		/// The API call failed because it was unable to allocate enough memory to
		/// perform the requested operation.
		/// </summary>
		CUDA_ERROR_OUT_OF_MEMORY = 2,

		/// <summary>
		/// This indicates that the CUDA driver has not been initialized with
		/// ::cuInit() or that initialization has failed.
		/// </summary>
		CUDA_ERROR_NOT_INITIALIZED = 3,

		/// <summary>
		/// This indicates that the CUDA driver is in the process of shutting down.
		/// </summary>
		CUDA_ERROR_DEINITIALIZED = 4,

		/// <summary>
		/// This indicates profiler is not initialized for this run. This can
		/// happen when the application is running with external profiling tools
		/// like visual profiler.
		/// </summary>
		CUDA_ERROR_PROFILER_DISABLED = 5,

		/// <summary>
		/// \deprecated
		/// This error return is deprecated as of CUDA 5.0. It is no longer an error
		/// to attempt to enable/disable the profiling via ::cuProfilerStart or
		/// ::cuProfilerStop without initialization.
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 5.0")]
		CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,

		/// <summary>
		/// \deprecated
		/// This error return is deprecated as of CUDA 5.0. It is no longer an error
		/// to call cuProfilerStart() when profiling is already enabled.
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 5.0")]
		CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,

		/// <summary>
		/// \deprecated
		/// This error return is deprecated as of CUDA 5.0. It is no longer an error
		/// to call cuProfilerStop() when profiling is already disabled.
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 5.0")]
		CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,

		/// <summary>
		/// This indicates that no CUDA-capable devices were detected by the installed
		/// CUDA driver.
		/// </summary>
		CUDA_ERROR_NO_DEVICE = 100,

		/// <summary>
		/// This indicates that the device ordinal supplied by the user does not
		/// correspond to a valid CUDA device.
		/// </summary>
		CUDA_ERROR_INVALID_DEVICE = 101,


		/// <summary>
		/// This indicates that the device kernel image is invalid. This can also
		/// indicate an invalid CUDA module.
		/// </summary>
		CUDA_ERROR_INVALID_IMAGE = 200,

		/// <summary>
		/// This most frequently indicates that there is no context bound to the
		/// current thread. This can also be returned if the context passed to an
		/// API call is not a valid handle (such as a context that has had
		/// ::cuCtxDestroy() invoked on it). This can also be returned if a user
		/// mixes different API versions (i.e. 3010 context with 3020 API calls).
		/// See ::cuCtxGetApiVersion() for more details.
		/// </summary>
		CUDA_ERROR_INVALID_CONTEXT = 201,

		/// <summary>
		/// This indicated that the context being supplied as a parameter to the
		/// API call was already the active context.
		/// \deprecated
		/// This error return is deprecated as of CUDA 3.2. It is no longer an
		/// error to attempt to push the active context via ::cuCtxPushCurrent().
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 3.2")]
		CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,

		/// <summary>
		/// This indicates that a map or register operation has failed.
		/// </summary>
		CUDA_ERROR_MAP_FAILED = 205,

		/// <summary>
		/// This indicates that an unmap or unregister operation has failed.
		/// </summary>
		CUDA_ERROR_UNMAP_FAILED = 206,

		/// <summary>
		/// This indicates that the specified array is currently mapped and thus
		/// cannot be destroyed.
		/// </summary>
		CUDA_ERROR_ARRAY_IS_MAPPED = 207,

		/// <summary>
		/// This indicates that the resource is already mapped.
		/// </summary>
		CUDA_ERROR_ALREADY_MAPPED = 208,

		/// <summary>
		/// This indicates that there is no kernel image available that is suitable
		/// for the device. This can occur when a user specifies code generation
		/// options for a particular CUDA source file that do not include the
		/// corresponding device configuration.
		/// </summary>
		CUDA_ERROR_NO_BINARY_FOR_GPU = 209,

		/// <summary>
		/// This indicates that a resource has already been acquired.
		/// </summary>
		CUDA_ERROR_ALREADY_ACQUIRED = 210,

		/// <summary>
		/// This indicates that a resource is not mapped.
		/// </summary>
		CUDA_ERROR_NOT_MAPPED = 211,

		/// <summary>
		/// This indicates that a mapped resource is not available for access as an
		/// array.
		/// </summary>
		CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,

		/// <summary>
		/// This indicates that a mapped resource is not available for access as a
		/// pointer.
		/// </summary>
		CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,

		/// <summary>
		/// This indicates that an uncorrectable ECC error was detected during
		/// execution.
		/// </summary>
		CUDA_ERROR_ECC_UNCORRECTABLE = 214,

		/// <summary>
		/// This indicates that the ::CUlimit passed to the API call is not
		/// supported by the active device.
		/// </summary>
		CUDA_ERROR_UNSUPPORTED_LIMIT = 215,

		/// <summary>
		/// This indicates that the ::CUcontext passed to the API call can
		/// only be bound to a single CPU thread at a time but is already 
		/// bound to a CPU thread.
		/// </summary>
		CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,

		/// <summary>
		/// This indicates that peer access is not supported across the given
		/// devices.
		/// </summary>
		CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,

		/// <summary>
		/// This indicates that a PTX JIT compilation failed.
		/// </summary>
		CUDA_ERROR_INVALID_PTX = 218,

		/// <summary>
		/// This indicates an error with OpenGL or DirectX context.
		/// </summary>
		CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,

		/// <summary>
		/// This indicates that an uncorrectable NVLink error was detected during the
		/// execution.
		/// </summary>
		CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,

		/// <summary>
		/// This indicates that the device kernel source is invalid.
		/// </summary>
		CUDA_ERROR_INVALID_SOURCE = 300,

		/// <summary>
		/// This indicates that the file specified was not found.
		/// </summary>
		CUDA_ERROR_FILE_NOT_FOUND = 301,

		/// <summary>
		/// This indicates that a link to a shared object failed to resolve.
		/// </summary>
		CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

		/// <summary>
		/// This indicates that initialization of a shared object failed.
		/// </summary>
		CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,

		/// <summary>
		/// This indicates that an OS call failed.
		/// </summary>
		CUDA_ERROR_OPERATING_SYSTEM = 304,

		/// <summary>
		/// This indicates that a resource handle passed to the API call was not
		/// valid. Resource handles are opaque types like ::CUstream and ::CUevent.
		/// </summary>
		CUDA_ERROR_INVALID_HANDLE = 400,

		/// <summary>
		/// This indicates that a named symbol was not found. Examples of symbols
		/// are global/constant variable names, texture names, and surface names.
		/// </summary>
		CUDA_ERROR_NOT_FOUND = 500,

		/// <summary>
		/// This indicates that asynchronous operations issued previously have not
		/// completed yet. This result is not actually an error, but must be indicated
		/// differently than ::CUDA_SUCCESS (which indicates completion). Calls that
		/// may return this value include ::cuEventQuery() and ::cuStreamQuery().
		/// </summary>
		CUDA_ERROR_NOT_READY = 600,

		/// <summary>
		/// While executing a kernel, the device encountered a
		/// load or store instruction on an invalid memory address.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_ILLEGAL_ADDRESS = 700,

		/// <summary>
		/// This indicates that a launch did not occur because it did not have
		/// appropriate resources. This error usually indicates that the user has
		/// attempted to pass too many arguments to the device kernel, or the
		/// kernel launch specifies too many threads for the kernel's register
		/// count. Passing arguments of the wrong size (i.e. a 64-bit pointer
		/// when a 32-bit int is expected) is equivalent to passing too many
		/// arguments and can also result in this error.
		/// </summary>
		CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,

		/// <summary>
		/// This indicates that the device kernel took too long to execute. This can
		/// only occur if timeouts are enabled - see the device attribute
		/// ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_LAUNCH_TIMEOUT = 702,

		/// <summary>
		/// This error indicates a kernel launch that uses an incompatible texturing
		/// mode.
		/// </summary>
		CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,

		/// <summary>
		/// This error indicates that a call to ::cuCtxEnablePeerAccess() is
		/// trying to re-enable peer access to a context which has already
		/// had peer access to it enabled.
		/// </summary>
		CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

		/// <summary>
		/// This error indicates that ::cuCtxDisablePeerAccess() is 
		/// trying to disable peer access which has not been enabled yet 
		/// via ::cuCtxEnablePeerAccess(). 
		/// </summary>
		CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,

		/// <summary>
		/// This error indicates that the primary context for the specified device
		/// has already been initialized.
		/// </summary>
		CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,

		/// <summary>
		/// This error indicates that the context current to the calling thread
		/// has been destroyed using ::cuCtxDestroy, or is a primary context which
		/// has not yet been initialized.
		/// </summary>
		CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,

		/// <summary>
		/// A device-side assert triggered during kernel execution. The context
		/// cannot be used anymore, and must be destroyed. All existing device 
		/// memory allocations from this context are invalid and must be 
		/// reconstructed if the program is to continue using CUDA.
		/// </summary>
		CUDA_ERROR_ASSERT = 710,

		/// <summary>
		/// This error indicates that the hardware resources required to enable
		/// peer access have been exhausted for one or more of the devices 
		/// passed to ::cuCtxEnablePeerAccess().
		/// </summary>
		CUDA_ERROR_TOO_MANY_PEERS = 711,

		/// <summary>
		/// This error indicates that the memory range passed to ::cuMemHostRegister()
		/// has already been registered.
		/// </summary>
		CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

		/// <summary>
		/// This error indicates that the pointer passed to ::cuMemHostUnregister()
		/// does not correspond to any currently registered memory region.
		/// </summary>
		CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,

		/// <summary>
		/// While executing a kernel, the device encountered a stack error.
		/// This can be due to stack corruption or exceeding the stack size limit.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_HARDWARE_STACK_ERROR = 714,

		/// <summary>
		/// While executing a kernel, the device encountered an illegal instruction.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,

		/// <summary>
		/// While executing a kernel, the device encountered a load or store instruction
		/// on a memory address which is not aligned.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_MISALIGNED_ADDRESS = 716,

		/// <summary>
		/// While executing a kernel, the device encountered an instruction
		/// which can only operate on memory locations in certain address spaces
		/// (global, shared, or local), but was supplied a memory address not
		/// belonging to an allowed address space.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,

		/// <summary>
		/// While executing a kernel, the device program counter wrapped its address space.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_INVALID_PC = 718,

		/// <summary>
		/// An exception occurred on the device while executing a kernel. Common
		/// causes include dereferencing an invalid device pointer and accessing
		/// out of bounds shared memory.
		/// This leaves the process in an inconsistent state and any further CUDA work
		/// will return the same error. To continue using CUDA, the process must be terminated
		/// and relaunched.
		/// </summary>
		CUDA_ERROR_LAUNCH_FAILED = 719,

		/// <summary>
		/// This error indicates that the attempted operation is not permitted.
		/// </summary>
		CUDA_ERROR_NOT_PERMITTED = 800,

		/// <summary>
		/// This error indicates that the attempted operation is not supported
		/// on the current system or device.
		/// </summary>
		CUDA_ERROR_NOT_SUPPORTED = 801,

		/// <summary>
		/// This indicates that an unknown internal error has occurred.
		/// </summary>
		CUDA_ERROR_UNKNOWN = 999
	}

	/// <summary>
	/// P2P Attributes
	/// </summary>
	public enum CUdevice_P2PAttribute {
		/// <summary>
		/// A relative value indicating the performance of the link between two devices
		/// </summary>
		CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01, 
		/// <summary>
		/// P2P Access is enable
		/// </summary>
		CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02, 
		/// <summary>
		/// Atomic operation over the link supported
		/// </summary>
		CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03  
	}

	/// <summary>
	/// 2D memory copy parameters
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_MEMCPY2D {
		/// <summary>
		/// Source X in bytes
		/// </summary>
		size_t srcXInBytes;         
		/// <summary>
		/// Source Y
		/// </summary>
		size_t srcY;                

		/// <summary>
		/// Source memory type (host, device, array)
		/// </summary>
		CUmemorytype srcMemoryType; 
		/// <summary>
		/// Source host pointer
		/// </summary>
		IntPtr srcHost;             
		/// <summary>
		/// Source device pointer
		/// </summary>
		CUdeviceptr srcDevice;      
		/// <summary>
		/// Source array reference
		/// </summary>
		CUarray srcArray;           
		/// <summary>
		/// Source pitch (ignored when src is array)
		/// </summary>
		size_t srcPitch;            

		/// <summary>
		/// Destination X in bytes
		/// </summary>
		size_t dstXInBytes;         
		/// <summary>
		/// Destination Y
		/// </summary>
		size_t dstY;                

		/// <summary>
		/// Destination memory type (host, device, array)
		/// </summary>
		CUmemorytype dstMemoryType; 
		/// <summary>
		/// Destination host pointer
		/// </summary>
		IntPtr dstHost;             
		/// <summary>
		/// Destination device pointer
		/// </summary>
		CUdeviceptr dstDevice;      
		/// <summary>
		/// Destination array reference
		/// </summary>
		CUarray dstArray;           
		/// <summary>
		/// Destination pitch (ignored when dst is array)
		/// </summary>
		size_t dstPitch;            

		/// <summary>
		/// Width of 2D memory copy in bytes
		/// </summary>
		size_t WidthInBytes;        
		/// <summary>
		/// Height of 2D memory copy
		/// </summary>
		size_t Height;              
	}

	/// <summary>
	/// 3D memory copy parameters
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_MEMCPY3D {
		/// <summary>
		/// Source X in bytes
		/// </summary>
		size_t srcXInBytes;         
		/// <summary>
		/// Source Y
		/// </summary>
		size_t srcY;                
		/// <summary>
		/// Source Z
		/// </summary>
		size_t srcZ;                
		/// <summary>
		/// Source LOD
		/// </summary>
		size_t srcLOD;              
		/// <summary>
		/// Source memory type (host, device, array)
		/// </summary>
		CUmemorytype srcMemoryType; 
		/// <summary>
		/// Source host pointer
		/// </summary>
		IntPtr srcHost;             
		/// <summary>
		/// Source device pointer
		/// </summary>
		CUdeviceptr srcDevice;      
		/// <summary>
		/// Source array reference
		/// </summary>
		CUarray srcArray;           
		/// <summary>
		/// Must be NULL
		/// </summary>
		IntPtr reserved0;           
		/// <summary>
		/// Source pitch (ignored when src is array)
		/// </summary>
		size_t srcPitch;            
		/// <summary>
		/// Source height (ignored when src is array; may be 0 if Depth==1)
		/// </summary>
		size_t srcHeight;           

		/// <summary>
		/// Destination X in bytes
		/// </summary>
		size_t dstXInBytes;         
		/// <summary>
		/// Destination Y
		/// </summary>
		size_t dstY;                
		/// <summary>
		/// Destination Z
		/// </summary>
		size_t dstZ;                
		/// <summary>
		/// Destination LOD
		/// </summary>
		size_t dstLOD;              
		/// <summary>
		/// Destination memory type (host, device, array)
		/// </summary>
		CUmemorytype dstMemoryType; 
		/// <summary>
		/// Destination host pointer
		/// </summary>
		IntPtr dstHost;             
		/// <summary>
		/// Destination device pointer
		/// </summary>
		CUdeviceptr dstDevice;      
		/// <summary>
		/// Destination array reference
		/// </summary>
		CUarray dstArray;           
		/// <summary>
		/// Must be NULL
		/// </summary>
		IntPtr reserved1;           
		/// <summary>
		/// Destination pitch (ignored when dst is array)
		/// </summary>
		size_t dstPitch;            
		/// <summary>
		/// Destination height (ignored when dst is array; may be 0 if Depth==1)
		/// </summary>
		size_t dstHeight;           

		/// <summary>
		/// Width of 3D memory copy in bytes
		/// </summary>
		size_t WidthInBytes;        
		/// <summary>
		/// Height of 3D memory copy
		/// </summary>
		size_t Height;              
		/// <summary>
		/// Depth of 3D memory copy
		/// </summary>
		size_t Depth;               
	}

	/// <summary>
	/// 3D memory cross-context copy parameters
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_MEMCPY3D_PEER {
		/// <summary>
		/// Source X in bytes
		/// </summary>
		size_t srcXInBytes;         
		/// <summary>
		/// Source Y
		/// </summary>
		size_t srcY;                
		/// <summary>
		/// Source Z
		/// </summary>
		size_t srcZ;                
		/// <summary>
		/// Source LOD
		/// </summary>
		size_t srcLOD;              
		/// <summary>
		/// Source memory type (host, device, array)
		/// </summary>
		CUmemorytype srcMemoryType; 
		/// <summary>
		/// Source host pointer
		/// </summary>
		IntPtr srcHost;             
		/// <summary>
		/// Source device pointer
		/// </summary>
		CUdeviceptr srcDevice;      
		/// <summary>
		/// Source array reference
		/// </summary>
		CUarray srcArray;           
		/// <summary>
		/// Source context (ignored with srcMemoryType is ::CU_MEMORYTYPE_ARRAY)
		/// </summary>
		CUcontext srcContext;       
		/// <summary>
		/// Source pitch (ignored when src is array)
		/// </summary>
		size_t srcPitch;            
		/// <summary>
		/// Source height (ignored when src is array; may be 0 if Depth==1)
		/// </summary>
		size_t srcHeight;           

		/// <summary>
		/// Destination X in bytes
		/// </summary>
		size_t dstXInBytes;         
		/// <summary>
		/// Destination Y
		/// </summary>
		size_t dstY;                
		/// <summary>
		/// Destination Z
		/// </summary>
		size_t dstZ;                
		/// <summary>
		/// Destination LOD
		/// </summary>
		size_t dstLOD;              
		/// <summary>
		/// Destination memory type (host, device, array)
		/// </summary>
		CUmemorytype dstMemoryType; 
		/// <summary>
		/// Destination host pointer
		/// </summary>
		IntPtr dstHost;             
		/// <summary>
		/// Destination device pointer
		/// </summary>
		CUdeviceptr dstDevice;      
		/// <summary>
		/// Destination array reference
		/// </summary>
		CUarray dstArray;           
		/// <summary>
		/// Destination context (ignored with dstMemoryType is ::CU_MEMORYTYPE_ARRAY)
		/// </summary>
		CUcontext dstContext;       
		/// <summary>
		/// Destination pitch (ignored when dst is array)
		/// </summary>
		size_t dstPitch;            
		/// <summary>
		/// Destination height (ignored when dst is array; may be 0 if Depth==1)
		/// </summary>
		size_t dstHeight;           

		/// <summary>
		/// Width of 3D memory copy in bytes
		/// </summary>
		size_t WidthInBytes;        
		/// <summary>
		/// Height of 3D memory copy
		/// </summary>
		size_t Height;              
		/// <summary>
		/// Depth of 3D memory copy
		/// </summary>
		size_t Depth;               
	}

	/// <summary>
	/// Array descriptor
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_ARRAY_DESCRIPTOR {
		/// <summary>
		/// Width of array
		/// </summary>
		size_t Width;             
		/// <summary>
		/// Height of array
		/// </summary>
		size_t Height;            

		/// <summary>
		/// Array format
		/// </summary>
		CUarray_format Format;    
		/// <summary>
		/// Channels per array element
		/// </summary>
		uint NumChannels; 
	}

	/// <summary>
	/// 3D array descriptor
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct CUDA_ARRAY3D_DESCRIPTOR {
		/// <summary>
		/// Width of 3D array
		/// </summary>
		size_t Width;             
		/// <summary>
		/// Height of 3D array
		/// </summary>
		size_t Height;            
		/// <summary>
		/// Depth of 3D array
		/// </summary>
		size_t Depth;             

		/// <summary>
		/// Array format
		/// </summary>
		CUarray_format Format;    
		/// <summary>
		/// Channels per array element
		/// </summary>
		uint NumChannels; 
		/// <summary>
		/// Flags
		/// </summary>
		uint Flags;       
	}

	/// <summary>
	/// Resource view format
	/// </summary>
	public enum CUresourceViewFormat {
		/// <summary>
		/// No resource view format (use underlying resource format)
		/// </summary>
		CU_RES_VIEW_FORMAT_NONE = 0x00, 
		/// <summary>
		/// 1 channel unsigned 8-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01, 
		/// <summary>
		/// 2 channel unsigned 8-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02, 
		/// <summary>
		/// 4 channel unsigned 8-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03, 
		/// <summary>
		/// 1 channel signed 8-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04, 
		/// <summary>
		/// 2 channel signed 8-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_2X8 = 0x05, 
		/// <summary>
		/// 4 channel signed 8-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_4X8 = 0x06, 
		/// <summary>
		/// 1 channel unsigned 16-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_1X16 = 0x07, 
		/// <summary>
		/// 2 channel unsigned 16-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_2X16 = 0x08, 
		/// <summary>
		/// 4 channel unsigned 16-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_4X16 = 0x09, 
		/// <summary>
		/// 1 channel signed 16-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0a, 
		/// <summary>
		/// 2 channel signed 16-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0b, 
		/// <summary>
		/// 4 channel signed 16-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0c, 
		/// <summary>
		/// 1 channel unsigned 32-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0d, 
		/// <summary>
		/// 2 channel unsigned 32-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0e, 
		/// <summary>
		/// 4 channel unsigned 32-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0f, 
		/// <summary>
		/// 1 channel signed 32-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_1X32 = 0x10, 
		/// <summary>
		/// 2 channel signed 32-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_2X32 = 0x11, 
		/// <summary>
		/// 4 channel signed 32-bit integers
		/// </summary>
		CU_RES_VIEW_FORMAT_SINT_4X32 = 0x12, 
		/// <summary>
		/// 1 channel 16-bit floating point
		/// </summary>
		CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13, 
		/// <summary>
		/// 2 channel 16-bit floating point
		/// </summary>
		CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14, 
		/// <summary>
		/// 4 channel 16-bit floating point
		/// </summary>
		CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15, 
		/// <summary>
		/// 1 channel 32-bit floating point
		/// </summary>
		CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16, 
		/// <summary>
		/// 2 channel 32-bit floating point
		/// </summary>
		CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17, 
		/// <summary>
		/// 4 channel 32-bit floating point
		/// </summary>
		CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18, 
		/// <summary>
		/// Block compressed 1
		/// </summary>
		CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19, 
		/// <summary>
		/// Block compressed 2
		/// </summary>
		CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a, 
		/// <summary>
		/// Block compressed 3
		/// </summary>
		CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b, 
		/// <summary>
		/// Block compressed 4 unsigned
		/// </summary>
		CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c, 
		/// <summary>
		/// Block compressed 4 signed
		/// </summary>
		CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d, 
		/// <summary>
		/// Block compressed 5 unsigned
		/// </summary>
		CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e, 
		/// <summary>
		/// Block compressed 5 signed
		/// </summary>
		CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f, 
		/// <summary>
		/// Block compressed 6 unsigned half-float
		/// </summary>
		CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20, 
		/// <summary>
		/// Block compressed 6 signed half-float
		/// </summary>
		CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21, 
		/// <summary>
		/// Block compressed 7
		/// </summary>
		CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22  
	}
}
