using System;
using System.Runtime.InteropServices;

namespace CUDAnshita.API {
	using size_t = Int64;
	using CUdevice = Int32;
	using CUcontext = IntPtr;
	using CUmodule = IntPtr;
	using CUfunction = IntPtr;
	using CUdeviceptr = IntPtr;
	using CUstream = IntPtr;
	using CUresult = cudaError;

	public enum CUjit_option {
		CU_JIT_MAX_REGISTERS = 0,
		CU_JIT_THREADS_PER_BLOCK,
		CU_JIT_WALL_TIME,
		CU_JIT_INFO_LOG_BUFFER,
		CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
		CU_JIT_ERROR_LOG_BUFFER,
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
		CU_JIT_OPTIMIZATION_LEVEL,
		CU_JIT_TARGET_FROM_CUCONTEXT,
		CU_JIT_TARGET,
		CU_JIT_FALLBACK_STRATEGY,
		CU_JIT_GENERATE_DEBUG_INFO,
		CU_JIT_LOG_VERBOSE,
		CU_JIT_GENERATE_LINE_INFO,
		CU_JIT_CACHE_MODE,

		CU_JIT_NUM_OPTIONS
	}

	public class NvCuda {
		const string DLL_PATH = "nvcuda.dll";
		const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
		const CharSet CHAR_SET = CharSet.Ansi;

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuInit(uint Flags);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuDeviceGet(ref CUdevice device, int ordinal);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuDeviceGetCount(ref int count);
		 

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuCtxCreate(ref CUcontext pctx, uint flags, CUdevice dev);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleLoadData(ref CUmodule module, IntPtr image);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleLoadDataEx(ref CUmodule module, IntPtr image, uint numOptions, CUjit_option options, IntPtr optionValues);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleGetFunction(ref CUfunction hfunc, CUmodule hmod, string name);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleUnload(CUmodule hmod);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuCtxDestroy(CUcontext ctx);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemAlloc(ref CUdeviceptr dptr, long bytesize);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemFree(CUdeviceptr dptr);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, long ByteCount);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount);
		

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuLaunchKernel(
			CUfunction f, 
			uint  gridDimX, uint  gridDimY, uint  gridDimZ,
			uint  blockDimX, uint  blockDimY, uint  blockDimZ,
			uint  sharedMemBytes, CUstream hStream,
			IntPtr kernelParams,
			IntPtr extra);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuCtxSynchronize();
		
	}
}
