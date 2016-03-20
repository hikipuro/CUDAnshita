using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CUDAnshita.API {
	using size_t = Int64;
	using CUdevice = Int32;
	using CUcontext = IntPtr;
	using CUmodule = IntPtr;
	using CUfunction = IntPtr;
	using CUdeviceptr = IntPtr;
	using CUstream = IntPtr;
	using CUresult = cudaError;

	/// <summary>
	/// http://docs.nvidia.com/cuda/cuda-driver-api/
	/// </summary>
	public class NvCuda {
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
		public static extern CUresult cuDeviceGetCount(ref int count);

		// ----- Context Management

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuCtxCreate(ref CUcontext pctx, uint flags, CUdevice dev);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuCtxDestroy(CUcontext ctx);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuCtxSynchronize();

		// ----- Module Management

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleLoadData(ref CUmodule module, IntPtr image);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleLoadDataEx(ref CUmodule module, IntPtr image, uint numOptions, CUjit_option options, IntPtr optionValues);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleGetFunction(ref CUfunction hfunc, CUmodule hmod, string name);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuModuleUnload(CUmodule hmod);

		// Memory Management

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemAlloc(ref CUdeviceptr dptr, size_t bytesize);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemFree(CUdeviceptr dptr);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, IntPtr srcHost, size_t ByteCount);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuMemcpyDtoH(IntPtr dstHost, CUdeviceptr srcDevice, size_t ByteCount);

		// ----- Execution Control

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern CUresult cuLaunchKernel(
			CUfunction f, 
			uint  gridDimX, uint  gridDimY, uint  gridDimZ,
			uint  blockDimX, uint  blockDimY, uint  blockDimZ,
			uint  sharedMemBytes, CUstream hStream,
			IntPtr kernelParams,
			IntPtr extra);
		
	}
}
