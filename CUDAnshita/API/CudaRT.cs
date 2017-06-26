using System.Runtime.InteropServices;

namespace CUDAnshita.API {
	/// <summary>
	/// http://docs.nvidia.com/cuda/cuda-runtime-api/
	/// </summary>
	public class CudaRT {
		const string DLL_PATH = "cudart64_80.dll";
		const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
		const CharSet CHAR_SET = CharSet.Ansi;

		// ----- Device Management

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaGetDeviceCount(ref int count);

		// ----- Version Management

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaDriverGetVersion(ref int driverVersion);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaRuntimeGetVersion(ref int runtimeVersion);

		// ----- Profiler Control

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaProfilerStart();

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		public static extern cudaError cudaProfilerStop();
	}
}
