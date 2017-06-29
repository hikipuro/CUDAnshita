using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using cufftHandle = Int32;
	using cudaStream_t = IntPtr;
	using cufftReal = Single;
	using cufftComplex = float2;
	using cufftDoubleReal = Double;
	using cufftDoubleComplex = double2;
	using size_t = Int64;

	/// <summary>
	/// Fast Fourier Transform (FFT) library.
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cufft/">http://docs.nvidia.com/cuda/cufft/</a>
	/// </remarks>
	public class cuFFT {
		public class API {
			const string DLL_PATH = "cufft64_80.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftPlan1d(ref cufftHandle plan,
											 int nx,
											 cufftType type,
											 int batch);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftPlan2d(ref cufftHandle plan,
											 int nx, int ny,
											 cufftType type);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftPlan3d(ref cufftHandle plan,
											 int nx, int ny, int nz,
											 cufftType type);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftPlanMany(ref cufftHandle plan,
											   int rank,
											   ref int n,
											   ref int inembed, int istride, int idist,
											   ref int onembed, int ostride, int odist,
											   cufftType type,
											   int batch);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan1d(cufftHandle plan,
												 int nx,
												 cufftType type,
												 int batch,
												 ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan2d(cufftHandle plan,
												 int nx, int ny,
												 cufftType type,
												 ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan3d(cufftHandle plan,
												 int nx, int ny, int nz,
												 cufftType type,
												 ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlanMany(cufftHandle plan,
												   int rank,
												   ref int n,
												   ref int inembed, int istride, int idist,
												   ref int onembed, int ostride, int odist,
												   cufftType type,
												   int batch,
												   ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlanMany64(cufftHandle plan,
													 int rank,
													 ref long n,
													 ref long inembed,
													 long istride,
													 long idist,
													 ref long onembed,
													 long ostride, long odist,
													 cufftType type,
													 long batch,
													 ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSizeMany64(cufftHandle plan,
													int rank,
													ref long n,
													ref long inembed,
													long istride, long idist,
													ref long onembed,
													long ostride, long odist,
													cufftType type,
													long batch,
													ref size_t workSize);




			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftEstimate1d(int nx,
												 cufftType type,
												 int batch,
												 ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftEstimate2d(int nx, int ny,
												 cufftType type,
												 ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftEstimate3d(int nx, int ny, int nz,
												 cufftType type,
												 ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftEstimateMany(int rank,
												   ref int n,
												   ref int inembed, int istride, int idist,
												   ref int onembed, int ostride, int odist,
												   cufftType type,
												   int batch,
												   ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftCreate(ref cufftHandle handle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize1d(cufftHandle handle,
												int nx,
												cufftType type,
												int batch,
												ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize2d(cufftHandle handle,
												int nx, int ny,
												cufftType type,
												ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize3d(cufftHandle handle,
												int nx, int ny, int nz,
												cufftType type,
												ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSizeMany(cufftHandle handle,
												  int rank, ref int n,
												  ref int inembed, int istride, int idist,
												  ref int onembed, int ostride, int odist,
												  cufftType type, int batch, ref size_t workArea);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize(cufftHandle handle, ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetWorkArea(cufftHandle plan, IntPtr workArea);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecC2C(cufftHandle plan,
											  ref cufftComplex idata,
											  ref cufftComplex odata,
											  int direction);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecR2C(cufftHandle plan,
											  ref cufftReal idata,
											  ref cufftComplex odata);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecC2R(cufftHandle plan,
											  ref cufftComplex idata,
											  ref cufftReal odata);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecZ2Z(cufftHandle plan,
											  ref cufftDoubleComplex idata,
											  ref cufftDoubleComplex odata,
											  int direction);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecD2Z(cufftHandle plan,
											  ref cufftDoubleReal idata,
											  ref cufftDoubleComplex odata);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecZ2D(cufftHandle plan,
											  ref cufftDoubleComplex idata,
											  ref cufftDoubleReal odata);


			// utility functions
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetStream(cufftHandle plan,
												cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetCompatibilityMode(cufftHandle plan,
														   cufftCompatibility mode);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftDestroy(cufftHandle plan);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetVersion(ref int version);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetProperty(libraryPropertyType type,
												  ref int value);


		}

		public const int MAX_CUFFT_ERROR = 0x11;
		public const int CUFFT_FORWARD = -1;
		public const int CUFFT_INVERSE = 1;
		public const int MAX_SHIM_RANK = 3;
	}

	/// <summary>
	/// CUFFT API function return values
	/// </summary>
	public enum cufftResult {
		CUFFT_SUCCESS = 0x0,
		CUFFT_INVALID_PLAN = 0x1,
		CUFFT_ALLOC_FAILED = 0x2,
		CUFFT_INVALID_TYPE = 0x3,
		CUFFT_INVALID_VALUE = 0x4,
		CUFFT_INTERNAL_ERROR = 0x5,
		CUFFT_EXEC_FAILED = 0x6,
		CUFFT_SETUP_FAILED = 0x7,
		CUFFT_INVALID_SIZE = 0x8,
		CUFFT_UNALIGNED_DATA = 0x9,
		CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
		CUFFT_INVALID_DEVICE = 0xB,
		CUFFT_PARSE_ERROR = 0xC,
		CUFFT_NO_WORKSPACE = 0xD,
		CUFFT_NOT_IMPLEMENTED = 0xE,
		CUFFT_LICENSE_ERROR = 0x0F,
		CUFFT_NOT_SUPPORTED = 0x10
	}

	/// <summary>
	/// CUFFT supports the following transform types
	/// </summary>
	public enum cufftType {
		CUFFT_R2C = 0x2a,     // Real to Complex (interleaved)
		CUFFT_C2R = 0x2c,     // Complex (interleaved) to Real
		CUFFT_C2C = 0x29,     // Complex to Complex, interleaved
		CUFFT_D2Z = 0x6a,     // Double to Double-Complex
		CUFFT_Z2D = 0x6c,     // Double-Complex to Double
		CUFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
	}

	/// <summary>
	/// CUFFT supports the following data layouts
	/// </summary>
	public enum cufftCompatibility {
		CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01    // The default value
	}
}
