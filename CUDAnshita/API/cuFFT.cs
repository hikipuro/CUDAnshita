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
	/// NVIDIA CUDA FFT library (CUFFT)
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
											   IntPtr n,
											   IntPtr inembed, // int*
											   int istride, int idist,
											   IntPtr onembed, // int*
											   int ostride, int odist,
											   cufftType type,
											   int batch);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan1d(cufftHandle plan,
												 int nx,
												 cufftType type,
												 int batch,
												 IntPtr workSize); // size_t*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan2d(cufftHandle plan,
												 int nx, int ny,
												 cufftType type,
												 IntPtr workSize); // size_t*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan3d(cufftHandle plan,
												 int nx, int ny, int nz,
												 cufftType type,
												 IntPtr workSize); // size_t*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlanMany(cufftHandle plan,
												   int rank,
												   IntPtr n, // int*
												   IntPtr inembed, // int*
												   int istride, int idist,
												   IntPtr onembed, // int*
												   int ostride, int odist,
												   cufftType type,
												   int batch,
												   IntPtr workSize); // size_t*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlanMany64(cufftHandle plan,
													 int rank,
													 IntPtr n, // long*
													 IntPtr inembed, // long*
													 long istride,
													 long idist,
													 IntPtr onembed, // long*
													 long ostride, long odist,
													 cufftType type,
													 long batch,
													 IntPtr workSize); // size_t*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSizeMany64(cufftHandle plan,
													int rank,
													IntPtr n, // long*
													IntPtr inembed, // long*
													long istride, long idist,
													IntPtr onembed, // long*
													long ostride, long odist,
													cufftType type,
													long batch,
													IntPtr workSize); // size_t*

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
												   IntPtr n, // int*
												   IntPtr inembed, // int*
												   int istride, int idist,
												   IntPtr onembed, // int*
												   int ostride, int odist,
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
												  int rank,
												  IntPtr n, // int*
												  IntPtr inembed, // int*
												  int istride, int idist,
												  IntPtr onembed, // int*
												  int ostride, int odist,
												  cufftType type, int batch,
												  ref size_t workArea);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize(cufftHandle handle, ref size_t workSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetWorkArea(cufftHandle plan, IntPtr workArea);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecC2C(cufftHandle plan,
											  IntPtr idata, // cufftComplex*
											  IntPtr odata, // cufftComplex*
											  cufftDirection direction);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecR2C(cufftHandle plan,
											  IntPtr idata,  // cufftReal*
											  IntPtr odata); // cufftComplex*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecC2R(cufftHandle plan,
											  IntPtr idata,  // cufftComplex*
											  IntPtr odata); // cufftReal*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecZ2Z(cufftHandle plan,
											  IntPtr idata, // cufftDoubleComplex*
											  IntPtr odata, // cufftDoubleComplex*
											  cufftDirection direction);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecD2Z(cufftHandle plan,
											  IntPtr idata,  // cufftDoubleReal*
											  IntPtr odata); // cufftDoubleComplex*

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecZ2D(cufftHandle plan,
											  IntPtr idata,  // cufftDoubleComplex*
											  IntPtr odata); // cufftDoubleReal*


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

		/// <summary>
		/// CUFFT transform directions 
		/// Forward FFT
		/// </summary>
		public const int CUFFT_FORWARD = -1;

		/// <summary>
		/// CUFFT transform directions 
		/// Inverse FFT
		/// </summary>
		public const int CUFFT_INVERSE = 1;

		/// <summary>
		/// structure definition used by the shim between old and new APIs
		/// </summary>
		public const int MAX_SHIM_RANK = 3;

		// ----- C# Interface

		public static cufftHandle Plan1d(int nx, cufftType type, int batch) {
			cufftHandle plan = 0;
			CheckStatus(API.cufftPlan1d(ref plan, nx, type, batch));
			return plan;
		}

		public static cufftHandle Plan2d(int nx, int ny, cufftType type) {
			cufftHandle plan = 0;
			CheckStatus(API.cufftPlan2d(ref plan, nx, ny, type));
			return plan;
		}

		public static cufftHandle Plan3d(int nx, int ny, int nz, cufftType type) {
			cufftHandle plan = 0;
			CheckStatus(API.cufftPlan3d(ref plan, nx, ny, nz, type));
			return plan;
		}

		public static cufftHandle PlanMany(int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch) {
			cufftHandle plan = 0;
			CheckStatus(API.cufftPlanMany(
				ref plan,
				rank,
				n,
				inembed, istride, idist,
				onembed, ostride, odist,
				type,
				batch
			));
			return plan;
		}

		public static IntPtr MakePlan1d(cufftHandle plan, int nx, cufftType type, int batch) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlan1d(plan, nx, type, batch, workSize));
			return workSize;
		}

		public static IntPtr MakePlan2d(cufftHandle plan, int nx, int ny, cufftType type) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlan2d(plan, nx, ny, type, workSize));
			return workSize;
		}

		public static IntPtr MakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlan3d(plan, nx, ny, nz, type, workSize));
			return workSize;
		}

		public static IntPtr MakePlanMany(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlanMany(
				plan,
				rank,
				n,
				inembed, istride, idist,
				onembed, ostride, odist,
				type,
				batch,
				workSize
			));
			return workSize;
		}

		public static IntPtr MakePlanMany64(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, IntPtr onembed, long ostride, long odist, cufftType type, long batch) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlanMany64(
				plan,
				rank,
				n,
				inembed, istride, idist,
				onembed, ostride, odist,
				type,
				batch,
				workSize
			));
			return workSize;
		}

		public static IntPtr GetSizeMany64(cufftHandle plan, int rank, IntPtr n, IntPtr inembed, long istride, long idist, IntPtr onembed, long ostride, long odist, cufftType type, long batch) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftGetSizeMany64(
				plan,
				rank,
				n,
				inembed, istride, idist,
				onembed, ostride, odist,
				type,
				batch,
				workSize
			));
			return workSize;
		}

		public static size_t Estimate1d(int nx, cufftType type, int batch) {
			size_t workSize = 0;
			CheckStatus(API.cufftEstimate1d(nx, type, batch, ref workSize));
			return workSize;
		}

		public static size_t Estimate2d(int nx, int ny, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftEstimate2d(nx, ny, type, ref workSize));
			return workSize;
		}

		public static size_t Estimate3d(int nx, int ny, int nz, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftEstimate3d(nx, ny, nz, type, ref workSize));
			return workSize;
		}

		public static size_t EstimateMany(int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch) {
			size_t workSize = 0;
			CheckStatus(API.cufftEstimateMany(
				rank,
				n,
				inembed, istride, idist,
				onembed, ostride, odist,
				type,
				batch,
				ref workSize
			));
			return workSize;
		}

		public static cufftHandle Create() {
			cufftHandle handle = 0;
			CheckStatus(API.cufftCreate(ref handle));
			return handle;
		}

		public static size_t GetSize1d(cufftHandle handle, int nx, cufftType type, int batch) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize1d(handle, nx, type, batch, ref workSize));
			return workSize;
		}

		public static size_t GetSize2d(cufftHandle handle, int nx, int ny, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize2d(handle, nx, ny, type, ref workSize));
			return workSize;
		}

		public static size_t GetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize3d(handle, nx, ny, nz, type, ref workSize));
			return workSize;
		}

		public static size_t GetSizeMany(cufftHandle handle, int rank, IntPtr n, IntPtr inembed, int istride, int idist, IntPtr onembed, int ostride, int odist, cufftType type, int batch) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSizeMany(
				handle,
				rank, n,
				inembed, istride, idist,
				onembed, ostride, odist,
				type, batch,
				ref workSize
			));
			return workSize;
		}

		public static size_t GetSize(cufftHandle handle) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize(handle, ref workSize));
			return workSize;
		}

		public static void SetWorkArea(cufftHandle plan, IntPtr workArea) {
			CheckStatus(API.cufftSetWorkArea(plan, workArea));
		}

		public static void SetAutoAllocation(cufftHandle plan, int autoAllocate) {
			CheckStatus(API.cufftSetAutoAllocation(plan, autoAllocate));
		}

		public static void ExecC2C(cufftHandle plan, IntPtr idata, IntPtr odata, cufftDirection direction) {
			CheckStatus(API.cufftExecC2C(plan, idata, odata, direction));
		}

		public static void ExecR2C(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecR2C(plan, idata, odata));
		}

		public static void ExecC2R(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecC2R(plan, idata, odata));
		}

		public static void ExecZ2Z(cufftHandle plan, IntPtr idata, IntPtr odata, cufftDirection direction) {
			CheckStatus(API.cufftExecZ2Z(plan, idata, odata, direction));
		}

		public static void ExecD2Z(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecD2Z(plan, idata, odata));
		}

		public static void ExecZ2D(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecZ2D(plan, idata, odata));
		}

		public static void SetStream(cufftHandle plan, cudaStream_t stream) {
			CheckStatus(API.cufftSetStream(plan, stream));
		}

		public static void SetCompatibilityMode(cufftHandle plan, cufftCompatibility mode) {
			CheckStatus(API.cufftSetCompatibilityMode(plan, mode));
		}

		public static void Destroy(cufftHandle plan) {
			CheckStatus(API.cufftDestroy(plan));
		}

		public static int GetVersion() {
			int version = 0;
			CheckStatus(API.cufftGetVersion(ref version));
			return version;
		}

		public static int GetProperty(libraryPropertyType type) {
			int value = 0;
			CheckStatus(API.cufftGetProperty(type, ref value));
			return value;
		}

		static void CheckStatus(cufftResult status) {
			if (status != cufftResult.CUFFT_SUCCESS) {
				throw new CudaException(status.ToString());
			}
		}
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
		/// <summary>
		/// Real to Complex (interleaved)
		/// </summary>
		CUFFT_R2C = 0x2a,

		/// <summary>
		/// Complex (interleaved) to Real
		/// </summary>
		CUFFT_C2R = 0x2c,

		/// <summary>
		/// Complex to Complex, interleaved
		/// </summary>
		CUFFT_C2C = 0x29,

		/// <summary>
		/// Double to Double-Complex
		/// </summary>
		CUFFT_D2Z = 0x6a,

		/// <summary>
		/// Double-Complex to Double
		/// </summary>
		CUFFT_Z2D = 0x6c,

		/// <summary>
		/// Double-Complex to Double-Complex
		/// </summary>
		CUFFT_Z2Z = 0x69
	}

	/// <summary>
	/// CUFFT supports the following data layouts
	/// </summary>
	public enum cufftCompatibility {
		/// <summary>
		/// The default value
		/// </summary>
		CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01
	}

	public enum cufftDirection {
		CUFFT_FORWARD = -1,
		CUFFT_INVERSE = 1
	}
}
