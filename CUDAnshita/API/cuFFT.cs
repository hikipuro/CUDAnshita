using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using cufftHandle = Int32;
	using cudaStream_t = IntPtr;
	using cufftReal = Single;
	using cufftComplex = float2;
	using cufftDoubleReal = Double;
	using cufftDoubleComplex = double2;
	using cufftXtWorkAreaPolicy = Int32;
	using size_t = Int64;

	/// <summary>
	/// NVIDIA CUDA FFT library (CUFFT)
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cufft/">http://docs.nvidia.com/cuda/cufft/</a>
	/// </remarks>
	public class cuFFT {
		/// <summary>
		/// FFT library DLL functions.
		/// </summary>
		public class API {
			//const string DLL_PATH = "cufft64_80.dll";
			const string DLL_PATH = "cufft64_10.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			/// <summary>
			/// Creates a 1D FFT plan configuration for a specified signal size and data type.
			/// The batch input parameter tells cuFFT how many 1D transforms to configure.
			/// </summary>
			/// <param name="plan">Pointer to a cufftHandle object.</param>
			/// <param name="nx">The transform size (e.g. 256 for a 256-point FFT)</param>
			/// <param name="type">The transform data type (e.g., CUFFT_C2C for single precision complex to complex)</param>
			/// <param name="batch">Number of transforms of size nx. Please consider using cufftPlanMany for multiple transforms.</param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftPlan1d(ref cufftHandle plan,
											 int nx,
											 cufftType type,
											 int batch);

			/// <summary>
			/// Creates a 2D FFT plan configuration according to specified signal sizes and data type.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="type"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftPlan2d(ref cufftHandle plan,
											 int nx, int ny,
											 cufftType type);

			/// <summary>
			/// Creates a 3D FFT plan configuration according to specified signal sizes and data type.
			/// This function is the same as cufftPlan2d() except that it takes a third size parameter nz.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="nz"></param>
			/// <param name="type"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftPlan3d(ref cufftHandle plan,
											 int nx, int ny, int nz,
											 cufftType type);

			/// <summary>
			/// Creates a FFT plan configuration of dimension rank, with sizes specified in the array n.
			/// The batch input parameter tells cuFFT how many transforms to configure.
			/// With this function, batched plans of 1, 2, or 3 dimensions may be created.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <returns></returns>
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

			/// <summary>
			/// Following a call to cufftCreate() makes a 1D FFT plan configuration
			/// for a specified signal size and data type.
			/// The batch input parameter tells cuFFT how many 1D transforms to configure.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="nx"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan1d(cufftHandle plan,
												 int nx,
												 cufftType type,
												 int batch,
												 IntPtr workSize); // size_t*

			/// <summary>
			/// Following a call to cufftCreate() makes a 2D FFT plan configuration
			/// according to specified signal sizes and data type.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="type"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan2d(cufftHandle plan,
												 int nx, int ny,
												 cufftType type,
												 IntPtr workSize); // size_t*

			/// <summary>
			/// Following a call to cufftCreate() makes a 3D FFT plan configuration
			/// according to specified signal sizes and data type.
			/// This function is the same as cufftPlan2d() except that it takes a third size parameter nz.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="nz"></param>
			/// <param name="type"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftMakePlan3d(cufftHandle plan,
												 int nx, int ny, int nz,
												 cufftType type,
												 IntPtr workSize); // size_t*

			/// <summary>
			/// Following a call to cufftCreate() makes a FFT plan configuration of dimension rank,
			/// with sizes specified in the array n.
			/// The batch input parameter tells cuFFT how many transforms to configure.
			/// With this function, batched plans of 1, 2, or 3 dimensions may be created.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
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

			/// <summary>
			/// Following a call to cufftCreate() makes a FFT plan configuration of dimension rank,
			/// with sizes specified in the array n.
			/// The batch input parameter tells cuFFT how many transforms to configure.
			/// With this function, batched plans of 1, 2, or 3 dimensions may be created.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
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

			/// <summary>
			/// Following a call to cufftCreate() makes an FFT plan configuration of dimension rank,
			/// with sizes specified in the array n.
			/// The batch input parameter tells cuFFT how many transforms to configure.
			/// With this function, batched plans of 1, 2, or 3 dimensions may be created.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="inputtype"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="outputtype"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <param name="executiontype"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtMakePlanMany(cufftHandle plan,
													int rank,
													IntPtr n, // long long int*
													IntPtr inembed, // long long int*
													long istride, // long long int
													long idist, // long long int
													cudaDataType inputtype,
													IntPtr onembed, // long long int*
													long ostride, // long long int
													long odist, // long long int
													cudaDataType outputtype,
													long batch, // long long int
													IntPtr workSize, // size_t*
													cudaDataType executiontype);

			/// <summary>
			/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimateSizeMany(),
			/// given the specified parameters, and taking into account any plan settings that may have been made.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
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

			/// <summary>
			/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimateSizeMany(),
			/// given the specified parameters that match signature of cufftXtMakePlanMany function,
			/// and taking into account any plan settings that may have been made.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="inputtype"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="outputtype"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <param name="executiontype"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtGetSizeMany(cufftHandle plan,
													int rank,
													IntPtr n, // long long int*
													IntPtr inembed, // long long int*
													long istride, // long long int
													long idist, // long long int
													cudaDataType inputtype,
													IntPtr onembed, // long long int*
													long ostride, // long long int
													long odist, // long long int
													cudaDataType outputtype,
													long batch, // long long int
													IntPtr workSize, // size_t*
													cudaDataType executiontype);

			/// <summary>
			/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
			/// This call returns an estimate for the size of the work area required,
			/// given the specified parameters, and assuming default plan settings.
			/// </summary>
			/// <param name="nx"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftEstimate1d(int nx,
												 cufftType type,
												 int batch,
												 ref size_t workSize);

			/// <summary>
			/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
			/// This call returns an estimate for the size of the work area required,
			/// given the specified parameters, and assuming default plan settings.
			/// </summary>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="type"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftEstimate2d(int nx, int ny,
												 cufftType type,
												 ref size_t workSize);

			/// <summary>
			/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
			/// This call returns an estimate for the size of the work area required,
			/// given the specified parameters, and assuming default plan settings.
			/// </summary>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="nz"></param>
			/// <param name="type"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftEstimate3d(int nx, int ny, int nz,
												 cufftType type,
												 ref size_t workSize);

			/// <summary>
			/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
			/// This call returns an estimate for the size of the work area required,
			/// given the specified parameters, and assuming default plan settings.
			/// </summary>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
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

			/// <summary>
			/// Creates only an opaque handle, and allocates small data structures on the host.
			/// The cufftMakePlan*() calls actually do the plan generation.
			/// </summary>
			/// <param name="handle"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftCreate(ref cufftHandle handle);

			/// <summary>
			/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimate1d(),
			/// given the specified parameters, and taking into account any plan settings that may have been made.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="nx"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize1d(cufftHandle handle,
												int nx,
												cufftType type,
												int batch,
												ref size_t workSize);

			/// <summary>
			/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimate2d(),
			/// given the specified parameters, and taking into account any plan settings that may have been made.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="type"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize2d(cufftHandle handle,
												int nx, int ny,
												cufftType type,
												ref size_t workSize);

			/// <summary>
			/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimate3d(),
			/// given the specified parameters, and taking into account any plan settings that may have been made.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="nx"></param>
			/// <param name="ny"></param>
			/// <param name="nz"></param>
			/// <param name="type"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize3d(cufftHandle handle,
												int nx, int ny, int nz,
												cufftType type,
												ref size_t workSize);

			/// <summary>
			/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimateSizeMany(),
			/// given the specified parameters, and taking into account any plan settings that may have been made.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="rank"></param>
			/// <param name="n"></param>
			/// <param name="inembed"></param>
			/// <param name="istride"></param>
			/// <param name="idist"></param>
			/// <param name="onembed"></param>
			/// <param name="ostride"></param>
			/// <param name="odist"></param>
			/// <param name="type"></param>
			/// <param name="batch"></param>
			/// <param name="workArea"></param>
			/// <returns></returns>
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

			/// <summary>
			/// Once plan generation has been done, either with the original API or the extensible API,
			/// this call returns the actual size of the work area required to support the plan.
			/// Callers who choose to manage work area allocation within their application
			/// must use this call after plan generation, and after any cufftSet*() calls subsequent
			/// to plan generation, if those calls might alter the required work space size.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetSize(cufftHandle handle, ref size_t workSize);

			/// <summary>
			/// cufftSetWorkArea() overrides the work area pointer associated with a plan.
			/// If the work area was auto-allocated, cuFFT frees the auto-allocated space.
			/// The cufftExecute*() calls assume that the work area pointer is valid and
			/// that it points to a contiguous region in device memory that does not
			/// overlap with any other work area. If this is not the case, results are indeterminate.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="workArea"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetWorkArea(cufftHandle plan, IntPtr workArea);

			/// <summary>
			/// cufftSetAutoAllocation() indicates that the caller intends to allocate and manage work areas
			/// for plans that have been generated. cuFFT default behavior is to allocate the work area
			/// at plan generation time.
			/// If cufftSetAutoAllocation() has been called with autoAllocate set to 0 ("false") prior
			/// to one of the cufftMakePlan*() calls, cuFFT does not allocate the work area.
			/// This is the preferred sequence for callers wishing to manage work area allocation.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="autoAllocate"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);

			/// <summary>
			/// cufftXtSetWorkAreaPolicy() indicates that the caller intends to change work area size for a given plan handle.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="policy"></param>
			/// <param name="workSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtSetWorkAreaPolicy(cufftHandle plan,
												cufftXtWorkAreaPolicy policy, // cufftXtWorkAreaPolicy
												IntPtr workSize); // size_t*

			/// <summary>
			/// cufftExecC2C() (cufftExecZ2Z()) executes a single-precision (double-precision) complex-to-complex
			/// transform plan in the transform direction as specified by direction parameter.
			/// cuFFT uses the GPU memory pointed to by the idata parameter as input data.
			/// This function stores the Fourier coefficients in the odata array.
			/// If idata and odata are the same, this method does an in-place transform.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="idata"></param>
			/// <param name="odata"></param>
			/// <param name="direction"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecC2C(cufftHandle plan,
											  IntPtr idata, // cufftComplex*
											  IntPtr odata, // cufftComplex*
											  cufftDirection direction);

			/// <summary>
			/// cufftExecR2C() (cufftExecD2Z()) executes a single-precision (double-precision) real-to-complex,
			/// implicitly forward, cuFFT transform plan.
			/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="idata"></param>
			/// <param name="odata"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecR2C(cufftHandle plan,
											  IntPtr idata,  // cufftReal*
											  IntPtr odata); // cufftComplex*

			/// <summary>
			/// cufftExecC2R() (cufftExecZ2D()) executes a single-precision (double-precision) complex-to-real,
			/// implicitly inverse, cuFFT transform plan.
			/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="idata"></param>
			/// <param name="odata"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecC2R(cufftHandle plan,
											  IntPtr idata,  // cufftComplex*
											  IntPtr odata); // cufftReal*

			/// <summary>
			/// cufftExecC2C() (cufftExecZ2Z()) executes a single-precision (double-precision) complex-to-complex
			/// transform plan in the transform direction as specified by direction parameter.
			/// cuFFT uses the GPU memory pointed to by the idata parameter as input data.
			/// This function stores the Fourier coefficients in the odata array.
			/// If idata and odata are the same, this method does an in-place transform.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="idata"></param>
			/// <param name="odata"></param>
			/// <param name="direction"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecZ2Z(cufftHandle plan,
											  IntPtr idata, // cufftDoubleComplex*
											  IntPtr odata, // cufftDoubleComplex*
											  cufftDirection direction);

			/// <summary>
			/// cufftExecR2C() (cufftExecD2Z()) executes a single-precision (double-precision) real-to-complex,
			/// implicitly forward, cuFFT transform plan.
			/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="idata"></param>
			/// <param name="odata"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecD2Z(cufftHandle plan,
											  IntPtr idata,  // cufftDoubleReal*
											  IntPtr odata); // cufftDoubleComplex*

			/// <summary>
			/// cufftExecC2R() (cufftExecZ2D()) executes a single-precision (double-precision) complex-to-real,
			/// implicitly inverse, cuFFT transform plan.
			/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="idata"></param>
			/// <param name="odata"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftExecZ2D(cufftHandle plan,
												IntPtr idata,  // cufftDoubleComplex*
												IntPtr odata); // cufftDoubleReal*

			/// <summary>
			/// Function cufftXtExec executes any cuFFT transform regardless of precision and type.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="input"></param>
			/// <param name="output"></param>
			/// <param name="direction"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtExec(cufftHandle plan,
												IntPtr input, // void*
												IntPtr output, // void*
												int direction);

			/// <summary>
			/// Function cufftXtExecDescriptor() executes any cuFFT transform regardless of precision and type.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="input"></param>
			/// <param name="output"></param>
			/// <param name="direction"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtExecDescriptor(cufftHandle plan,
												IntPtr input, // cudaLibXtDesc*
												IntPtr output, // cudaLibXtDesc*
												int direction);

			/// <summary>
			/// cufftXtSetGPUs() indentifies which GPUs are to be used with the plan.
			/// As in the single GPU case cufftCreate() creates a plan and cufftMakePlan*() does the plan generation.
			/// This call will return an error if a non-default stream has been associated with the plan.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="nGPUs"></param>
			/// <param name="whichGPUs"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtSetGPUs(cufftHandle plan,
												int nGPUs,
												IntPtr whichGPUs); // int*

			/// <summary>
			/// cufftXtSetWorkArea() overrides the work areas associated with a plan.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="workArea"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtSetWorkArea(cufftHandle plan,
												IntPtr workArea); // void**

			/// <summary>
			/// cufftXtExecDescriptorC2C() (cufftXtExecDescriptorZ2Z()) executes a single-precision (double-precision)
			/// complex-to-complex transform plan in the transform direction as specified by direction parameter.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="input"></param>
			/// <param name="output"></param>
			/// <param name="direction"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtExecDescriptorC2C(cufftHandle plan,
												IntPtr input, // cudaLibXtDesc*
												IntPtr output, // cudaLibXtDesc*
												int direction);

			/// <summary>
			/// cufftXtExecDescriptorC2C() (cufftXtExecDescriptorZ2Z()) executes a single-precision (double-precision)
			/// complex-to-complex transform plan in the transform direction as specified by direction parameter.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="input"></param>
			/// <param name="output"></param>
			/// <param name="direction"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan,
												IntPtr input, // cudaLibXtDesc*
												IntPtr output, // cudaLibXtDesc*
												int direction);

			/// <summary>
			/// cufftXtMalloc() allocates a descriptor, and all memory for data in GPUs associated with the plan,
			/// and returns a pointer to the descriptor.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="descriptor"></param>
			/// <param name="format"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtMalloc(cufftHandle plan,
												IntPtr descriptor, // cudaLibXtDesc**
												cufftXtSubFormat format); // cufftXtSubFormat

			/// <summary>
			/// cufftXtFree() frees the descriptor and all memory associated with it.
			/// The descriptor and memory must have been returned by a previous call to cufftXtMalloc().
			/// </summary>
			/// <param name="descriptor"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtFree(ref cudaLibXtDesc descriptor);

			/// <summary>
			/// cufftXtMemcpy() copies data between buffers on the host and GPUs or between GPUs.
			/// The enumerated parameter cufftXtCopyType_t indicates the type and direction of transfer.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="dstPointer"></param>
			/// <param name="srcPointer"></param>
			/// <param name="type"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtMemcpy(cufftHandle plan,
												IntPtr dstPointer, // void*
												IntPtr srcPointer, // void*
												cufftXtCopyType type);

			/// <summary>
			/// cufftXtSetCallback() specifies a load or store callback to be used with the plan.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="callbackRoutine"></param>
			/// <param name="type"></param>
			/// <param name="callerInfo"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtSetCallback(cufftHandle plan,
												IntPtr callbackRoutine, // void**
												cufftXtCallbackType type,
												IntPtr callerInfo); // void**

			/// <summary>
			/// cufftXtClearCallback() instructs cuFFT to stop invoking the specified callback type when executing the plan.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="type"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtClearCallback(cufftHandle plan,
												cufftXtCallbackType type);

			/// <summary>
			/// cufftXtSetCallbackSharedSize() instructs cuFFT to dynamically allocate shared memory at launch time,
			/// for use by the callback.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="type"></param>
			/// <param name="sharedSize"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan,
												cufftXtCallbackType type,
												size_t sharedSize);

			// utility functions

			/// <summary>
			/// Associates a CUDA stream with a cuFFT plan.
			/// </summary>
			/// <param name="plan"></param>
			/// <param name="stream"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetStream(cufftHandle plan,
												cudaStream_t stream);

			[Obsolete("Function cufftSetCompatibilityMode was removed in version 9.1.")]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftSetCompatibilityMode(cufftHandle plan,
														   cufftCompatibility mode);

			/// <summary>
			/// Frees all GPU resources associated with a cuFFT plan and destroys the internal plan data structure.
			/// This function should be called once a plan is no longer needed, to avoid wasting GPU memory.
			/// </summary>
			/// <param name="plan">The cufftHandle object of the plan to be destroyed.</param>
			/// <returns>CUFFT_SUCCESS, CUFFT_INVALID_PLAN</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftDestroy(cufftHandle plan);

			/// <summary>
			/// Returns the version number of cuFFT.
			/// </summary>
			/// <param name="version"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cufftResult cufftGetVersion(ref int version);

			/// <summary>
			/// Return in *value the number for the property described by type of the dynamically linked CUFFT library.
			/// </summary>
			/// <param name="type"></param>
			/// <param name="value"></param>
			/// <returns></returns>
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

		public const int MAX_CUDA_DESCRIPTOR_GPUS = 64;
		public const int cuFFTFORWARD = -1;
		public const int cuFFTINVERSE = 1;

		// ----- C# Interface

		/// <summary>
		/// Creates a 1D FFT plan configuration for a specified signal size and data type.
		/// The batch input parameter tells cuFFT how many 1D transforms to configure.
		/// </summary>
		/// <param name="nx">The transform size (e.g. 256 for a 256-point FFT)</param>
		/// <param name="type">The transform data type (e.g., CUFFT_C2C for single precision complex to complex)</param>
		/// <param name="batch">Number of transforms of size nx. Please consider using cufftPlanMany for multiple transforms.</param>
		/// <returns>Pointer to a cufftHandle object.</returns>
		public static cufftHandle Plan1d(int nx, cufftType type, int batch) {
			cufftHandle plan = 0;
			CheckStatus(API.cufftPlan1d(ref plan, nx, type, batch));
			return plan;
		}

		/// <summary>
		/// Creates a 2D FFT plan configuration according to specified signal sizes and data type.
		/// </summary>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static cufftHandle Plan2d(int nx, int ny, cufftType type) {
			cufftHandle plan = 0;
			CheckStatus(API.cufftPlan2d(ref plan, nx, ny, type));
			return plan;
		}

		/// <summary>
		/// Creates a 3D FFT plan configuration according to specified signal sizes and data type.
		/// This function is the same as cufftPlan2d() except that it takes a third size parameter nz.
		/// </summary>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="nz"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static cufftHandle Plan3d(int nx, int ny, int nz, cufftType type) {
			cufftHandle plan = 0;
			CheckStatus(API.cufftPlan3d(ref plan, nx, ny, nz, type));
			return plan;
		}

		/// <summary>
		/// Creates a FFT plan configuration of dimension rank, with sizes specified in the array n.
		/// The batch input parameter tells cuFFT how many transforms to configure.
		/// With this function, batched plans of 1, 2, or 3 dimensions may be created.
		/// </summary>
		/// <param name="rank"></param>
		/// <param name="n"></param>
		/// <param name="inembed"></param>
		/// <param name="istride"></param>
		/// <param name="idist"></param>
		/// <param name="onembed"></param>
		/// <param name="ostride"></param>
		/// <param name="odist"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
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

		/// <summary>
		/// Following a call to cufftCreate() makes a 1D FFT plan configuration
		/// for a specified signal size and data type.
		/// The batch input parameter tells cuFFT how many 1D transforms to configure.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="nx"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
		public static IntPtr MakePlan1d(cufftHandle plan, int nx, cufftType type, int batch) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlan1d(plan, nx, type, batch, workSize));
			return workSize;
		}

		/// <summary>
		/// Following a call to cufftCreate() makes a 2D FFT plan configuration
		/// according to specified signal sizes and data type.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static IntPtr MakePlan2d(cufftHandle plan, int nx, int ny, cufftType type) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlan2d(plan, nx, ny, type, workSize));
			return workSize;
		}

		/// <summary>
		/// Following a call to cufftCreate() makes a 3D FFT plan configuration
		/// according to specified signal sizes and data type.
		/// This function is the same as cufftPlan2d() except that it takes a third size parameter nz.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="nz"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static IntPtr MakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type) {
			IntPtr workSize = IntPtr.Zero;
			CheckStatus(API.cufftMakePlan3d(plan, nx, ny, nz, type, workSize));
			return workSize;
		}

		/// <summary>
		/// Following a call to cufftCreate() makes a FFT plan configuration of dimension rank,
		/// with sizes specified in the array n.
		/// The batch input parameter tells cuFFT how many transforms to configure.
		/// With this function, batched plans of 1, 2, or 3 dimensions may be created.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="rank"></param>
		/// <param name="n"></param>
		/// <param name="inembed"></param>
		/// <param name="istride"></param>
		/// <param name="idist"></param>
		/// <param name="onembed"></param>
		/// <param name="ostride"></param>
		/// <param name="odist"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
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

		/// <summary>
		/// Following a call to cufftCreate() makes a FFT plan configuration of dimension rank,
		/// with sizes specified in the array n.
		/// The batch input parameter tells cuFFT how many transforms to configure.
		/// With this function, batched plans of 1, 2, or 3 dimensions may be created.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="rank"></param>
		/// <param name="n"></param>
		/// <param name="inembed"></param>
		/// <param name="istride"></param>
		/// <param name="idist"></param>
		/// <param name="onembed"></param>
		/// <param name="ostride"></param>
		/// <param name="odist"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
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

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimateSizeMany(),
		/// given the specified parameters, and taking into account any plan settings that may have been made.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="rank"></param>
		/// <param name="n"></param>
		/// <param name="inembed"></param>
		/// <param name="istride"></param>
		/// <param name="idist"></param>
		/// <param name="onembed"></param>
		/// <param name="ostride"></param>
		/// <param name="odist"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
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

		/// <summary>
		/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
		/// This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings.
		/// </summary>
		/// <param name="nx"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
		public static size_t Estimate1d(int nx, cufftType type, int batch) {
			size_t workSize = 0;
			CheckStatus(API.cufftEstimate1d(nx, type, batch, ref workSize));
			return workSize;
		}

		/// <summary>
		/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
		/// This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings.
		/// </summary>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static size_t Estimate2d(int nx, int ny, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftEstimate2d(nx, ny, type, ref workSize));
			return workSize;
		}

		/// <summary>
		/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
		/// This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings.
		/// </summary>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="nz"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static size_t Estimate3d(int nx, int ny, int nz, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftEstimate3d(nx, ny, nz, type, ref workSize));
			return workSize;
		}

		/// <summary>
		/// During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
		/// This call returns an estimate for the size of the work area required,
		/// given the specified parameters, and assuming default plan settings.
		/// </summary>
		/// <param name="rank"></param>
		/// <param name="n"></param>
		/// <param name="inembed"></param>
		/// <param name="istride"></param>
		/// <param name="idist"></param>
		/// <param name="onembed"></param>
		/// <param name="ostride"></param>
		/// <param name="odist"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
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

		/// <summary>
		/// Creates only an opaque handle, and allocates small data structures on the host.
		/// The cufftMakePlan*() calls actually do the plan generation.
		/// </summary>
		/// <returns></returns>
		public static cufftHandle Create() {
			cufftHandle handle = 0;
			CheckStatus(API.cufftCreate(ref handle));
			return handle;
		}

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimate1d(),
		/// given the specified parameters, and taking into account any plan settings that may have been made.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="nx"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
		public static size_t GetSize1d(cufftHandle handle, int nx, cufftType type, int batch) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize1d(handle, nx, type, batch, ref workSize));
			return workSize;
		}

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimate2d(),
		/// given the specified parameters, and taking into account any plan settings that may have been made.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static size_t GetSize2d(cufftHandle handle, int nx, int ny, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize2d(handle, nx, ny, type, ref workSize));
			return workSize;
		}

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimate3d(),
		/// given the specified parameters, and taking into account any plan settings that may have been made.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="nx"></param>
		/// <param name="ny"></param>
		/// <param name="nz"></param>
		/// <param name="type"></param>
		/// <returns></returns>
		public static size_t GetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize3d(handle, nx, ny, nz, type, ref workSize));
			return workSize;
		}

		/// <summary>
		/// This call gives a more accurate estimate of the work area size required for a plan than cufftEstimateSizeMany(),
		/// given the specified parameters, and taking into account any plan settings that may have been made.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="rank"></param>
		/// <param name="n"></param>
		/// <param name="inembed"></param>
		/// <param name="istride"></param>
		/// <param name="idist"></param>
		/// <param name="onembed"></param>
		/// <param name="ostride"></param>
		/// <param name="odist"></param>
		/// <param name="type"></param>
		/// <param name="batch"></param>
		/// <returns></returns>
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

		/// <summary>
		/// Once plan generation has been done, either with the original API or the extensible API,
		/// this call returns the actual size of the work area required to support the plan.
		/// Callers who choose to manage work area allocation within their application
		/// must use this call after plan generation, and after any cufftSet*() calls subsequent
		/// to plan generation, if those calls might alter the required work space size.
		/// </summary>
		/// <param name="handle"></param>
		/// <returns></returns>
		public static size_t GetSize(cufftHandle handle) {
			size_t workSize = 0;
			CheckStatus(API.cufftGetSize(handle, ref workSize));
			return workSize;
		}

		/// <summary>
		/// cufftSetWorkArea() overrides the work area pointer associated with a plan.
		/// If the work area was auto-allocated, cuFFT frees the auto-allocated space.
		/// The cufftExecute*() calls assume that the work area pointer is valid and
		/// that it points to a contiguous region in device memory that does not
		/// overlap with any other work area. If this is not the case, results are indeterminate.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="workArea"></param>
		public static void SetWorkArea(cufftHandle plan, IntPtr workArea) {
			CheckStatus(API.cufftSetWorkArea(plan, workArea));
		}

		/// <summary>
		/// cufftSetAutoAllocation() indicates that the caller intends to allocate and manage work areas
		/// for plans that have been generated. cuFFT default behavior is to allocate the work area
		/// at plan generation time.
		/// If cufftSetAutoAllocation() has been called with autoAllocate set to 0 ("false") prior
		/// to one of the cufftMakePlan*() calls, cuFFT does not allocate the work area.
		/// This is the preferred sequence for callers wishing to manage work area allocation.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="autoAllocate"></param>
		public static void SetAutoAllocation(cufftHandle plan, int autoAllocate) {
			CheckStatus(API.cufftSetAutoAllocation(plan, autoAllocate));
		}

		/// <summary>
		/// cufftExecC2C() (cufftExecZ2Z()) executes a single-precision (double-precision) complex-to-complex
		/// transform plan in the transform direction as specified by direction parameter.
		/// cuFFT uses the GPU memory pointed to by the idata parameter as input data.
		/// This function stores the Fourier coefficients in the odata array.
		/// If idata and odata are the same, this method does an in-place transform.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="idata"></param>
		/// <param name="odata"></param>
		/// <param name="direction"></param>
		public static void ExecC2C(cufftHandle plan, IntPtr idata, IntPtr odata, cufftDirection direction) {
			CheckStatus(API.cufftExecC2C(plan, idata, odata, direction));
		}

		/// <summary>
		/// cufftExecR2C() (cufftExecD2Z()) executes a single-precision (double-precision) real-to-complex,
		/// implicitly forward, cuFFT transform plan.
		/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="idata"></param>
		/// <param name="odata"></param>
		public static void ExecR2C(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecR2C(plan, idata, odata));
		}

		/// <summary>
		/// cufftExecC2R() (cufftExecZ2D()) executes a single-precision (double-precision) complex-to-real,
		/// implicitly inverse, cuFFT transform plan.
		/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="idata"></param>
		/// <param name="odata"></param>
		public static void ExecC2R(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecC2R(plan, idata, odata));
		}

		/// <summary>
		/// cufftExecC2C() (cufftExecZ2Z()) executes a single-precision (double-precision) complex-to-complex
		/// transform plan in the transform direction as specified by direction parameter.
		/// cuFFT uses the GPU memory pointed to by the idata parameter as input data.
		/// This function stores the Fourier coefficients in the odata array.
		/// If idata and odata are the same, this method does an in-place transform.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="idata"></param>
		/// <param name="odata"></param>
		/// <param name="direction"></param>
		public static void ExecZ2Z(cufftHandle plan, IntPtr idata, IntPtr odata, cufftDirection direction) {
			CheckStatus(API.cufftExecZ2Z(plan, idata, odata, direction));
		}

		/// <summary>
		/// cufftExecR2C() (cufftExecD2Z()) executes a single-precision (double-precision) real-to-complex,
		/// implicitly forward, cuFFT transform plan.
		/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="idata"></param>
		/// <param name="odata"></param>
		public static void ExecD2Z(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecD2Z(plan, idata, odata));
		}

		/// <summary>
		/// cufftExecC2R() (cufftExecZ2D()) executes a single-precision (double-precision) complex-to-real,
		/// implicitly inverse, cuFFT transform plan.
		/// cuFFT uses as input data the GPU memory pointed to by the idata parameter.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="idata"></param>
		/// <param name="odata"></param>
		public static void ExecZ2D(cufftHandle plan, IntPtr idata, IntPtr odata) {
			CheckStatus(API.cufftExecZ2D(plan, idata, odata));
		}

		/// <summary>
		/// Associates a CUDA stream with a cuFFT plan.
		/// </summary>
		/// <param name="plan"></param>
		/// <param name="stream"></param>
		public static void SetStream(cufftHandle plan, cudaStream_t stream) {
			CheckStatus(API.cufftSetStream(plan, stream));
		}

		[Obsolete("Function cufftSetCompatibilityMode was removed in version 9.1.")]
		public static void SetCompatibilityMode(cufftHandle plan, cufftCompatibility mode) {
			CheckStatus(API.cufftSetCompatibilityMode(plan, mode));
		}

		/// <summary>
		/// Frees all GPU resources associated with a cuFFT plan and destroys the internal plan data structure.
		/// This function should be called once a plan is no longer needed, to avoid wasting GPU memory.
		/// </summary>
		/// <param name="plan"></param>
		public static void Destroy(cufftHandle plan) {
			CheckStatus(API.cufftDestroy(plan));
		}

		/// <summary>
		/// Returns the version number of cuFFT.
		/// </summary>
		/// <returns></returns>
		public static int GetVersion() {
			int version = 0;
			CheckStatus(API.cufftGetVersion(ref version));
			return version;
		}

		/// <summary>
		/// Return in *value the number for the property described by type of the dynamically linked CUFFT library.
		/// </summary>
		/// <param name="type"></param>
		/// <returns></returns>
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
	/// (cuFFT) CUFFT API function return values.
	/// </summary>
	public enum cufftResult {
		/// <summary>
		/// The cuFFT operation was successful
		/// </summary>
		CUFFT_SUCCESS = 0x0,
		/// <summary>
		/// cuFFT was passed an invalid plan handle
		/// </summary>
		CUFFT_INVALID_PLAN = 0x1,
		/// <summary>
		/// cuFFT failed to allocate GPU or CPU memory
		/// </summary>
		CUFFT_ALLOC_FAILED = 0x2,
		/// <summary>
		/// No longer used
		/// </summary>
		CUFFT_INVALID_TYPE = 0x3,
		/// <summary>
		/// User specified an invalid pointer or parameter
		/// </summary>
		CUFFT_INVALID_VALUE = 0x4,
		/// <summary>
		/// Driver or internal cuFFT library error
		/// </summary>
		CUFFT_INTERNAL_ERROR = 0x5,
		/// <summary>
		/// Failed to execute an FFT on the GPU
		/// </summary>
		CUFFT_EXEC_FAILED = 0x6,
		/// <summary>
		/// The cuFFT library failed to initialize
		/// </summary>
		CUFFT_SETUP_FAILED = 0x7,
		/// <summary>
		/// User specified an invalid transform size
		/// </summary>
		CUFFT_INVALID_SIZE = 0x8,
		/// <summary>
		/// No longer used
		/// </summary>
		CUFFT_UNALIGNED_DATA = 0x9,
		/// <summary>
		/// Missing parameters in call
		/// </summary>
		CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
		/// <summary>
		/// Execution of a plan was on different GPU than plan creation
		/// </summary>
		CUFFT_INVALID_DEVICE = 0xB,
		/// <summary>
		/// Internal plan database error
		/// </summary>
		CUFFT_PARSE_ERROR = 0xC,
		/// <summary>
		/// No workspace has been provided prior to plan execution
		/// </summary>
		CUFFT_NO_WORKSPACE = 0xD,
		/// <summary>
		/// Function does not implement functionality for parameters given.
		/// </summary>
		CUFFT_NOT_IMPLEMENTED = 0xE,
		/// <summary>
		/// Used in previous versions.
		/// </summary>
		CUFFT_LICENSE_ERROR = 0x0F,
		/// <summary>
		/// Operation is not supported for parameters given.
		/// </summary>
		CUFFT_NOT_SUPPORTED = 0x10
	}

	/// <summary>
	/// (cuFFT) CUFFT supports the following transform types.
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
	/// (cuFFT) CUFFT supports the following data layouts.
	/// </summary>
	public enum cufftCompatibility {
		/// <summary>
		/// The default value
		/// </summary>
		CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01
	}

	/// <summary>
	/// (cuFFT) Transform direction.
	/// </summary>
	public enum cufftDirection {
		CUFFT_FORWARD = -1,
		CUFFT_INVERSE = 1
	}

	/// <summary>
	/// (cuFFT) The cuFFT library supports callback funtions for all combinations of single or double precision,
	/// real or complex data, load or store. These are enumerated in the parameter cufftXtCallbackType.
	/// </summary>
	public enum cufftXtCallbackType {
		CUFFT_CB_LD_COMPLEX = 0x0,
		CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
		CUFFT_CB_LD_REAL = 0x2,
		CUFFT_CB_LD_REAL_DOUBLE = 0x3,
		CUFFT_CB_ST_COMPLEX = 0x4,
		CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
		CUFFT_CB_ST_REAL = 0x6,
		CUFFT_CB_ST_REAL_DOUBLE = 0x7,
		CUFFT_CB_UNDEFINED = 0x8
	}

	/// <summary>
	/// (cuFFT) cufftXtSubFormat_t is an enumerated type that indicates if the buffer will be used
	/// for input or output and the ordering of the data.
	/// </summary>
	public enum cufftXtSubFormat {
		/// <summary>
		/// by default input is in linear order across GPUs
		/// </summary>
		CUFFT_XT_FORMAT_INPUT = 0,
		/// <summary>
		/// by default output is in scrambled order depending on transform
		/// </summary>
		CUFFT_XT_FORMAT_OUTPUT = 1,
		/// <summary>
		/// by default inplace is input order, which is linear across GPUs
		/// </summary>
		CUFFT_XT_FORMAT_INPLACE = 2,
		/// <summary>
		/// shuffled output order after execution of the transform
		/// </summary>
		CUFFT_XT_FORMAT_INPLACE_SHUFFLED = 3,
		CUFFT_FORMAT_UNDEFINED = 5
	}

	/// <summary>
	/// (cuFFT) cufftXtCopyType_t is an enumerated type for multiple GPU functions that specifies the type of copy for cufftXtMemcpy().
	/// </summary>
	public enum cufftXtCopyType {
		/// <summary>
		/// copies data from a contiguous host buffer to multiple device buffers,
		/// in the layout cuFFT requires for input data.
		/// dstPointer must point to a cudaLibXtDesc structure,
		/// and srcPointer must point to a host memory buffer.
		/// </summary>
		CUFFT_COPY_HOST_TO_DEVICE,
		/// <summary>
		/// copies data from multiple device buffers, in the layout cuFFT produces for output data,
		/// to a contiguous host buffer. dstPointer must point to a host memory buffer, and srcPointer
		/// must point to a cudaLibXtDesc structure.
		/// </summary>
		CUFFT_COPY_DEVICE_TO_HOST,
		/// <summary>
		/// copies data from multiple device buffers, in the layout cuFFT produces for output data,
		/// to multiple device buffers, in the layout cuFFT requires for input data. dstPointer
		/// and srcPointer must point to different cudaLibXtDesc structures (and therefore memory locations).
		/// That is, the copy cannot be in-place.
		/// </summary>
		CUFFT_COPY_DEVICE_TO_DEVICE
	}

	/// <summary>
	/// (cuFFT) A descriptor type used in multiple GPU routines that contains information about the GPUs and their memory locations.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaXtDesc {
		/// <summary>
		/// descriptor version
		/// </summary>
		public int version;
		/// <summary>
		/// number of GPUs
		/// </summary>
		public int nGPUs;
		/// <summary>
		/// array of device IDs
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = cuFFT.MAX_CUDA_DESCRIPTOR_GPUS)]
		public int[] GPUs;
		/// <summary>
		/// array of pointers to data, one per GPU
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = cuFFT.MAX_CUDA_DESCRIPTOR_GPUS)]
		public IntPtr[] data;
		/// <summary>
		/// array of data sizes, one per GPU
		/// </summary>
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = cuFFT.MAX_CUDA_DESCRIPTOR_GPUS)]
		public size_t[] size;
		/// <summary>
		/// opaque CUDA utility structure
		/// </summary>
		public IntPtr cudaXtState; // void*
	}

	/// <summary>
	/// (cuFFT) A descriptor type used in multiple GPU routines that contains information about the library used.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudaLibXtDesc {
		/// <summary>
		/// descriptor version
		/// </summary>
		public Int32 version;
		/// <summary>
		/// multi-GPU memory descriptor
		/// </summary>
		public IntPtr descriptor; // cudaXtDesc*
		/// <summary>
		/// which library recognizes the format
		/// </summary>
		public IntPtr library; // libFormat
		/// <summary>
		/// library specific enumerator of sub formats
		/// </summary>
		public int subFormat;
		/// <summary>
		/// library specific descriptor e.g. FFT transform plan object
		/// </summary>
		public IntPtr libDescriptor; // void*
	}
}
