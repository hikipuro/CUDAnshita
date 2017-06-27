using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	/// <summary>
	/// The cuBLAS library is an implementation of BLAS (Basic Linear Algebra Subprograms).
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cublas/">http://docs.nvidia.com/cuda/cublas/</a> 
	/// </remarks>
	public class cuBLAS {
		public class API {
			const string DLL_PATH = "cublas64_80.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			// ----- Helper Function

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, EntryPoint = "cublasCreate_v2")]
			public static extern cublasStatus cublasCreate(ref IntPtr handle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, EntryPoint = "cublasDestroy_v2")]
			public static extern cublasStatus cublasDestroy(IntPtr handle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, EntryPoint = "cublasGetVersion_v2")]
			public static extern cublasStatus cublasGetVersion(IntPtr handle, ref int version);

			/*
			cublasStatus cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
			cublasStatus cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId);
			cublasStatus cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode);
			cublasStatus cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);
			cublasStatus cublasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);
			cublasStatus cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);
			cublasStatus cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);
			cublasStatus cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);
			cublasStatus cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream);
			cublasStatus cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream);
			cublasStatus cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
			cublasStatus cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
			cublasStatus cublasSetAtomicsMode(cublasHandlet handle, cublasAtomicsModet mode);
			cublasStatus cublasGetAtomicsMode(cublasHandlet handle, cublasAtomicsModet* mode);
			*/

			// ----- Level-1 Function

			// ----- Level-2 Function

			// ----- Level-3 Function

			// ----- BLAS-extension

			// ----- Legacy API

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasInit();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasShutdown();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetError();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetVersion(ref int version);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasAlloc(int n, int elemSize, void** devicePtr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasFree(void* devicePtr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetKernelStream(cudaStream_t stream);
			*/
		}

		public static int GetVersion() {
			IntPtr handle = IntPtr.Zero;
			int version = 0;
			CheckStatus(API.cublasCreate(ref handle));
			CheckStatus(API.cublasGetVersion(handle, ref version));
			CheckStatus(API.cublasDestroy(handle));
			return version;
		}

		static void CheckStatus(cublasStatus status) {
			if (status != cublasStatus.CUBLAS_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}
		}
	}

	/// <summary>
	/// The type is used for function status returns.
	/// All cuBLAS library functions return their status.
	/// </summary>
	public enum cublasStatus {
		///<summary>The operation completed successfully.</summary>
		CUBLAS_STATUS_SUCCESS = 0,
		///<summary>The cuBLAS library was not initialized.</summary>
		CUBLAS_STATUS_NOT_INITIALIZED = 1,
		///<summary>Resource allocation failed inside the cuBLAS library.</summary>
		CUBLAS_STATUS_ALLOC_FAILED = 3,
		///<summary>An unsupported value or parameter was passed to the function (a negative vector size, for example).</summary>
		CUBLAS_STATUS_INVALID_VALUE = 7,
		///<summary>The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.</summary>
		CUBLAS_STATUS_ARCH_MISMATCH = 8,
		///<summary>An access to GPU memory space failed, which is usually caused by a failure to bind a texture.</summary>
		CUBLAS_STATUS_MAPPING_ERROR = 11,
		///<summary>The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.</summary>
		CUBLAS_STATUS_EXECUTION_FAILED = 13,
		///<summary>An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.</summary>
		CUBLAS_STATUS_INTERNAL_ERROR = 14,
		///<summary>The functionnality requested is not supported</summary>
		CUBLAS_STATUS_NOT_SUPPORTED = 15,
		///<summary>The functionnality requested requires some license and an error was detected when trying to check the current licensing.</summary>
		CUBLAS_STATUS_LICENSE_ERROR = 16
	}

	/// <summary>
	/// The cublasOperation_t type indicates which operation needs to be performed with the dense matrix.
	/// Its values correspond to Fortran characters ‘N’ or ‘n’ (non-transpose), ‘T’ or ‘t’ (transpose) and ‘C’ or ‘c’ (conjugate transpose) that are often used as parameters to legacy BLAS implementations.
	/// </summary>
	public enum cublasOperation {
		///<summary>the non-transpose operation is selected</summary>
		CUBLAS_OP_N = 0,
		///<summary>the transpose operation is selected</summary>
		CUBLAS_OP_T = 1,
		///<summary>the conjugate transpose operation is selected</summary>
		CUBLAS_OP_C = 2
	}

	/// <summary>
	/// The type indicates which part (lower or upper) of the dense matrix was filled and consequently should be used by the function.
	/// Its values correspond to Fortran characters ‘L’ or ‘l’ (lower) and ‘U’ or ‘u’ (upper) that are often used as parameters to legacy BLAS implementations.
	/// </summary>
	public enum cublasFillMode {
		///<summary>the lower part of the matrix is filled</summary>
		CUBLAS_FILL_MODE_LOWER = 0,
		///<summary>the upper part of the matrix is filled</summary>
		CUBLAS_FILL_MODE_UPPER = 1
	}

	/// <summary>
	/// The type indicates whether the main diagonal of the dense matrix is unity and consequently should not be touched or modified by the function.
	/// Its values correspond to Fortran characters ‘N’ or ‘n’ (non-unit) and ‘U’ or ‘u’ (unit) that are often used as parameters to legacy BLAS implementations.
	/// </summary>
	public enum cublasDiagType {
		///<summary>the matrix diagonal has non-unit elements</summary>
		CUBLAS_DIAG_NON_UNIT = 0,
		///<summary>the matrix diagonal has unit elements</summary>
		CUBLAS_DIAG_UNIT = 1
	}

	/// <summary>
	/// The type indicates whether the dense matrix is on the left or right side in the matrix equation solved by a particular function.
	/// Its values correspond to Fortran characters ‘L’ or ‘l’ (left) and ‘R’ or ‘r’ (right) that are often used as parameters to legacy BLAS implementations.
	/// </summary>
	public enum cublasSideMode {
		///<summary>the matrix is on the left side in the equation</summary>
		CUBLAS_SIDE_LEFT = 0,
		///<summary>the matrix is on the right side in the equation</summary>
		CUBLAS_SIDE_RIGHT = 1
	}

	/// <summary>
	/// The cublasPointerMode_t type indicates whether the scalar values are passed by reference on the host or device.
	/// It is important to point out that if several scalar values are present in the function call, all of them must conform to the same single pointer mode.
	/// The pointer mode can be set and retrieved using cublasSetPointerMode() and cublasGetPointerMode() routines, respectively.
	/// </summary>
	public enum cublasPointerMode {
		///<summary>the scalars are passed by reference on the host</summary>
		CUBLAS_POINTER_MODE_HOST = 0,
		///<summary>the scalars are passed by reference on the device</summary>
		CUBLAS_POINTER_MODE_DEVICE = 1
	}

	/// <summary>
	/// The type indicates whether cuBLAS routines which has an alternate implementation using atomics can be used.
	/// The atomics mode can be set and queried using and routines, respectively.
	/// </summary>
	public enum cublasAtomicsMode {
		///<summary>the usage of atomics is not allowed</summary>
		CUBLAS_ATOMICS_NOT_ALLOWED = 0,
		///<summary>the usage of atomics is allowed</summary>
		CUBLAS_ATOMICS_ALLOWED = 1
	}

	/// <summary>
	/// cublasGemmAlgo_t type is an enumerant to specify the algorithm for matrix-matrix multiplication.
	/// It is used to run cublasGemmEx routine with specific algorithm. CUBLAS has the following algorithm options.
	/// </summary>
	public enum cublasGemmAlgo {
		///<summary>Default algorithm of cublas</summary>
		CUBLAS_GEMM_DFALT = -1,
		///<summary>Algorithm 0</summary>
		CUBLAS_GEMM_ALGO0 = 0,
		///<summary>Algorithm 1</summary>
		CUBLAS_GEMM_ALGO1 = 1,
		///<summary>Algorithm 2</summary>
		CUBLAS_GEMM_ALGO2 = 2,
		///<summary>Algorithm 3</summary>
		CUBLAS_GEMM_ALGO3 = 3,
		///<summary>Algorithm 4</summary>
		CUBLAS_GEMM_ALGO4 = 4,
		///<summary>Algorithm 5</summary>
		CUBLAS_GEMM_ALGO5 = 5,
		///<summary>Algorithm 6</summary>
		CUBLAS_GEMM_ALGO6 = 6,
		///<summary>Algorithm 7</summary>
		CUBLAS_GEMM_ALGO7 = 7
	}
}
