using System;
using System.Runtime.InteropServices;
using CUDAnshita.Errors;

namespace CUDAnshita {
	using cublasHandle_t = IntPtr;

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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasCreate_v2(
				ref cublasHandle_t handle
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasDestroy_v2(
				cublasHandle_t handle
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetVersion_v2(
				cublasHandle_t handle,
				ref int version
			);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetStream(
				cublasHandle_t handle,
				cudaStream_t streamId
			);
			
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetStream(
				cublasHandle_t handle,
				cudaStream_t* streamId
			);
			*/

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetPointerMode(
				cublasHandle_t handle,
				ref cublasPointerMode mode
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetPointerMode(
				cublasHandle_t handle,
				cublasPointerMode mode
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetVector(
				int n,
				int elemSize,
				IntPtr x,
				int incx,
				IntPtr y,
				int incy
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetVector(
				int n,
				int elemSize,
				IntPtr x,
				int incx,
				IntPtr y,
				int incy
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetMatrix(
				int rows,
				int cols,
				int elemSize,
				IntPtr A,
				int lda,
				IntPtr B,
				int ldb
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetMatrix(
				int rows,
				int cols,
				int elemSize,
				double[] A,
				int lda,
				IntPtr B,
				int ldb
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetMatrix(
				int rows,
				int cols,
				int elemSize,
				float[] A,
				int lda,
				IntPtr B,
				int ldb
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetMatrix(
				int rows,
				int cols,
				int elemSize,
				IntPtr A,
				int lda,
				IntPtr B,
				int ldb
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetMatrix(
				int rows,
				int cols,
				int elemSize,
				IntPtr A,
				int lda,
				double[] B,
				int ldb
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetMatrix(
				int rows,
				int cols,
				int elemSize,
				IntPtr A,
				int lda,
				float[] B,
				int ldb
			);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream);
			
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream);
			
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
			
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
			
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetAtomicsMode(cublasHandlet handle, cublasAtomicsModet mode);
			
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetAtomicsMode(cublasHandlet handle, cublasAtomicsModet* mode);
			*/

			// ----- Level-1 Function

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasIcamax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
			*/

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSdot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x,
				int incx,
				IntPtr y,
				int incy,
				ref float result
			);
			

			// ----- Level-2 Function

			// ----- Level-3 Function

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSgemm(
				cublasHandle_t handle,
				cublasOperation transa,
				cublasOperation transb,
				int m,
				int n,
				int k,
				ref float alpha,
				IntPtr A,
				int lda,
				IntPtr B,
				int ldb,
				ref float beta,
				IntPtr C,
				int ldc
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasDgemm_v2(
				cublasHandle_t handle,
				cublasOperation transa,
				cublasOperation transb,
				int m,
				int n,
				int k,
				ref double alpha,
				IntPtr A,
				int lda,
				IntPtr B,
				int ldb,
				ref double beta,
				IntPtr C,
				int ldc
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasDsymm_v2(
				cublasHandle_t handle,
				cublasSideMode side,
				cublasFillMode uplo,
				int m,
				int n,
				ref double alpha,
				IntPtr A,
				int lda,
				IntPtr B,
				int ldb,
				ref double beta,
				IntPtr C,
				int ldc
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasDtrmm_v2(
				cublasHandle_t handle,
				cublasSideMode side,
				cublasFillMode uplo,
				cublasOperation trans,
				cublasDiagType diag,
				int m,
				int n,
				ref double alpha,
				IntPtr A,
				int lda,
				IntPtr B,
				int ldb,
				IntPtr C,
				int ldc
			);
			

			// ----- BLAS-extension

			// ----- Legacy API

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasInit();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasShutdown();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetError();

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetVersion(
				ref int version
			);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasAlloc(int n, int elemSize, void** devicePtr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasFree(void* devicePtr);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetKernelStream(cudaStream_t stream);
			*/
		}

		// ----- C# Interface

		IntPtr handle = IntPtr.Zero;

		public cuBLAS() {
			//CheckStatus(API.cublasCreate_v2(ref handle));
		}

		~cuBLAS() {
			if (handle != IntPtr.Zero) {
				CheckStatus(API.cublasDestroy_v2(handle));
				handle = IntPtr.Zero;
			}
		}

		public void SetVector<T>(T[] x, int incx, IntPtr y, int incy) {
			int n = x.Length;
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * n;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(x, 0, ptr, x.Length);

			CheckStatus(API.cublasSetVector(n, elemSize, ptr, incx, y, incy));
			Marshal.FreeHGlobal(ptr);
		}

		public T[] GetVector<T>(int n, IntPtr x, int incx, int incy) {
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * n;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);

			CheckStatus(API.cublasGetVector(n, elemSize, x, incx, ptr, incy));

			T[] result = new T[n];
			MarshalUtil.Copy<T>(ptr, result, 0, n);
			Marshal.FreeHGlobal(ptr);
			return result;
		}

		public void SetMatrix<T>(int rows, int cols, T[] A, int lda, IntPtr B, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * rows * cols;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(A, 0, ptr, A.Length);

			CheckStatus(API.cublasSetMatrix(rows, cols, elemSize, ptr, lda, B, ldb));
			Marshal.FreeHGlobal(ptr);
		}

		public void SetMatrix(int rows, int cols, double[] A, int lda, IntPtr B, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(double));
			CheckStatus(API.cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}

		public void SetMatrix(int rows, int cols, float[] A, int lda, IntPtr B, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(float));
			CheckStatus(API.cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}

		public T[] GetMatrix<T>(int rows, int cols, IntPtr A, int lda, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * rows * cols;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);

			CheckStatus(API.cublasGetMatrix(rows, cols, elemSize, A, lda, ptr, ldb));

			T[] result = new T[rows * cols];
			MarshalUtil.Copy<T>(ptr, result, 0, rows * cols);
			Marshal.FreeHGlobal(ptr);
			return result;
		}

		public double[] GetMatrixD(int rows, int cols, IntPtr A, int lda, int ldb) {
			double[] result = new double[rows * cols];
			int elemSize = Marshal.SizeOf(typeof(double));
			CheckStatus(API.cublasGetMatrix(rows, cols, elemSize, A, lda, result, ldb));
			return result;
		}

		public float[] GetMatrixF(int rows, int cols, IntPtr A, int lda, int ldb) {
			float[] result = new float[rows * cols];
			int elemSize = Marshal.SizeOf(typeof(double));
			CheckStatus(API.cublasGetMatrix(rows, cols, elemSize, A, lda, result, ldb));
			return result;
		}

		public float Sdot(int n, IntPtr x, int incx, IntPtr y, int incy) {
			float result = 0;
			CheckStatus(API.cublasSdot_v2(handle, n, x, incx, y, incy, ref result));
			return result;
		}

		public void Dgemm(cublasOperation transa, cublasOperation transb, int m, int n, int k, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			IntPtr handle = CreateHandle();
			CheckStatus(API.cublasDgemm_v2(handle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
			FreeHandle(handle);
		}

		public void Dsymm(cublasSideMode side, cublasFillMode uplo, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			IntPtr handle = CreateHandle();
			CheckStatus(API.cublasDsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
			FreeHandle(handle);
		}

		public void Dtrmm(cublasSideMode side, cublasFillMode uplo, cublasOperation trans, cublasDiagType diag, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			IntPtr handle = CreateHandle();
			CheckStatus(API.cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
			FreeHandle(handle);
		}


		IntPtr CreateHandle() {
			IntPtr handle = IntPtr.Zero;
			CheckStatus(API.cublasCreate_v2(ref handle));
			return handle;
		}

		void FreeHandle(IntPtr handle) {
			CheckStatus(API.cublasDestroy_v2(handle));
		}

		public static int GetVersion() {
			IntPtr handle = IntPtr.Zero;
			int version = 0;
			CheckStatus(API.cublasCreate_v2(ref handle));
			CheckStatus(API.cublasGetVersion_v2(handle, ref version));
			CheckStatus(API.cublasDestroy_v2(handle));
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
