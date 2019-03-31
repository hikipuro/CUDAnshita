using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using cublasStatus_t = cublasStatus;
	using cublasHandle_t = IntPtr;
	using cudaStream_t = IntPtr;
	using cublasPointerMode_t = cublasPointerMode;
	using cublasAtomicsMode_t = cublasAtomicsMode;
	using cuComplex = float2;
	using cuDoubleComplex = double2;
	using cublasOperation_t = cublasOperation;
	using cublasFillMode_t = cublasFillMode;
	using cublasSideMode_t = cublasSideMode;
	using cublasDiagType_t = cublasDiagType;
	using cublasGemmAlgo_t = cublasGemmAlgo;
	using cublasMath_t = cublasMath;

	public partial class cuBLAS {
		// ----- C# Interface

		/// <summary>
		/// This function initializes the CUBLAS library and creates a handle to an opaque structure holding the CUBLAS library context.
		/// </summary>
		/// <returns></returns>
		public static cublasHandle_t Create_v2() {
			cublasHandle_t handle = IntPtr.Zero;
			CheckStatus(API.cublasCreate_v2(ref handle));
			return handle;
		}

		/// <summary>
		/// This function releases hardware resources used by the CUBLAS library.
		/// </summary>
		/// <param name="handle"></param>
		public static void Destroy_v2(cublasHandle_t handle) {
			CheckStatus(API.cublasDestroy_v2(handle));
		}

		/// <summary>
		/// This function returns the version number of the cuBLAS library.
		/// </summary>
		/// <param name="handle"></param>
		/// <returns></returns>
		public static int GetVersion_v2(cublasHandle_t handle) {
			int version = 0;
			CheckStatus(API.cublasGetVersion_v2(handle, ref version));
			return version;
		}

		/// <summary>
		/// This function returns the value of the requested property in memory pointed to by value.
		/// Refer to libraryPropertyType for supported types.
		/// </summary>
		/// <param name="type"></param>
		/// <returns></returns>
		public static int GetProperty(libraryPropertyType type) {
			int value = 0;
			CheckStatus(API.cublasGetProperty(type, ref value));
			return value;
		}

		/// <summary>
		/// This function sets the cuBLAS library stream, which will be used to execute all subsequent calls to the cuBLAS library functions.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="streamId"></param>
		public static void SetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
			CheckStatus(API.cublasSetStream_v2(handle, streamId));
		}

		/// <summary>
		/// This function gets the cuBLAS library stream, which is being used to execute all calls to the cuBLAS library functions.
		/// </summary>
		/// <param name="handle"></param>
		/// <returns></returns>
		public static cudaStream_t GetStream_v2(cublasHandle_t handle) {
			cudaStream_t streamId = IntPtr.Zero;
			CheckStatus(API.cublasGetStream_v2(handle, ref streamId));
			return streamId;
		}

		/// <summary>
		/// This function obtains the pointer mode used by the cuBLAS library.
		/// Please see the section on the cublasPointerMode_t type for more details.
		/// </summary>
		/// <param name="handle"></param>
		/// <returns></returns>
		public static cublasPointerMode_t GetPointerMode_v2(cublasHandle_t handle) {
			cublasPointerMode_t mode = cublasPointerMode_t.CUBLAS_POINTER_MODE_HOST;
			CheckStatus(API.cublasGetPointerMode_v2(handle, ref mode));
			return mode;
		}

		/// <summary>
		/// This function sets the pointer mode used by the cuBLAS library.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		public static void SetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
			CheckStatus(API.cublasSetPointerMode_v2(handle, mode));
		}

		/// <summary>
		/// This function queries the atomic mode of a specific cuBLAS context.
		/// </summary>
		/// <param name="handle"></param>
		/// <returns></returns>
		public static cublasAtomicsMode_t GetAtomicsMode(cublasHandle_t handle) {
			cublasAtomicsMode_t mode = cublasAtomicsMode_t.CUBLAS_ATOMICS_NOT_ALLOWED;
			CheckStatus(API.cublasGetAtomicsMode(handle, ref mode));
			return mode;
		}

		public static void SetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
			CheckStatus(API.cublasSetAtomicsMode(handle, mode));
		}

		/// <summary>
		/// This function copies n elements from a vector x in host memory space to a vector y in GPU memory space.
		/// </summary>
		/// <param name="n">element count</param>
		/// <param name="elemSize"></param>
		/// <param name="x">host memory pointer</param>
		/// <param name="incx"></param>
		/// <param name="y">device memory pointer</param>
		/// <param name="incy"></param>
		public static void SetVector(int n, int elemSize, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasSetVector(n, elemSize, x, incx, y, incy));
		}

		/// <summary>
		/// This function copies n elements from a vector x in host memory space to a vector y in GPU memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void SetVector<T>(T[] x, int incx, IntPtr y, int incy) {
			int n = x.Length;
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * n;
			IntPtr xPointer = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(x, 0, xPointer, x.Length);
			SetVector(n, elemSize, xPointer, incx, y, incy);
			Marshal.FreeHGlobal(xPointer);
		}

		/// <summary>
		/// This function copies n elements from a vector x in host memory space to a vector y in GPU memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="x"></param>
		/// <param name="y"></param>
		public static void SetVector<T>(T[] x, IntPtr y) {
			SetVector<T>(x, 1, y, 1);
		}

		/// <summary>
		/// This function copies n elements from a vector x in GPU memory space to a vector y in host memory space.
		/// </summary>
		/// <param name="n"></param>
		/// <param name="elemSize"></param>
		/// <param name="x">device memory pointer</param>
		/// <param name="incx"></param>
		/// <param name="y">host memory pointer</param>
		/// <param name="incy"></param>
		public static void GetVector(int n, int elemSize, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasGetVector(n, elemSize, x, incx, y, incy));
		}

		/// <summary>
		/// This function copies n elements from a vector x in GPU memory space to a vector y in host memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="incy"></param>
		/// <returns></returns>
		public static T[] GetVector<T>(int n, IntPtr x, int incx, int incy) {
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * n;
			IntPtr yPointer = Marshal.AllocHGlobal(byteSize);
			GetVector(n, elemSize, x, incx, yPointer, incy);

			T[] result = new T[n];
			MarshalUtil.Copy<T>(yPointer, result, 0, n);
			Marshal.FreeHGlobal(yPointer);
			return result;
		}

		/// <summary>
		/// This function copies n elements from a vector x in GPU memory space to a vector y in host memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <returns></returns>
		public static T[] GetVector<T>(int n, IntPtr x) {
			return GetVector<T>(n, x, 1, 1);
		}

		/// <summary>
		/// This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space.
		/// </summary>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="elemSize"></param>
		/// <param name="A">host memory pointer</param>
		/// <param name="lda"></param>
		/// <param name="B">device memory pointer</param>
		/// <param name="ldb"></param>
		public static void SetMatrix(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}

		/// <summary>
		/// This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		public static void SetMatrix<T>(int rows, int cols, T[] A, int lda, IntPtr B, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * rows * cols;
			IntPtr APointer = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(A, 0, APointer, A.Length);
			SetMatrix(rows, cols, elemSize, APointer, lda, B, ldb);
			Marshal.FreeHGlobal(APointer);
		}

		/// <summary>
		/// This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="A"></param>
		/// <param name="B"></param>
		public static void SetMatrix<T>(int rows, int cols, T[] A, IntPtr B) {
			SetMatrix<T>(rows, cols, A, rows, B, rows);
		}

		/// <summary>
		/// This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space.
		/// </summary>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="elemSize"></param>
		/// <param name="A">device memory pointer</param>
		/// <param name="lda"></param>
		/// <param name="B">host memory pointer</param>
		/// <param name="ldb"></param>
		public static void GetMatrix(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}

		/// <summary>
		/// This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="ldb"></param>
		/// <returns></returns>
		public static T[] GetMatrix<T>(int rows, int cols, IntPtr A, int lda, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * rows * cols;
			IntPtr BPointer = Marshal.AllocHGlobal(byteSize);
			GetMatrix(rows, cols, elemSize, A, lda, BPointer, ldb);
			T[] result = new T[rows * cols];
			MarshalUtil.Copy<T>(BPointer, result, 0, rows * cols);
			Marshal.FreeHGlobal(BPointer);
			return result;
		}

		/// <summary>
		/// This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="A"></param>
		/// <returns></returns>
		public static T[] GetMatrix<T>(int rows, int cols, IntPtr A) {
			return GetMatrix<T>(rows, cols, A, rows, rows);
		}

		/// <summary>
		/// This function has the same functionality as cublasSetVector(),
		/// with the exception that the data transfer is done asynchronously
		/// (with respect to the host) using the given CUDA™ stream parameter.
		/// </summary>
		/// <param name="n"></param>
		/// <param name="elemSize"></param>
		/// <param name="hostPtr"></param>
		/// <param name="incx"></param>
		/// <param name="devicePtr"></param>
		/// <param name="incy"></param>
		/// <param name="stream"></param>
		public static void SetVectorAsync(int n, int elemSize, IntPtr hostPtr, int incx, IntPtr devicePtr, int incy, cudaStream_t stream) {
			CheckStatus(API.cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream));
		}

		/// <summary>
		/// This function has the same functionality as cublasGetVector(),
		/// with the exception that the data transfer is done asynchronously
		/// (with respect to the host) using the given CUDA™ stream parameter.
		/// </summary>
		/// <param name="n"></param>
		/// <param name="elemSize"></param>
		/// <param name="devicePtr"></param>
		/// <param name="incx"></param>
		/// <param name="hostPtr"></param>
		/// <param name="incy"></param>
		/// <param name="stream"></param>
		public static void GetVectorAsync(int n, int elemSize, IntPtr devicePtr, int incx, IntPtr hostPtr, int incy, cudaStream_t stream) {
			CheckStatus(API.cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream));
		}

		/// <summary>
		/// This function has the same functionality as cublasSetMatrix(),
		/// with the exception that the data transfer is done asynchronously
		/// (with respect to the host) using the given CUDA™ stream parameter.
		/// </summary>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="elemSize"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="stream"></param>
		public static void SetMatrixAsync(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb, cudaStream_t stream) {
			CheckStatus(API.cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
		}

		/// <summary>
		/// This function has the same functionality as cublasGetMatrix(),
		/// with the exception that the data transfer is done asynchronously
		/// (with respect to the host) using the given CUDA™ stream parameter.
		/// </summary>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="elemSize"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="stream"></param>
		public static void GetMatrixAsync(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb, cudaStream_t stream) {
			CheckStatus(API.cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="srName"></param>
		/// <param name="info"></param>
		public static void Xerbla(string srName, int info) {
			API.cublasXerbla(srName, info);
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x">device memory pointer</param>
		/// <param name="xType"></param>
		/// <param name="incx"></param>
		/// <param name="result">host or device</param>
		/// <param name="resultType"></param>
		/// <param name="executionType"></param>
		public static void Nrm2Ex(cublasHandle_t handle, int n, IntPtr x, cudaDataType xType, int incx, IntPtr result, cudaDataType resultType, cudaDataType executionType) {
			CheckStatus(API.cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType));
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static float Snrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0f;
			CheckStatus(API.cublasSnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public static void Snrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result) {
			cublasPointerMode mode = GetPointerMode_v2(handle);
			SetPointerMode_v2(handle, cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE);
			CheckStatus(API.cublasSnrm2_v2(handle, n, x, incx, result));
			SetPointerMode_v2(handle, mode);
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static double Dnrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0f;
			CheckStatus(API.cublasDnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static float Scnrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0f;
			CheckStatus(API.cublasScnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static double Dznrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0f;
			CheckStatus(API.cublasDznrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="xType"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="yType"></param>
		/// <param name="incy"></param>
		/// <param name="resultType"></param>
		/// <param name="executionType"></param>
		/// <returns></returns>
		public static IntPtr DotEx(cublasHandle_t handle, int n, IntPtr x, cudaDataType xType, int incx, IntPtr y, cudaDataType yType, int incy, cudaDataType resultType, cudaDataType executionType) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasDotEx(
				handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType
			));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="xType"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="yType"></param>
		/// <param name="incy"></param>
		/// <param name="resultType"></param>
		/// <param name="executionType"></param>
		/// <returns></returns>
		public static IntPtr DotcEx(cublasHandle_t handle, int n, IntPtr x, cudaDataType xType, int incx, IntPtr y, cudaDataType yType, int incy, cudaDataType resultType, cudaDataType executionType) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasDotcEx(
				handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType
			));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <returns></returns>
		public static IntPtr Sdot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasSdot_v2(
				handle, n, x, incx, y, incy, result
			));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <returns></returns>
		public static IntPtr Ddot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasDdot_v2(
				handle, n, x, incx, y, incy, result
			));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <returns></returns>
		public static cuComplex Cdotu_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuComplex result = new cuComplex();
			CheckStatus(API.cublasCdotu_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <returns></returns>
		public static cuComplex Cdotc_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuComplex result = new cuComplex();
			CheckStatus(API.cublasCdotc_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <returns></returns>
		public static cuDoubleComplex Zdotu_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuDoubleComplex result = new cuDoubleComplex();
			CheckStatus(API.cublasZdotu_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <returns></returns>
		public static cuDoubleComplex Zdotc_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuDoubleComplex result = new cuDoubleComplex();
			CheckStatus(API.cublasZdotc_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}

		/// <summary>
		/// This function scales the vector x by the scalar α and overwrites it with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="alphaType"></param>
		/// <param name="x"></param>
		/// <param name="xType"></param>
		/// <param name="incx"></param>
		/// <param name="executionType"></param>
		public static void ScalEx(cublasHandle_t handle, int n, IntPtr alpha, cudaDataType alphaType, IntPtr x, cudaDataType xType, int incx, cudaDataType executionType) {
			CheckStatus(API.cublasScalEx(
				handle, n, alpha, alphaType, x, xType, incx, executionType
			));
		}

		/// <summary>
		/// This function scales the vector x by the scalar α and overwrites it with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Sscal_v2(cublasHandle_t handle, int n, float alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasSscal_v2(handle, n, ref alpha, x, incx));
		}

		/// <summary>
		/// This function scales the vector x by the scalar α and overwrites it with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Dscal_v2(cublasHandle_t handle, int n, double alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasDscal_v2(handle, n, ref alpha, x, incx));
		}

		/// <summary>
		/// This function scales the vector x by the scalar α and overwrites it with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Cscal_v2(cublasHandle_t handle, int n, cuComplex alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasCscal_v2(handle, n, ref alpha, x, incx));
		}

		/// <summary>
		/// This function scales the vector x by the scalar α and overwrites it with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Csscal_v2(cublasHandle_t handle, int n, float alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasCsscal_v2(handle, n, ref alpha, x, incx));
		}

		/// <summary>
		/// This function scales the vector x by the scalar α and overwrites it with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Zscal_v2(cublasHandle_t handle, int n, cuDoubleComplex alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasZscal_v2(handle, n, ref alpha, x, incx));
		}

		/// <summary>
		/// This function scales the vector x by the scalar α and overwrites it with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Zdscal_v2(cublasHandle_t handle, int n, double alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasZdscal_v2(handle, n, ref alpha, x, incx));
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="alphaType"></param>
		/// <param name="x"></param>
		/// <param name="xType"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="yType"></param>
		/// <param name="incy"></param>
		/// <param name="executiontype"></param>
		public static void AxpyEx(cublasHandle_t handle, int n, IntPtr alpha, cudaDataType alphaType, IntPtr x, cudaDataType xType, int incx, IntPtr y, cudaDataType yType, int incy, cudaDataType executiontype) {
			CheckStatus(API.cublasAxpyEx(
				handle,
				n,
				alpha,
				alphaType,
				x,
				xType,
				incx,
				y,
				yType,
				incy,
				executiontype
			));
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Saxpy_v2(cublasHandle_t handle, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasSaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Daxpy_v2(cublasHandle_t handle, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasDaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Caxpy_v2(cublasHandle_t handle, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasCaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zaxpy_v2(cublasHandle_t handle, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasZaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}

		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Scopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasScopy_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Dcopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasDcopy_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Ccopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasCcopy_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zcopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasZcopy_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Sswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasSswap_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Dswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasDswap_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Cswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasCswap_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasZswap_v2(handle, n, x, incx, y, incy));
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Isamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIsamax_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Idamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIdamax_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Icamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIcamax_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Izamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIzamax_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Isamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIsamin_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Idamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIdamin_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Icamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIcamin_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static int Izamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIzamin_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static float Sasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0;
			CheckStatus(API.cublasSasum_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static double Dasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0;
			CheckStatus(API.cublasDasum_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static float Scasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0;
			CheckStatus(API.cublasScasum_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <returns></returns>
		public static double Dzasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0;
			CheckStatus(API.cublasDzasum_v2(handle, n, x, incx, ref result));
			return result;
		}

		/// <summary>
		/// This function applies Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Srot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] c, float[] s) {
			CheckStatus(API.cublasSrot_v2(handle, n, x, incx, y, incy, c, s));
		}

		/// <summary>
		/// This function applies Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Drot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] c, double[] s) {
			CheckStatus(API.cublasDrot_v2(handle, n, x, incx, y, incy, c, s));
		}

		/// <summary>
		/// This function applies Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Crot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] c, cuComplex[] s) {
			CheckStatus(API.cublasCrot_v2(handle, n, x, incx, y, incy, c, s));
		}

		/// <summary>
		/// This function applies Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Csrot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] c, float[] s) {
			CheckStatus(API.cublasCsrot_v2(handle, n, x, incx, y, incy, c, s));
		}

		/// <summary>
		/// This function applies Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Zrot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] c, cuDoubleComplex[] s) {
			CheckStatus(API.cublasZrot_v2(handle, n, x, incx, y, incy, c, s));
		}

		/// <summary>
		/// This function applies Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Zdrot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] c, double[] s) {
			CheckStatus(API.cublasZdrot_v2(handle, n, x, incx, y, incy, c, s));
		}

		/// <summary>
		/// This function constructs the Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Srotg_v2(cublasHandle_t handle, float a, float b, float c, float s) {
			CheckStatus(API.cublasSrotg_v2(handle, ref a, ref b, ref c, ref s));
		}

		/// <summary>
		/// This function constructs the Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Drotg_v2(cublasHandle_t handle, double a, double b, double c, double s) {
			CheckStatus(API.cublasDrotg_v2(handle, ref a, ref b, ref c, ref s));
		}

		/// <summary>
		/// This function constructs the Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Crotg_v2(cublasHandle_t handle, cuComplex a, cuComplex b, float c, cuComplex s) {
			CheckStatus(API.cublasCrotg_v2(handle, ref a, ref b, ref c, ref s));
		}

		/// <summary>
		/// This function constructs the Givens rotation matrix.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c"></param>
		/// <param name="s"></param>
		public static void Zrotg_v2(cublasHandle_t handle, cuDoubleComplex a, cuDoubleComplex b, double c, cuDoubleComplex s) {
			CheckStatus(API.cublasZrotg_v2(handle, ref a, ref b, ref c, ref s));
		}

		/// <summary>
		/// This function applies the modified Givens transformation.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="param"></param>
		public static void Srotm_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] param) {
			CheckStatus(API.cublasSrotm_v2(handle, n, x, incx, y, incy, param));
		}

		/// <summary>
		/// This function applies the modified Givens transformation.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="param"></param>
		public static void Drotm_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] param) {
			CheckStatus(API.cublasDrotm_v2(handle, n, x, incx, y, incy, param));
		}

		/// <summary>
		/// This function constructs the modified Givens transformation.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="d1"></param>
		/// <param name="d2"></param>
		/// <param name="x1"></param>
		/// <param name="y1"></param>
		/// <param name="param"></param>
		public static void Srotmg_v2(cublasHandle_t handle, float d1, float d2, float x1, float y1, float[] param) {
			CheckStatus(API.cublasSrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param));
		}

		/// <summary>
		/// This function constructs the modified Givens transformation.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="d1"></param>
		/// <param name="d2"></param>
		/// <param name="x1"></param>
		/// <param name="y1"></param>
		/// <param name="param"></param>
		public static void Drotmg_v2(cublasHandle_t handle, double d1, double d2, double x1, double y1, double[] param) {
			CheckStatus(API.cublasDrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param));
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Sgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float alpha, IntPtr A, int lda, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Dgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double alpha, IntPtr A, int lda, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Cgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasCgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="kl"></param>
		/// <param name="ku"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Sgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, float alpha, IntPtr A, int lda, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSgbmv_v2(
				handle, trans,
				m, n,
				kl, ku,
				ref alpha,
				A, lda,
				x, incx,
				ref beta,
				y, incy
			));
		}

		/// <summary>
		/// This function performs the banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="kl"></param>
		/// <param name="ku"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Dgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, double alpha, IntPtr A, int lda, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDgbmv_v2(
				handle, trans,
				m, n,
				kl, ku,
				ref alpha,
				A, lda,
				x, incx,
				ref beta,
				y, incy
			));
		}

		/// <summary>
		/// This function performs the banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="kl"></param>
		/// <param name="ku"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Cgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasCgbmv_v2(
				handle, trans,
				m, n,
				kl, ku,
				ref alpha,
				A, lda,
				x, incx,
				ref beta,
				y, incy
			));
		}

		/// <summary>
		/// This function performs the banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="kl"></param>
		/// <param name="ku"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZgbmv_v2(
				handle, trans,
				m, n,
				kl, ku,
				ref alpha,
				A, lda,
				x, incx,
				ref beta,
				y, incy
			));
		}

		/// <summary>
		/// This function performs the triangular matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Strmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Dtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ctrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ztrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Stbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Dtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ctbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ztbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Stpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Dtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ctpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ztpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Strsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Dtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ctrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ztrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Stpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Dtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ctpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ztpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Stbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Dtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ctbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public static void Ztbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Ssymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr A, int lda, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Dsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr A, int lda, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Csymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasCsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Chemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasChemv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZhemv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Ssbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, float alpha, IntPtr A, int lda, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSsbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Dsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, double alpha, IntPtr A, int lda, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDsbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Chbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasChbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZhbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Sspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr AP, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSspmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Dspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr AP, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDspmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Chpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr AP, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasChpmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="AP"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="beta"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public static void Zhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr AP, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZhpmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}

		/// <summary>
		/// This function performs the rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Sger_v2(cublasHandle_t handle, int m, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasSger_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Dger_v2(cublasHandle_t handle, int m, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasDger_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Cgeru_v2(cublasHandle_t handle, int m, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCgeru_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Cgerc_v2(cublasHandle_t handle, int m, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCgerc_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Zgeru_v2(cublasHandle_t handle, int m, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZgeru_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Zgerc_v2(cublasHandle_t handle, int m, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZgerc_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Ssyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasSsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Dsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasDsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Csyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasCsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Zsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasZsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Cher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasCher_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Zher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasZher_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		/// <summary>
		/// This function performs the packed symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="AP"></param>
		public static void Sspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasSspr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}

		/// <summary>
		/// This function performs the packed symmetric rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="AP"></param>
		public static void Dspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasDspr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}

		/// <summary>
		/// This function performs the packed Hermitian rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="AP"></param>
		public static void Chpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasChpr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}

		/// <summary>
		/// This function performs the packed Hermitian rank-1 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="AP"></param>
		public static void Zhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasZhpr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Ssyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasSsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Dsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasDsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Csyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Zsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the Hermitian rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Cher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCher2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the Hermitian rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Zher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZher2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		/// <summary>
		/// This function performs the packed symmetric rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="AP"></param>
		public static void Sspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasSspr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}

		/// <summary>
		/// This function performs the packed symmetric rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="AP"></param>
		public static void Dspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasDspr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}

		/// <summary>
		/// This function performs the packed Hermitian rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="AP"></param>
		public static void Chpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasChpr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}

		/// <summary>
		/// This function performs the packed Hermitian rank-2 update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="AP"></param>
		public static void Zhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasZhpr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Sgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSgemm_v2(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda,
				B, ldb,
				ref beta,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Dgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDgemm_v2(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda,
				B, ldb,
				ref beta,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Cgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCgemm_v2(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda,
				B, ldb,
				ref beta,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the complex matrix-matrix multiplication, using Gauss complexity reduction algorithm.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Cgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCgemm3m(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda,
				B, ldb,
				ref beta,
				C, ldc
			));
		}
		// 
		// cublasCgemm3mEx

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZgemm_v2(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda,
				B, ldb,
				ref beta,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the complex matrix-matrix multiplication, using Gauss complexity reduction algorithm.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZgemm3m(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda,
				B, ldb,
				ref beta,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Hgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, __half alpha, IntPtr A, int lda, IntPtr B, int ldb, __half beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasHgemm(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda,
				B, ldb,
				ref beta,
				C, ldc
			));
		}

		/// <summary>
		/// This function is an extension of cublas&lt;t&gt;gemm where the input matrices and output matrices can have a lower precision.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="Atype"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="Btype"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="Ctype"></param>
		/// <param name="ldc"></param>
		public static void SgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, IntPtr A, cudaDataType Atype, int lda, IntPtr B, cudaDataType Btype, int ldb, float beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasSgemmEx(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, Atype, lda,
				B, Btype, ldb,
				ref beta,
				C, Ctype, ldc
			));
		}

		/// <summary>
		/// This function is an extension of cublas&lt;t>gemm that allows the user to individally
		/// specify the data types for each of the A, B and C matrices, the precision of computation
		/// and the GEMM algorithm to be run. Supported combinations of arguments are listed further down in this section.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="Atype"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="Btype"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="Ctype"></param>
		/// <param name="ldc"></param>
		/// <param name="computeType"></param>
		/// <param name="algo"></param>
		public static void GemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, IntPtr alpha, IntPtr A, cudaDataType Atype, int lda, IntPtr B, cudaDataType Btype, int ldb, IntPtr beta, IntPtr C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo) {
			CheckStatus(API.cublasGemmEx(
				handle, transa, transb,
				m, n, k,
				alpha,
				A, Atype, lda,
				B, Btype, ldb,
				beta,
				C, Ctype, ldc,
				computeType, algo
			));
		}

		/// <summary>
		/// This function is an extension of cublas&lt;t>gemm where the input matrices and output matrices can have a lower precision.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="Atype"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="Btype"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="Ctype"></param>
		/// <param name="ldc"></param>
		public static void CgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuComplex alpha, IntPtr A, cudaDataType Atype, int lda, IntPtr B, cudaDataType Btype, int ldb, cuComplex beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCgemmEx(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, Atype, lda,
				B, Btype, ldb,
				ref beta,
				C, Ctype, ldc
			));
		}
		// cublasUint8gemmBias

		/// <summary>
		/// This function performs the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Ssyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Dsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Csyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="Atype"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="Ctype"></param>
		/// <param name="ldc"></param>
		public static void CsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, cudaDataType Atype, int lda, cuComplex beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCsyrkEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="Atype"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="Ctype"></param>
		/// <param name="ldc"></param>
		public static void Csyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, cudaDataType Atype, int lda, cuComplex beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCsyrk3mEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Cherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCherk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZherk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="Atype"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="Ctype"></param>
		/// <param name="ldc"></param>
		public static void CherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, cudaDataType Atype, int lda, float beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCherkEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="Atype"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="Ctype"></param>
		/// <param name="ldc"></param>
		public static void Cherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, cudaDataType Atype, int lda, float beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCherk3mEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- 2 k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Ssyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- 2 k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Dsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- 2 k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Csyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric rank- 2 k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian rank- 2 k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Cher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCher2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian rank- 2 k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZher2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs a variation of the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Ssyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs a variation of the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Dsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs a variation of the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Csyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs a variation of the symmetric rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs a variation of the Hermitian rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Cherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCherkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs a variation of the Hermitian rank- k update.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZherkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Ssymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, float alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Dsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Csymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Chemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasChemm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function performs the Hermitian matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZhemm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		public static void Strsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, float alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}

		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		public static void Dtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}

		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		public static void Ctrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}

		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		public static void Ztrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}

		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Strmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, float alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}

		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Dtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}

		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Ctrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}

		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Ztrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="batchCount"></param>
		public static void SgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, IntPtr Aarray, int lda, IntPtr Barray, int ldb, float beta, IntPtr Carray, int ldc, int batchCount) {
			CheckStatus(API.cublasSgemmBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				Aarray, lda,
				Barray, ldb,
				ref beta,
				Carray, ldc,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="batchCount"></param>
		public static void DgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, IntPtr Aarray, int lda, IntPtr Barray, int ldb, double beta, IntPtr Carray, int ldc, int batchCount) {
			CheckStatus(API.cublasDgemmBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				Aarray, lda,
				Barray, ldb,
				ref beta,
				Carray, ldc,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="batchCount"></param>
		public static void CgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuComplex alpha, IntPtr Aarray, int lda, IntPtr Barray, int ldb, cuComplex beta, IntPtr Carray, int ldc, int batchCount) {
			CheckStatus(API.cublasCgemmBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				Aarray, lda,
				Barray, ldb,
				ref beta,
				Carray, ldc,
				batchCount
			));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="batchCount"></param>
		public static void Cgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuComplex alpha, IntPtr Aarray, int lda, IntPtr Barray, int ldb, cuComplex beta, IntPtr Carray, int ldc, int batchCount) {
			CheckStatus(API.cublasCgemm3mBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				Aarray, lda,
				Barray, ldb,
				ref beta,
				Carray, ldc,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="beta"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="batchCount"></param>
		public static void ZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuDoubleComplex alpha, IntPtr Aarray, int lda, IntPtr Barray, int ldb, cuDoubleComplex beta, IntPtr Carray, int ldc, int batchCount) {
			CheckStatus(API.cublasZgemmBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				Aarray, lda,
				Barray, ldb,
				ref beta,
				Carray, ldc,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="strideA"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="strideB"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="strideC"></param>
		/// <param name="batchCount"></param>
		public static void SgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, IntPtr A, int lda, long strideA, IntPtr B, int ldb, long strideB, float beta, IntPtr C, int ldc, long strideC, int batchCount) {
			CheckStatus(API.cublasSgemmStridedBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda, strideA,
				B, ldb, strideB,
				ref beta,
				C, ldc, strideC,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="strideA"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="strideB"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="strideC"></param>
		/// <param name="batchCount"></param>
		public static void DgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double alpha, IntPtr A, int lda, long strideA, IntPtr B, int ldb, long strideB, double beta, IntPtr C, int ldc, long strideC, int batchCount) {
			CheckStatus(API.cublasDgemmStridedBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda, strideA,
				B, ldb, strideB,
				ref beta,
				C, ldc, strideC,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="strideA"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="strideB"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="strideC"></param>
		/// <param name="batchCount"></param>
		public static void CgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuComplex alpha, IntPtr A, int lda, long strideA, IntPtr B, int ldb, long strideB, cuComplex beta, IntPtr C, int ldc, long strideC, int batchCount) {
			CheckStatus(API.cublasCgemmStridedBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda, strideA,
				B, ldb, strideB,
				ref beta,
				C, ldc, strideC,
				batchCount
			));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="strideA"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="strideB"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="strideC"></param>
		/// <param name="batchCount"></param>
		public static void Cgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuComplex alpha, IntPtr A, int lda, long strideA, IntPtr B, int ldb, long strideB, cuComplex beta, IntPtr C, int ldc, long strideC, int batchCount) {
			CheckStatus(API.cublasCgemm3mStridedBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda, strideA,
				B, ldb, strideB,
				ref beta,
				C, ldc, strideC,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="strideA"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="strideB"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="strideC"></param>
		/// <param name="batchCount"></param>
		public static void ZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, long strideA, IntPtr B, int ldb, long strideB, cuDoubleComplex beta, IntPtr C, int ldc, long strideC, int batchCount) {
			CheckStatus(API.cublasZgemmStridedBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda, strideA,
				B, ldb, strideB,
				ref beta,
				C, ldc, strideC,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication of a batch of matrices.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="k"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="strideA"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="strideB"></param>
		/// <param name="beta"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="strideC"></param>
		/// <param name="batchCount"></param>
		public static void HgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, __half alpha, IntPtr A, int lda, long strideA, IntPtr B, int ldb, long strideB, __half beta, IntPtr C, int ldc, long strideC, int batchCount) {
			CheckStatus(API.cublasHgemmStridedBatched(
				handle, transa, transb,
				m, n, k,
				ref alpha,
				A, lda, strideA,
				B, ldb, strideB,
				ref beta,
				C, ldc, strideC,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix addition/transposition.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Sgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, float alpha, IntPtr A, int lda, float beta, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasSgeam(
				handle, transa, transb,
				m, n,
				ref alpha,
				A, lda,
				ref beta,
				B, ldb,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix addition/transposition.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Dgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, double alpha, IntPtr A, int lda, double beta, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasDgeam(
				handle, transa, transb,
				m, n,
				ref alpha,
				A, lda,
				ref beta,
				B, ldb,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix addition/transposition.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Cgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, cuComplex alpha, IntPtr A, int lda, cuComplex beta, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasCgeam(
				handle, transa, transb,
				m, n,
				ref alpha,
				A, lda,
				ref beta,
				B, ldb,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix addition/transposition.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="transa"></param>
		/// <param name="transb"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="beta"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, cuDoubleComplex beta, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasZgeam(
				handle, transa, transb,
				m, n,
				ref alpha,
				A, lda,
				ref beta,
				B, ldb,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the LU factorization of each Aarray[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void SgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}

		/// <summary>
		/// This function performs the LU factorization of each Aarray[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void DgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}

		/// <summary>
		/// This function performs the LU factorization of each Aarray[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void CgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}

		/// <summary>
		/// This function performs the LU factorization of each Aarray[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void ZgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void SgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void DgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void CgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="P"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void ZgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}

		/// <summary>
		/// This function solves an array of systems of linear equations.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="devIpiv"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void SgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}

		/// <summary>
		/// This function solves an array of systems of linear equations.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="devIpiv"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void DgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}

		/// <summary>
		/// This function solves an array of systems of linear equations.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="devIpiv"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void CgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}

		/// <summary>
		/// This function solves an array of systems of linear equations.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="devIpiv"></param>
		/// <param name="Barray"></param>
		/// <param name="ldb"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void ZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="batchCount"></param>
		public static void StrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, float alpha, IntPtr A, int lda, IntPtr B, int ldb, int batchCount) {
			CheckStatus(API.cublasStrsmBatched(
				handle, side, uplo, trans, diag,
				m, n,
				ref alpha,
				A, lda,
				B, ldb,
				batchCount
			));
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="batchCount"></param>
		public static void DtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb, int batchCount) {
			CheckStatus(API.cublasDtrsmBatched(
				handle, side, uplo, trans, diag,
				m, n,
				ref alpha,
				A, lda,
				B, ldb,
				batchCount
			));
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="batchCount"></param>
		public static void CtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, int batchCount) {
			CheckStatus(API.cublasCtrsmBatched(
				handle, side, uplo, trans, diag,
				m, n,
				ref alpha,
				A, lda,
				B, ldb,
				batchCount
			));
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="side"></param>
		/// <param name="uplo"></param>
		/// <param name="trans"></param>
		/// <param name="diag"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="alpha"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="B"></param>
		/// <param name="ldb"></param>
		/// <param name="batchCount"></param>
		public static void ZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, int batchCount) {
			CheckStatus(API.cublasZtrsmBatched(
				handle, side, uplo, trans, diag,
				m, n,
				ref alpha,
				A, lda,
				B, ldb,
				batchCount
			));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void SmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void DmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void CmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void ZmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i = 0, ...,batchSize-1 using Householder reflections.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="TauArray"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void SgeqrfBatched(cublasHandle_t handle, int m, int n, IntPtr Aarray, int lda, IntPtr TauArray, int info, int batchSize) {
			CheckStatus(API.cublasSgeqrfBatched(
				handle,
				m, n,
				Aarray, lda,
				TauArray,
				ref info,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i = 0, ...,batchSize-1 using Householder reflections.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="TauArray"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void DgeqrfBatched(cublasHandle_t handle, int m, int n, IntPtr Aarray, int lda, IntPtr TauArray, int info, int batchSize) {
			CheckStatus(API.cublasDgeqrfBatched(
				handle,
				m, n,
				Aarray, lda,
				TauArray,
				ref info,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i = 0, ...,batchSize-1 using Householder reflections.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="TauArray"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void CgeqrfBatched(cublasHandle_t handle, int m, int n, IntPtr Aarray, int lda, IntPtr TauArray, int info, int batchSize) {
			CheckStatus(API.cublasCgeqrfBatched(
				handle,
				m, n,
				Aarray, lda,
				TauArray,
				ref info,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i = 0, ...,batchSize-1 using Householder reflections.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="TauArray"></param>
		/// <param name="info"></param>
		/// <param name="batchSize"></param>
		public static void ZgeqrfBatched(cublasHandle_t handle, int m, int n, IntPtr Aarray, int lda, IntPtr TauArray, int info, int batchSize) {
			CheckStatus(API.cublasZgeqrfBatched(
				handle,
				m, n,
				Aarray, lda,
				TauArray,
				ref info,
				batchSize
			));
		}

		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="devInfoArray"></param>
		/// <param name="batchSize"></param>
		public static void SgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, IntPtr Aarray, int lda, IntPtr Carray, int ldc, int info, IntPtr devInfoArray, int batchSize) {
			CheckStatus(API.cublasSgelsBatched(
				handle, trans,
				m, n, nrhs,
				Aarray, lda,
				Carray, ldc,
				ref info,
				devInfoArray,
				batchSize
			));
		}

		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="devInfoArray"></param>
		/// <param name="batchSize"></param>
		public static void DgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, IntPtr Aarray, int lda, IntPtr Carray, int ldc, int info, IntPtr devInfoArray, int batchSize) {
			CheckStatus(API.cublasDgelsBatched(
				handle, trans,
				m, n, nrhs,
				Aarray, lda,
				Carray, ldc,
				ref info,
				devInfoArray,
				batchSize
			));
		}

		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="devInfoArray"></param>
		/// <param name="batchSize"></param>
		public static void CgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, IntPtr Aarray, int lda, IntPtr Carray, int ldc, int info, IntPtr devInfoArray, int batchSize) {
			CheckStatus(API.cublasCgelsBatched(
				handle, trans,
				m, n, nrhs,
				Aarray, lda,
				Carray, ldc,
				ref info,
				devInfoArray,
				batchSize
			));
		}

		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="trans"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="nrhs"></param>
		/// <param name="Aarray"></param>
		/// <param name="lda"></param>
		/// <param name="Carray"></param>
		/// <param name="ldc"></param>
		/// <param name="info"></param>
		/// <param name="devInfoArray"></param>
		/// <param name="batchSize"></param>
		public static void ZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, IntPtr Aarray, int lda, IntPtr Carray, int ldc, int info, IntPtr devInfoArray, int batchSize) {
			CheckStatus(API.cublasZgelsBatched(
				handle, trans,
				m, n, nrhs,
				Aarray, lda,
				Carray, ldc,
				ref info,
				devInfoArray,
				batchSize
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Sdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasSdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Ddgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasDdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Cdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasCdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="m"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="C"></param>
		/// <param name="ldc"></param>
		public static void Zdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasZdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}

		/// <summary>
		/// This function performs the conversion from the triangular packed format to the triangular format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Stpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasStpttr(handle, uplo, n, AP, A, lda));
		}

		/// <summary>
		/// This function performs the conversion from the triangular packed format to the triangular format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Dtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasDtpttr(handle, uplo, n, AP, A, lda));
		}

		/// <summary>
		/// This function performs the conversion from the triangular packed format to the triangular format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Ctpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasCtpttr(handle, uplo, n, AP, A, lda));
		}

		/// <summary>
		/// This function performs the conversion from the triangular packed format to the triangular format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="AP"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		public static void Ztpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasZtpttr(handle, uplo, n, AP, A, lda));
		}

		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular packed format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="AP"></param>
		public static void Strttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasStrttp(handle, uplo, n, A, lda, AP));
		}

		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular packed format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="AP"></param>
		public static void Dtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasDtrttp(handle, uplo, n, A, lda, AP));
		}

		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular packed format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="AP"></param>
		public static void Ctrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasCtrttp(handle, uplo, n, A, lda, AP));
		}

		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular packed format.
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="uplo"></param>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="AP"></param>
		public static void Ztrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasZtrttp(handle, uplo, n, A, lda, AP));
		}

		static void CheckStatus(cublasStatus status) {
			if (status != cublasStatus.CUBLAS_STATUS_SUCCESS) {
				throw new CudaException(status.ToString());
			}
		}
	}
}
