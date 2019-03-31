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

	/// <summary>
	/// The cuBLAS library is an implementation of BLAS (Basic Linear Algebra Subprograms).
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cublas/">http://docs.nvidia.com/cuda/cublas/</a> 
	/// </remarks>
	public partial class cuBLAS {
		/// <summary>
		/// cuBLAS DLL functions.
		/// </summary>
		public class API {
			//const string DLL_PATH = "cublas64_80.dll";
			const string DLL_PATH = "cublas64_10.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			// ----- v1

			/// <summary>
			/// initialize the library.
			/// </summary>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasInit();

			/// <summary>
			/// shuts down the library.
			/// </summary>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasShutdown();

			/// <summary>
			/// retrieves the error status of the library.
			/// </summary>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetError();

			/// <summary>
			/// This function returns the version number of the cuBLAS library.
			/// </summary>
			/// <param name="version"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetVersion(ref int version);

			/// <summary>
			/// allocates the device memory for the library.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="elemSize"></param>
			/// <param name="devicePtr"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasAlloc(int n, int elemSize, ref IntPtr devicePtr);

			/// <summary>
			/// releases the device memory allocated for the library.
			/// </summary>
			/// <param name="devicePtr"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasFree(IntPtr devicePtr);

			/// <summary>
			/// sets the stream to be used by the library.
			/// </summary>
			/// <param name="stream"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetKernelStream(cudaStream_t stream);

			// ---------------- CUBLAS BLAS1 functions ----------------
			// NRM2

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasSnrm2(int n, ref float x, int incx);

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDnrm2(int n, ref double x, int incx);

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasScnrm2(int n, ref cuComplex x, int incx);

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDznrm2(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// DOT

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasSdot(int n, ref float x, int incx, ref float y, int incy);

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDdot(int n, ref double x, int incx, ref double y, int incy);

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuComplex cublasCdotu(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuComplex cublasCdotc(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuDoubleComplex cublasZdotu(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuDoubleComplex cublasZdotc(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SCAL

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSscal(int n, float alpha, ref float x, int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDscal(int n, double alpha, ref double x, int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCscal(int n, cuComplex alpha, ref cuComplex x, int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZscal(int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsscal(int n, float alpha, ref cuComplex x, int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZdscal(int n, double alpha, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// AXPY

			/// <summary>
			/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSaxpy(int n, float alpha, ref float x, int incx, ref float y, int incy);

			/// <summary>
			/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDaxpy(int n, double alpha, ref double x, int incx, ref double y, int incy);

			/// <summary>
			/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCaxpy(int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy);

			/// <summary>
			/// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZaxpy(int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// COPY

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasScopy(int n, ref float x, int incx, ref float y, int incy);

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDcopy(int n, ref double x, int incx, ref double y, int incy);

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCcopy(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZcopy(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SWAP

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSswap(int n, ref float x, int incx, ref float y, int incy);

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDswap(int n, ref double x, int incx, ref double y, int incy);

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCswap(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZswap(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// AMAX

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIsamax(int n, ref float x, int incx);

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIdamax(int n, ref double x, int incx);

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIcamax(int n, ref cuComplex x, int incx);

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIzamax(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// AMIN

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIsamin(int n, ref float x, int incx);

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIdamin(int n, ref double x, int incx);

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIcamin(int n, ref cuComplex x, int incx);

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIzamin(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// ASUM

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasSasum(int n, ref float x, int incx);

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDasum(int n, ref double x, int incx);

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasScasum(int n, ref cuComplex x, int incx);

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDzasum(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// ROT

			/// <summary>
			/// This function applies Givens rotation matrix.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="sc"></param>
			/// <param name="ss"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrot(int n, ref float x, int incx, ref float y, int incy, float sc, float ss);

			/// <summary>
			/// This function applies Givens rotation matrix.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="sc"></param>
			/// <param name="ss"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrot(int n, ref double x, int incx, ref double y, int incy, double sc, double ss);

			/// <summary>
			/// This function applies Givens rotation matrix.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="c"></param>
			/// <param name="s"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCrot(int n, ref cuComplex x, int incx, ref cuComplex y, int incy, float c, cuComplex s);

			/// <summary>
			/// This function applies Givens rotation matrix.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="sc"></param>
			/// <param name="cs"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZrot(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, double sc, cuDoubleComplex cs);

			/// <summary>
			/// This function applies Givens rotation matrix.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="c"></param>
			/// <param name="s"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsrot(int n, ref cuComplex x, int incx, ref cuComplex y, int incy, float c, float s);

			/// <summary>
			/// This function applies Givens rotation matrix.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="c"></param>
			/// <param name="s"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZdrot(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, double c, double s);

			// -----------------------------------------------------------------------
			// ROTG

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="sa"></param>
			/// <param name="sb"></param>
			/// <param name="sc"></param>
			/// <param name="ss"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrotg(ref float sa, ref float sb, ref float sc, ref float ss);

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="sa"></param>
			/// <param name="sb"></param>
			/// <param name="sc"></param>
			/// <param name="ss"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrotg(ref double sa, ref double sb, ref double sc, ref double ss);

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="ca"></param>
			/// <param name="cb"></param>
			/// <param name="sc"></param>
			/// <param name="cs"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCrotg(ref cuComplex ca, cuComplex cb, ref float sc, ref cuComplex cs);

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="ca"></param>
			/// <param name="cb"></param>
			/// <param name="sc"></param>
			/// <param name="cs"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZrotg(ref cuDoubleComplex ca, cuDoubleComplex cb, ref double sc, ref cuDoubleComplex cs);

			// -----------------------------------------------------------------------
			// ROTM

			/// <summary>
			/// This function applies the modified Givens transformation.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="sparam"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrotm(int n, ref float x, int incx, ref float y, int incy, ref float sparam);

			/// <summary>
			/// This function applies the modified Givens transformation.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="sparam"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrotm(int n, ref double x, int incx, ref double y, int incy, ref double sparam);

			// -----------------------------------------------------------------------
			// ROTMG

			/// <summary>
			/// This function constructs the modified Givens transformation.
			/// </summary>
			/// <param name="sd1"></param>
			/// <param name="sd2"></param>
			/// <param name="sx1"></param>
			/// <param name="sy1"></param>
			/// <param name="sparam"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrotmg(ref float sd1, ref float sd2, ref float sx1, ref float sy1, ref float sparam);

			/// <summary>
			/// This function constructs the modified Givens transformation.
			/// </summary>
			/// <param name="sd1"></param>
			/// <param name="sd2"></param>
			/// <param name="sx1"></param>
			/// <param name="sy1"></param>
			/// <param name="sparam"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrotmg(ref double sd1, ref double sd2, ref double sx1, ref double sy1, ref double sparam);

			// --------------- CUBLAS BLAS2 functions  ----------------
			// GEMV

			/// <summary>
			/// This function performs the matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSgemv(char trans, int m, int n, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			/// <summary>
			/// This function performs the matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDgemv(char trans, int m, int n, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			/// <summary>
			/// This function performs the matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgemv(char trans, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			/// <summary>
			/// This function performs the matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgemv(char trans, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// GBMV

			/// <summary>
			/// This function performs the banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			/// <summary>
			/// This function performs the banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDgbmv(char trans, int m, int n, int kl, int ku, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			/// <summary>
			/// This function performs the banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgbmv(char trans, int m, int n, int kl, int ku, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			/// <summary>
			/// This function performs the banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgbmv(char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// TRMV

			/// <summary>
			/// This function performs the triangular matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrmv(char uplo, char trans, char diag, int n, ref float A, int lda, ref float x, int incx);

			/// <summary>
			/// This function performs the triangular matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrmv(char uplo, char trans, char diag, int n, ref double A, int lda, ref double x, int incx);

			/// <summary>
			/// This function performs the triangular matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrmv(char uplo, char trans, char diag, int n, ref cuComplex A, int lda, ref cuComplex x, int incx);

			/// <summary>
			/// This function performs the triangular matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrmv(char uplo, char trans, char diag, int n, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TBMV

			/// <summary>
			/// This function performs the triangular banded matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStbmv(char uplo, char trans, char diag, int n, int k, ref float A, int lda, ref float x, int incx);

			/// <summary>
			/// This function performs the triangular banded matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtbmv(char uplo, char trans, char diag, int n, int k, ref double A, int lda, ref double x, int incx);

			/// <summary>
			/// This function performs the triangular banded matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtbmv(char uplo, char trans, char diag, int n, int k, ref cuComplex A, int lda, ref cuComplex x, int incx);

			/// <summary>
			/// This function performs the triangular banded matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtbmv(char uplo, char trans, char diag, int n, int k, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TPMV

			/// <summary>
			/// This function performs the triangular packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStpmv(char uplo, char trans, char diag, int n, ref float AP, ref float x, int incx);

			/// <summary>
			/// This function performs the triangular packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtpmv(char uplo, char trans, char diag, int n, ref double AP, ref double x, int incx);

			/// <summary>
			/// This function performs the triangular packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtpmv(char uplo, char trans, char diag, int n, ref cuComplex AP, ref cuComplex x, int incx);

			/// <summary>
			/// This function performs the triangular packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtpmv(char uplo, char trans, char diag, int n, ref cuDoubleComplex AP, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TRSV

			/// <summary>
			/// This function solves the triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrsv(char uplo, char trans, char diag, int n, ref float A, int lda, ref float x, int incx);

			/// <summary>
			/// This function solves the triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrsv(char uplo, char trans, char diag, int n, ref double A, int lda, ref double x, int incx);

			/// <summary>
			/// This function solves the triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrsv(char uplo, char trans, char diag, int n, ref cuComplex A, int lda, ref cuComplex x, int incx);

			/// <summary>
			/// This function solves the triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrsv(char uplo, char trans, char diag, int n, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TPSV

			/// <summary>
			/// This function solves the packed triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStpsv(char uplo, char trans, char diag, int n, ref float AP, ref float x, int incx);

			/// <summary>
			/// This function solves the packed triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtpsv(char uplo, char trans, char diag, int n, ref double AP, ref double x, int incx);

			/// <summary>
			/// This function solves the packed triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtpsv(char uplo, char trans, char diag, int n, ref cuComplex AP, ref cuComplex x, int incx);

			/// <summary>
			/// This function solves the packed triangular linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtpsv(char uplo, char trans, char diag, int n, ref cuDoubleComplex AP, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TBSV

			/// <summary>
			/// This function solves the triangular banded linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStbsv(char uplo, char trans, char diag, int n, int k, ref float A, int lda, ref float x, int incx);

			/// <summary>
			/// This function solves the triangular banded linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtbsv(char uplo, char trans, char diag, int n, int k, ref double A, int lda, ref double x, int incx);

			/// <summary>
			/// This function solves the triangular banded linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtbsv(char uplo, char trans, char diag, int n, int k, ref cuComplex A, int lda, ref cuComplex x, int incx);

			/// <summary>
			/// This function solves the triangular banded linear system with a single right-hand-side.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="trans"></param>
			/// <param name="diag"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtbsv(char uplo, char trans, char diag, int n, int k, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// SYMV/HEMV

			/// <summary>
			/// This function performs the symmetric matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsymv(char uplo, int n, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			/// <summary>
			/// This function performs the symmetric matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsymv(char uplo, int n, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			/// <summary>
			/// This function performs the symmetric matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChemv(char uplo, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			/// <summary>
			/// This function performs the symmetric matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhemv(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SBMV/HBMV

			/// <summary>
			/// This function performs the symmetric banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsbmv(char uplo, int n, int k, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			/// <summary>
			/// This function performs the symmetric banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsbmv(char uplo, int n, int k, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			/// <summary>
			/// This function performs the symmetric banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChbmv(char uplo, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			/// <summary>
			/// This function performs the symmetric banded matrix-vector multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhbmv(char uplo, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SPMV/HPMV

			/// <summary>
			/// This function performs the symmetric packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="beta"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSspmv(char uplo, int n, float alpha, ref float AP, ref float x, int incx, float beta, ref float y, int incy);

			/// <summary>
			/// This function performs the symmetric packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="beta"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDspmv(char uplo, int n, double alpha, ref double AP, ref double x, int incx, double beta, ref double y, int incy);

			/// <summary>
			/// This function performs the symmetric packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="beta"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChpmv(char uplo, int n, cuComplex alpha, ref cuComplex AP, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			/// <summary>
			/// This function performs the symmetric packed matrix-vector multiplication.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="AP"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="beta"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhpmv(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex AP, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// GER

			/// <summary>
			/// This function performs the rank-1 update.
			/// </summary>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSger(int m, int n, float alpha, ref float x, int incx, ref float y, int incy, ref float A, int lda);

			/// <summary>
			/// This function performs the rank-1 update.
			/// </summary>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDger(int m, int n, double alpha, ref double x, int incx, ref double y, int incy, ref double A, int lda);

			/// <summary>
			/// This function performs the rank-1 update.
			/// </summary>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgeru(int m, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex A, int lda);

			/// <summary>
			/// This function performs the rank-1 update.
			/// </summary>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgerc(int m, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex A, int lda);

			/// <summary>
			/// This function performs the rank-1 update.
			/// </summary>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgeru(int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex A, int lda);

			/// <summary>
			/// This function performs the rank-1 update.
			/// </summary>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgerc(int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex A, int lda);

			// -----------------------------------------------------------------------
			// SYR/HER

			/// <summary>
			/// This function performs the symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyr(char uplo, int n, float alpha, ref float x, int incx, ref float A, int lda);

			/// <summary>
			/// This function performs the symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyr(char uplo, int n, double alpha, ref double x, int incx, ref double A, int lda);

			/// <summary>
			/// This function performs the symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCher(char uplo, int n, float alpha, ref cuComplex x, int incx, ref cuComplex A, int lda);

			/// <summary>
			/// This function performs the symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZher(char uplo, int n, double alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex A, int lda);

			// -----------------------------------------------------------------------
			// SPR/HPR

			/// <summary>
			/// This function performs the packed symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSspr(char uplo, int n, float alpha, ref float x, int incx, ref float AP);

			/// <summary>
			/// This function performs the packed symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDspr(char uplo, int n, double alpha, ref double x, int incx, ref double AP);

			/// <summary>
			/// This function performs the packed symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChpr(char uplo, int n, float alpha, ref cuComplex x, int incx, ref cuComplex AP);

			/// <summary>
			/// This function performs the packed symmetric rank-1 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhpr(char uplo, int n, double alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex AP);

			// -----------------------------------------------------------------------
			// SYR2/HER2

			/// <summary>
			/// This function performs the symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyr2(char uplo, int n, float alpha, ref float x, int incx, ref float y, int incy, ref float A, int lda);

			/// <summary>
			/// This function performs the symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyr2(char uplo, int n, double alpha, ref double x, int incx, ref double y, int incy, ref double A, int lda);

			/// <summary>
			/// This function performs the symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCher2(char uplo, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex A, int lda);

			/// <summary>
			/// This function performs the symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZher2(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex A, int lda);

			// -----------------------------------------------------------------------
			// SPR2/HPR2

			/// <summary>
			/// This function performs the packed symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSspr2(char uplo, int n, float alpha, ref float x, int incx, ref float y, int incy, ref float AP);

			/// <summary>
			/// This function performs the packed symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDspr2(char uplo, int n, double alpha, ref double x, int incx, ref double y, int incy, ref double AP);

			/// <summary>
			/// This function performs the packed symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChpr2(char uplo, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex AP);

			/// <summary>
			/// This function performs the packed symmetric rank-2 update.
			/// </summary>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="AP"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhpr2(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex AP);

			// ------------------------BLAS3 Functions -------------------------------
			// GEMM

			/// <summary>
			/// This function performs the matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, ref float A, int lda, ref float B, int ldb, float beta, ref float C, int ldc);

			/// <summary>
			/// This function performs the matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDgemm(char transa, char transb, int m, int n, int k, double alpha, ref double A, int lda, ref double B, int ldb, double beta, ref double C, int ldc);

			/// <summary>
			/// This function performs the matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgemm(char transa, char transb, int m, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			/// <summary>
			/// This function performs the matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgemm(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// SYRK

			/// <summary>
			/// This function performs the symmetric rank- k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyrk(char uplo, char trans, int n, int k, float alpha, ref float A, int lda, float beta, ref float C, int ldc);

			/// <summary>
			/// This function performs the symmetric rank- k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyrk(char uplo, char trans, int n, int k, double alpha, ref double A, int lda, double beta, ref double C, int ldc);

			/// <summary>
			/// This function performs the symmetric rank- k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsyrk(char uplo, char trans, int n, int k, cuComplex alpha, ref cuComplex A, int lda, cuComplex beta, ref cuComplex C, int ldc);

			/// <summary>
			/// This function performs the symmetric rank- k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZsyrk(char uplo, char trans, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// HERK

			/// <summary>
			/// This function performs the Hermitian rank- k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCherk(char uplo, char trans, int n, int k, float alpha, ref cuComplex A, int lda, float beta, ref cuComplex C, int ldc);

			/// <summary>
			/// This function performs the Hermitian rank- k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZherk(char uplo, char trans, int n, int k, double alpha, ref cuDoubleComplex A, int lda, double beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// SYR2K

			/// <summary>
			/// This function performs the symmetric rank- 2 k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, ref float A, int lda, ref float B, int ldb, float beta, ref float C, int ldc);

			/// <summary>
			/// This function performs the symmetric rank- 2 k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, ref double A, int lda, ref double B, int ldb, double beta, ref double C, int ldc);

			/// <summary>
			/// This function performs the symmetric rank- 2 k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsyr2k(char uplo, char trans, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			/// <summary>
			/// This function performs the symmetric rank- 2 k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZsyr2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// HER2K

			/// <summary>
			/// This function performs the Hermitian rank- 2 k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCher2k(char uplo, char trans, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, float beta, ref cuComplex C, int ldc);

			/// <summary>
			/// This function performs the Hermitian rank- 2 k update.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZher2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, double beta, ref cuDoubleComplex C, int ldc);

			// -----------------------------------------------------------------------
			// SYMM

			/// <summary>
			/// This function performs the symmetric matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsymm(char side, char uplo, int m, int n, float alpha, ref float A, int lda, ref float B, int ldb, float beta, ref float C, int ldc);

			/// <summary>
			/// This function performs the symmetric matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsymm(char side, char uplo, int m, int n, double alpha, ref double A, int lda, ref double B, int ldb, double beta, ref double C, int ldc);

			/// <summary>
			/// This function performs the symmetric matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsymm(char side, char uplo, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			/// <summary>
			/// This function performs the symmetric matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZsymm(char side, char uplo, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -----------------------------------------------------------------------
			// HEMM

			/// <summary>
			/// This function performs the Hermitian matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChemm(char side, char uplo, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			/// <summary>
			/// This function performs the Hermitian matrix-matrix multiplication.
			/// </summary>
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
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhemm(char side, char uplo, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -----------------------------------------------------------------------
			// TRSM

			/// <summary>
			/// This function solves the triangular linear system with multiple right-hand-sides.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, ref float A, int lda, ref float B, int ldb);

			/// <summary>
			/// This function solves the triangular linear system with multiple right-hand-sides.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, ref double A, int lda, ref double B, int ldb);

			/// <summary>
			/// This function solves the triangular linear system with multiple right-hand-sides.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrsm(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb);

			/// <summary>
			/// This function solves the triangular linear system with multiple right-hand-sides.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrsm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb);

			// -----------------------------------------------------------------------
			// TRMM

			/// <summary>
			/// This function performs the triangular matrix-matrix multiplication.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, ref float A, int lda, ref float B, int ldb);

			/// <summary>
			/// This function performs the triangular matrix-matrix multiplication.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, ref double A, int lda, ref double B, int ldb);

			/// <summary>
			/// This function performs the triangular matrix-matrix multiplication.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrmm(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb);

			/// <summary>
			/// This function performs the triangular matrix-matrix multiplication.
			/// </summary>
			/// <param name="side"></param>
			/// <param name="uplo"></param>
			/// <param name="transa"></param>
			/// <param name="diag"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrmm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb);

			// ----- v2

			/// <summary>
			/// This function initializes the CUBLAS library and creates a handle to an opaque structure holding the CUBLAS library context.
			/// </summary>
			/// <param name="handle"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCreate_v2(ref cublasHandle_t handle);

			/// <summary>
			/// This function releases hardware resources used by the CUBLAS library.
			/// </summary>
			/// <param name="handle"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

			/// <summary>
			/// This function returns the version number of the cuBLAS library.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="version"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, ref int version);

			/// <summary>
			/// This function returns the value of the requested property in memory pointed to by value.
			/// Refer to libraryPropertyType for supported types.
			/// </summary>
			/// <param name="type"></param>
			/// <param name="value"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetProperty(libraryPropertyType type, ref int value);

			/// <summary>
			/// This function sets the cuBLAS library stream, which will be used to execute all subsequent calls to the cuBLAS library functions.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="streamId"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);

			/// <summary>
			/// This function gets the cuBLAS library stream, which is being used to execute all calls to the cuBLAS library functions.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="streamId"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, ref cudaStream_t streamId);

			/// <summary>
			/// This function obtains the pointer mode used by the cuBLAS library.
			/// Please see the section on the cublasPointerMode_t type for more details.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="mode"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, ref cublasPointerMode_t mode);

			/// <summary>
			/// This function sets the pointer mode used by the cuBLAS library.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="mode"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);

			/// <summary>
			/// This function queries the atomic mode of a specific cuBLAS context.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="mode"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, ref cublasAtomicsMode_t mode);

			/// <summary>
			/// This function allows or disallows the usage of atomics in the cuBLAS library for all routines which have an alternate implementation.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="mode"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);

			/// <summary>
			/// This function copies n elements from a vector x in host memory space to a vector y in GPU memory space.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="elemSize"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetVector(
				int n, int elemSize,
				IntPtr x, // [host] const void *
				int incx,
				IntPtr y, // [device] void *
				int incy);

			/// <summary>
			/// This function copies n elements from a vector x in GPU memory space to a vector y in host memory space.
			/// </summary>
			/// <param name="n"></param>
			/// <param name="elemSize"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetVector(
				int n, int elemSize,
				IntPtr x, // [device] const void *
				int incx,
				IntPtr y, // [host] void *
				int incy);

			/// <summary>
			/// This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space.
			/// </summary>
			/// <param name="rows"></param>
			/// <param name="cols"></param>
			/// <param name="elemSize"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetMatrix(
				int rows, int cols, int elemSize,
				IntPtr A, // [host] const void *
				int lda,
				IntPtr B, // [device] void *
				int ldb);

			/// <summary>
			/// This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space.
			/// </summary>
			/// <param name="rows"></param>
			/// <param name="cols"></param>
			/// <param name="elemSize"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="ldb"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetMatrix(
				int rows, int cols, int elemSize,
				IntPtr A, // [device] const void *
				int lda,
				IntPtr B, // [host] void *
				int ldb);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetVectorAsync(
				int n, int elemSize,
				IntPtr hostPtr, int incx,
				IntPtr devicePtr, int incy,
				cudaStream_t stream);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetVectorAsync(
				int n, int elemSize,
				IntPtr devicePtr, int incx,
				IntPtr hostPtr, int incy,
				cudaStream_t stream);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetMatrixAsync(
				int rows, int cols, int elemSize,
				IntPtr A, int lda, IntPtr B,
				int ldb, cudaStream_t stream);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetMatrixAsync(
				int rows, int cols, int elemSize,
				IntPtr A, int lda, IntPtr B,
				int ldb, cudaStream_t stream);

			/// <summary>
			/// The cublasSetMathMode function enables you to choose whether or not to use Tensor Core operations in the library by setting the math mode to either CUBLAS_TENSOR_OP_MATH or CUBLAS_DEFAULT_MATH.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="mode"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);

			/// <summary>
			/// This function returns the math mode used by the library routines.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="mode"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetMathMode(cublasHandle_t handle, ref cublasMath_t mode);

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudnnStatus_t cublasLoggerConfigure(
			//	int logIsOn,
			//	int logToStdOut,
			//	int logToStdErr,
			//	string logFileName); // const char*

			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudnnStatus_t cublasGetLoggerCallback(ref cublasLogCallback userCallback);
			//[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			//public static extern cudnnStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasXerbla(string srName, int info);

			// ---------------- CUBLAS BLAS1 functions ----------------

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="xType"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <param name="resultType"></param>
			/// <param name="executionType"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasNrm2Ex(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const void *
				cudaDataType xType,
				int incx,
				IntPtr result, // [host or device] void *
				cudaDataType resultType,
				cudaDataType executionType);

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSnrm2_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const float*
				int incx,
				ref float result); // [host] 

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSnrm2_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const float*
				int incx,
				IntPtr result); // [device] 

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDnrm2_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const double*
				int incx,
				ref double result);  // [host or device] 

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScnrm2_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuComplex*
				int incx,
				ref float result);  // [host or device] 

			/// <summary>
			/// This function computes the Euclidean norm of the vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDznrm2_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuDoubleComplex*
				int incx,
				ref double result);  // [host or device] 

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
			/// <param name="result"></param>
			/// <param name="resultType"></param>
			/// <param name="executionType"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDotEx(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const void *
				cudaDataType xType, 
				int incx,
				IntPtr y, // [device] const void *
				cudaDataType yType,
				int incy,
				IntPtr result, // [host or device] void *
				cudaDataType resultType,
				cudaDataType executionType);

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
			/// <param name="result"></param>
			/// <param name="resultType"></param>
			/// <param name="executionType"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDotcEx(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const void *
				cudaDataType xType, 
				int incx,
				IntPtr y, // [device] const void *
				cudaDataType yType,
				int incy,
				IntPtr result, // [host or device] void *
				cudaDataType resultType,
				cudaDataType executionType);

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSdot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const float *
				int incx,
				IntPtr y, // [device] const float *
				int incy,
				IntPtr result);  // [host or device] 

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDdot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x,
				int incx,
				IntPtr y,
				int incy,
				IntPtr result);  // [host or device] 

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdotu_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] const cuComplex *
				int incy,
				ref cuComplex result);  // [[host or device] ] cuComplex *

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdotc_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] const cuComplex *
				int incy,
				ref cuComplex result); // [[host or device] ] cuComplex *

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdotu_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] const cuDoubleComplex *
				int incy,
				ref cuDoubleComplex result); // [[host or device] ] cuDoubleComplex *

			/// <summary>
			/// This function computes the dot product of vectors x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdotc_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] const cuDoubleComplex *
				int incy,
				ref cuDoubleComplex result); // [[host or device] ] cuDoubleComplex *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScalEx(
				cublasHandle_t handle,
				int n,
				IntPtr alpha,  // [host or device] const void *
				cudaDataType alphaType,
				IntPtr x, // [device] void *
				cudaDataType xType,
				int incx,
				cudaDataType executionType);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSscal_v2(
				cublasHandle_t handle,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x,     // [device] float *
				int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDscal_v2(
				cublasHandle_t handle,
				int n,
				ref double alpha,  // [host or device] const double *
				IntPtr x, // [device] double *
				int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCscal_v2(
				cublasHandle_t handle,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr x,  // [device] cuComplex *
				int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsscal_v2(
				cublasHandle_t handle,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] cuComplex *
				int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZscal_v2(
				cublasHandle_t handle,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr x,  // [device] cuDoubleComplex *
				int incx);

			/// <summary>
			/// This function scales the vector x by the scalar α and overwrites it with the result.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="alpha"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdscal_v2(
				cublasHandle_t handle,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] cuDoubleComplex *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasAxpyEx(
				cublasHandle_t handle,
				int n,
				IntPtr alpha, // [host or device] const void *
				cudaDataType alphaType,
				IntPtr x, // [device] const void *
				cudaDataType xType,
				int incx,
				IntPtr y, // [device] void *
				cudaDataType yType,
				int incy,
				cudaDataType executiontype);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSaxpy_v2(
				cublasHandle_t handle,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] const float *
				int incx,
				IntPtr y, // [device] float *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDaxpy_v2(
				cublasHandle_t handle,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] const double *
				int incx,
				IntPtr y, // [device] double *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCaxpy_v2(
				cublasHandle_t handle,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] cuComplex *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZaxpy_v2(
				cublasHandle_t handle,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScopy_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const float *
				int incx,
				IntPtr y, // [device] float *
				int incy);

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDcopy_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const double *
				int incx,
				IntPtr y, // [device] double *
				int incy);

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCcopy_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] cuComplex *
				int incy);

			/// <summary>
			/// This function copies the vector x into the vector y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZcopy_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSswap_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] float *
				int incx,
				IntPtr y, // [device] float *
				int incy);

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDswap_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] double *
				int incx,
				IntPtr y, // [device] double *
				int incy);

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCswap_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] cuComplex *
				int incx,
				IntPtr y, // [device] cuComplex *
				int incy);

			/// <summary>
			/// This function interchanges the elements of vector x and y.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="y"></param>
			/// <param name="incy"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZswap_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] cuDoubleComplex *
				int incx,
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIsamax_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const float *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIdamax_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const double *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIcamax_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function finds the (smallest) index of the element of the maximum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIzamax_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIsamin_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const float *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIdamin_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const double *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIcamin_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function finds the (smallest) index of the element of the minimum magnitude.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIzamin_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref int result); // [host or device] int *

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSasum_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const float *
				int incx,
				ref float result); // [host or device] float *

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDasum_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const double *
				int incx,
				ref double result); // [host or device] double *

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScasum_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref float result); // [host or device] float *

			/// <summary>
			/// This function computes the sum of the absolute values of the elements of vector x.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="n"></param>
			/// <param name="x"></param>
			/// <param name="incx"></param>
			/// <param name="result"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDzasum_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref double result); // [host or device] double *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] float *
				int incx,
				IntPtr y, // [device] float *
				int incy,
				float[] c,  // [host or device] const float *
				float[] s); // [host or device] const float *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] double *
				int incx,
				IntPtr y, // [device] double *
				int incy,
				double[] c,  // [host or device] const double *
				double[] s); // [host or device] const double *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCrot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] cuComplex *
				int incx,
				IntPtr y, // [device] cuComplex *
				int incy,
				float[] c,      // [host or device] const float *
				cuComplex[] s); // [host or device] const cuComplex *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsrot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] cuComplex *
				int incx,
				IntPtr y, // [device] cuComplex *
				int incy,
				float[] c,  // [host or device] const float *
				float[] s); // [host or device] const float *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZrot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] cuDoubleComplex *
				int incx,
				IntPtr y, // [device] cuDoubleComplex *
				int incy,
				double[] c,            // [host or device] const double *
				cuDoubleComplex[] s);  // [host or device] const cuDoubleComplex *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdrot_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] cuDoubleComplex *
				int incx,
				IntPtr y, // [device] cuDoubleComplex *
				int incy,
				double[] c,  // [host or device] const double *
				double[] s); // [host or device] const double *

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="s"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotg_v2(
				cublasHandle_t handle,
				ref float a,   // [host or device] float *
				ref float b,   // [host or device] float *
				ref float c,   // [host or device] float *
				ref float s);  // [host or device] float *

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="s"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotg_v2(
				cublasHandle_t handle,
				ref double a,  // [host or device] double *
				ref double b,  // [host or device] double *
				ref double c,  // [host or device] double *
				ref double s); // [host or device] double *

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="s"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCrotg_v2(
				cublasHandle_t handle,
				ref cuComplex a,  // [host or device] cuComplex *
				ref cuComplex b,  // [host or device] cuComplex *
				ref float c,      // [host or device] float *
				ref cuComplex s); // [host or device] cuComplex *

			/// <summary>
			/// This function constructs the Givens rotation matrix.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="s"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZrotg_v2(
				cublasHandle_t handle,
				ref cuDoubleComplex a,  // [host or device] cuDoubleComplex *
				ref cuDoubleComplex b,  // [host or device] cuDoubleComplex *
				ref double c,           // [host or device] double *
				ref cuDoubleComplex s); // [host or device] cuDoubleComplex *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotm_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] float *
				int incx,
				IntPtr y, // [device] float *
				int incy,
				float[] param);  // [host or device] const float *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotm_v2(
				cublasHandle_t handle,
				int n,
				IntPtr x, // [device] double *
				int incx,
				IntPtr y, // [device] double *
				int incy,
				double[] param);  // [host or device] const double *

			/// <summary>
			/// This function constructs the modified Givens transformation.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="d1"></param>
			/// <param name="d2"></param>
			/// <param name="x1"></param>
			/// <param name="y1"></param>
			/// <param name="param"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotmg_v2(
				cublasHandle_t handle,
				ref float d1,        // [host or device] float *
				ref float d2,        // [host or device] float *
				ref float x1,        // [host or device] float *
				ref float y1,  // [host or device] const float *
				float[] param);    // [host or device] float *

			/// <summary>
			/// This function constructs the modified Givens transformation.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="d1"></param>
			/// <param name="d2"></param>
			/// <param name="x1"></param>
			/// <param name="y1"></param>
			/// <param name="param"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotmg_v2(
				cublasHandle_t handle,
				ref double d1,        // [host or device] double *
				ref double d2,        // [host or device] double *
				ref double x1,        // [host or device] double *
				ref double y1,  // [host or device] const double *
				double[] param);    // [host or device] double *

			// --------------- CUBLAS BLAS2 functions  ----------------

			// GEMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] const float *
				int incx,
				ref float beta,  // [host or device] const float *
				IntPtr y, // [device] float *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] const double *
				int incx,
				ref double beta, // [host or device] const double *
				IntPtr y, // [device] double *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr y, // [device] cuComplex *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			// GBMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgbmv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int kl,
				int ku,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] const float *
				int incx,
				ref float beta, // [host or device] const float *
				IntPtr y, // [device] float *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgbmv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int kl,
				int ku,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] const double *
				int incx,
				ref double beta, // [host or device] const double *
				IntPtr y, // [device] double *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgbmv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int kl,
				int ku,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr y, // [device] cuComplex *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgbmv_v2(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int kl,
				int ku,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			// TRMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] float *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] double *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] cuComplex *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] cuComplex *
				int incx);

			// TBMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] float *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] double *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] cuComplex *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] cuDoubleComplex *
				int incx);

			// TPMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const float *
				IntPtr x, // [device] float *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const double *
				IntPtr x, // [device] double *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const cuComplex *
				IntPtr x, // [device] cuComplex *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const cuDoubleComplex *
				IntPtr x, // [device] cuDoubleComplex *
				int incx);

			// TRSV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] float *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] double *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x,  // [device] cuComplex *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x,  // [device] cuDoubleComplex *
				int incx);

			// TPSV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const float *
				IntPtr x, // [device] float *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const double *
				IntPtr x, // [device] double *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const cuComplex *
				IntPtr x, // [device] cuComplex *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				IntPtr AP, // [device] const cuDoubleComplex *
				IntPtr x, // [device] cuDoubleComplex *
				int incx);

			// TBSV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStbsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] float *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtbsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] double *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtbsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] cuComplex *
				int incx);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtbsv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int n,
				int k,
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] cuDoubleComplex *
				int incx);

			// SYMV/HEMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsymv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] const float *
				int incx,
				ref float beta, // [host or device] const float *
				IntPtr y, // [device] float *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsymv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] const double *
				int incx,
				ref double beta, // [host or device] const double *
				IntPtr y, // [device] double *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsymv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr y, // [device] cuComplex *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsymv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuDoubleComplex alpha,  // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref cuDoubleComplex beta,   // [host or device] const cuDoubleComplex *
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChemv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr y, // [device] cuComplex *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhemv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuDoubleComplex alpha,  // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref cuDoubleComplex beta,   // [host or device] const cuDoubleComplex *
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			// SBMV/HBMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				int k,
				ref float alpha,   // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] const float *
				int incx,
				ref float beta,  // [host or device] const float *
				IntPtr y, // [device] float *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				int k,
				ref double alpha,   // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] const double *
				int incx,
				ref double beta,   // [host or device] const double *
				IntPtr y, // [device] double *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr y, // [device] cuComplex *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhbmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			// SPMV/HPMV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha,  // [host or device] const float *
				IntPtr AP, // [device] const float *
				IntPtr x, // [device] const float *
				int incx,
				ref float beta,   // [host or device] const float *
				IntPtr y, // [device] float *
				int incy);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr AP, // [device] const double *
				IntPtr x, // [device] const double *
				int incx,
				ref double beta,  // [host or device] const double *
				IntPtr y, // [device] double *
				int incy);

			/// <summary>
			/// This function performs the Hermitian packed matrix-vector multiplication.
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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr AP, // [device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr y, // [device] cuComplex *
				int incy);

			/// <summary>
			/// This function performs the Hermitian packed matrix-vector multiplication.
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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpmv_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr AP, // [device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr y, // [device] cuDoubleComplex *
				int incy);

			// GER

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSger_v2(
				cublasHandle_t handle,
				int m,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] const float *
				int incx,
				IntPtr y, // [device] const float *
				int incy,
				IntPtr A, // [device] float *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDger_v2(
				cublasHandle_t handle,
				int m,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] const double *
				int incx,
				IntPtr y, // [device] const double *
				int incy,
				IntPtr A, // [device] double *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeru_v2(
				cublasHandle_t handle,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] const cuComplex *
				int incy,
				IntPtr A, // [device] cuComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgerc_v2(
				cublasHandle_t handle,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] const cuComplex *
				int incy,
				IntPtr A, // [device] cuComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeru_v2(
				cublasHandle_t handle,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] const cuDoubleComplex *
				int incy,
				IntPtr A, // [device] cuDoubleComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgerc_v2(
				cublasHandle_t handle,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] const cuDoubleComplex *
				int incy,
				IntPtr A, // [device] cuDoubleComplex *
				int lda);

			// SYR/HER

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] const float *
				int incx,
				IntPtr A, // [device] float *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] const double *
				int incx,
				IntPtr A, // [device] double *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr A, // [device] cuComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr A, // [device] cuDoubleComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr A,  // [device] cuComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr A, // [device] cuDoubleComplex *
				int lda);

			// SPR/HPR

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] const float *
				int incx,
				IntPtr AP); // [device] float *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] const double *
				int incx,
				IntPtr AP); // [device] double *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr AP); // [device] cuComplex *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpr_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr AP); // [device] cuDoubleComplex *                     

			// SYR2/HER2

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr x, // [device] const float *
				int incx,
				IntPtr y, // [device] const float *
				int incy,
				IntPtr A, // [device] float *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr x, // [device] const double *
				int incx,
				IntPtr y, // [device] const double *
				int incy,
				IntPtr A, // [device] double *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo, int n,
				ref cuComplex alpha,  // [host or device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] const cuComplex *
				int incy,
				IntPtr A, // [device] cuComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuDoubleComplex alpha,  // [host or device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] const cuDoubleComplex *
				int incy,
				IntPtr A, // [device] cuDoubleComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo, int n,
				ref cuComplex alpha,  // [host or device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] const cuComplex *
				int incy,
				IntPtr A, // [device] cuComplex *
				int lda);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuDoubleComplex alpha,  // [host or device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] const cuDoubleComplex *
				int incy,
				IntPtr A, // [device] cuDoubleComplex *
				int lda);

			// SPR2/HPR2

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref float alpha,  // [host or device] const float *
				IntPtr x, // [device] const float *
				int incx,
				IntPtr y, // [device] const float *
				int incy,
				IntPtr AP); // [device] float *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref double alpha,  // [host or device] const double *
				IntPtr x, // [device] const double *
				int incx,
				IntPtr y, // [device] const double *
				int incy,
				IntPtr AP); // [device] double *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr y, // [device] const cuComplex *
				int incy,
				IntPtr AP); // [device] cuComplex *

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpr2_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr y, // [device] const cuDoubleComplex *
				int incy,
				IntPtr AP); // [device] cuDoubleComplex *

			// ---------------- CUBLAS BLAS3 functions ----------------

			// GEMM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemm_v2(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref float alpha, // [host] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr B, // [device] const float *
				int ldb,
				ref float beta, // [host] const float *
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemm_v2(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				IntPtr alpha, // [device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr B, // [device] const float *
				int ldb,
				IntPtr beta, // [device] const float *
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemm_v2(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref double alpha, // [host] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr B, // [device] const double *
				int ldb,
				ref double beta, // [host] const double *
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemm_v2(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				IntPtr alpha, // [device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr B, // [device] const double *
				int ldb,
				IntPtr beta, // [device] const double *
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm_v2(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3m(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <param name="Atype"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="Btype"></param>
			/// <param name="ldb"></param>
			/// <param name="beta"></param>
			/// <param name="C"></param>
			/// <param name="Ctype"></param>
			/// <param name="ldc"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mEx(
				cublasHandle_t handle,
				cublasOperation_t transa, cublasOperation_t transb,
				int m, int n, int k,
				ref cuComplex alpha,
				IntPtr A,
				cudaDataType Atype, 
                int lda,
				IntPtr B,
				cudaDataType Btype, 
                int ldb,
				ref cuComplex beta,
				IntPtr C,
				cudaDataType Ctype, 
                int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemm_v2(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemm3m(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasHgemm(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref __half alpha, // [host or device] const __half *
				IntPtr A, // [device] const __half *
				int lda,
				IntPtr B, // [device] const __half *
				int ldb,
				ref __half beta, // [host or device] const __half *
				IntPtr C, // [device] __half *
				int ldc);

			// IO in FP16/FP32, computation in float

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmEx(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const void *
				cudaDataType Atype,
                int lda,
				IntPtr B, // [device] const void *
				cudaDataType Btype,
                int ldb,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] void *
				cudaDataType Ctype,
                int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGemmEx(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				IntPtr alpha, // [host or device] const void *
				IntPtr A, // [device] const void *
				cudaDataType Atype,
				int lda,
				IntPtr B, // [device] const void *
				cudaDataType Btype,
				int ldb,
				IntPtr beta, // [host or device] const void *
				IntPtr C, // [device] void *
				cudaDataType Ctype,
				int ldc,
				cudaDataType computeType,
				cublasGemmAlgo_t algo);

			// IO in Int8 complex/cuComplex, computation in ref cuComplex /

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmEx(
				cublasHandle_t handle,
				cublasOperation_t transa, cublasOperation_t transb,
				int m, int n, int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const void *
				cudaDataType Atype, 
                int lda,
				IntPtr B, // [device] const void *
				cudaDataType Btype, 
                int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] void *
				cudaDataType Ctype, 
                int ldc);

			/// <summary>
			/// 
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="transa"></param>
			/// <param name="transb"></param>
			/// <param name="transc"></param>
			/// <param name="m"></param>
			/// <param name="n"></param>
			/// <param name="k"></param>
			/// <param name="A"></param>
			/// <param name="A_bias"></param>
			/// <param name="lda"></param>
			/// <param name="B"></param>
			/// <param name="B_bias"></param>
			/// <param name="ldb"></param>
			/// <param name="C"></param>
			/// <param name="C_bias"></param>
			/// <param name="ldc"></param>
			/// <param name="C_mult"></param>
			/// <param name="C_shift"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasUint8gemmBias(
				cublasHandle_t handle,
				cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc,
				int m, int n, int k,
				ref byte A, int A_bias, int lda,
				ref byte B, int B_bias, int ldb,
				ref byte C, int C_bias, int ldc,
				int C_mult, int C_shift);

			// SYRK

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyrk_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyrk_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref double alpha,  // [host or device] const double *
				IntPtr A, // [device] const float *
				int lda,
				ref double beta,  // [host or device] const double *
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrk_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyrk_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr C,  // [device] cuDoubleComplex *
				int ldc);

			// IO in Int8 complex/cuComplex, computation in ref ref cuComplex

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrkEx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				cudaDataType Atype, 
                int lda,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				cudaDataType Ctype, 
                int ldc);

			// IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrk3mEx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const void *
				cudaDataType Atype, 
                int lda,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] void *
				cudaDataType Ctype, 
                int ldc);

			// HERK

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherk_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref float alpha,  // [host or device] const float *
				IntPtr A, // [device] const cuComplex *
				int lda,
				ref float beta,   // [host or device] const float *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZherk_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref double alpha,  // [host or device] const double *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				ref double beta,  // [host or device] const double *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// IO in Int8 complex/cuComplex, computation in ref ref cuComplex

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherkEx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const void *
				cudaDataType Atype,
                int lda,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] void *
				cudaDataType Ctype,
                int ldc);

			// IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherk3mEx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const void *
				cudaDataType Atype, 
                int lda,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] void *
				cudaDataType Ctype, 
                int ldc);

			// SYR2K

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr2k_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr B, // [device] const float *
				int ldb,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr2k_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr B, // [device] const double *
				int ldb,
				ref double beta, // [host or device] const double *
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr2k_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr2k_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuDoubleComplex alpha,  // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref cuDoubleComplex beta,  // [host or device] const cuDoubleComplex *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// HER2K

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher2k_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref float beta,   // [host or device] const float *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher2k_v2(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref double beta, // [host or device] const double *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// SYRKX : eXtended SYRK

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyrkx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr B, // [device] const float *
				int ldb,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyrkx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr B, // [device] const double *
				int ldb,
				ref double beta, // [host or device] const double *
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrkx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C,  // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyrkx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr C,  // [device] cuDoubleComplex *
				int ldc);

			// HERKX : eXtended HERK

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherkx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZherkx(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref double beta, // [host or device] const double *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// SYMM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsymm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				int m,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr B, // [device] const float *
				int ldb,
				ref float beta, // [host or device] const float *
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsymm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				int m,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr B, // [device] const double *
				int ldb,
				ref double beta, // [host or device] const double *
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsymm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsymm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// HEMM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChemm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhemm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// TRSM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr B, // [device] float *
				int ldb);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr B, // [device] double *
				int ldb);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] cuComplex *
				int ldb);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] cuDoubleComplex *
				int ldb);

			// TRMM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrmm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				IntPtr B, // [device] const float *
				int ldb,
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrmm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				IntPtr B, // [device] const double *
				int ldb,
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrmm_v2(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr B, // [device] const cuComplex *
				int ldb,
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrmm_v2(
				cublasHandle_t handle, cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// BATCH GEMM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref float alpha,  // [host or device] const float *
				IntPtr Aarray, // [device] const float *Aarray[]
				int lda,
				IntPtr Barray, // [device] const float *Barray[]
				int ldb,
				ref float beta,   // [host or device] const float *
				IntPtr Carray, // [device] float *Carray[]
				int ldc,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemmBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref double alpha,  // [host or device] const double *
				IntPtr Aarray, // [device] const double *Aarray[]
				int lda,
				IntPtr Barray, // [device] const double *Barray[]
				int ldb,
				ref double beta,  // [host or device] const double *
				IntPtr Carray, // [device] double *Carray[]
				int ldc,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr Aarray, // [device] const cuComplex *Aarray[]
				int lda,
				IntPtr Barray, // [device] const cuComplex *Barray[]
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr Carray, // [device] cuComplex *Carray[]
				int ldc,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr Aarray, // [device] const cuComplex *Aarray[]
				int lda,
				IntPtr Barray, // [device] const cuComplex *Barray[]
				int ldb,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr Carray, // [device] cuComplex *Carray[]
				int ldc,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemmBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr Aarray, // [device] const cuDoubleComplex *Aarray[]
				int lda,
				IntPtr Barray, // [device] const cuDoubleComplex *Barray[]
				int ldb,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr Carray, // [device] cuDoubleComplex *Carray[]
				int ldc,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmStridedBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref float alpha,  // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				long strideA,   // purposely signed
				IntPtr B, // [device] const float *
				int ldb,
				long strideB,
				ref float beta,   // [host or device] const float *
				IntPtr C, // [device] float *
				int ldc,
				long strideC,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemmStridedBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref double alpha,  // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				long strideA,   // purposely signed
				IntPtr B, // [device] const double *
				int ldb,
				long strideB,
				ref double beta,   // [host or device] const double *
				IntPtr C, // [device] double *
				int ldc,
				long strideC,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmStridedBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuComplex alpha,  // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				long strideA,   // purposely signed
				IntPtr B, // [device] const cuComplex *
				int ldb,
				long strideB,
				ref cuComplex beta,   // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc,
				long strideC,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mStridedBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuComplex alpha,  // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				long strideA,   // purposely signed
				IntPtr B, // [device] const cuComplex *
				int ldb,
				long strideB,
				ref cuComplex beta,   // [host or device] const cuComplex *
				IntPtr C, // [device] cuComplex *
				int ldc,
				long strideC,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemmStridedBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref cuDoubleComplex alpha,  // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				long strideA,   // purposely signed
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				long strideB,
				ref cuDoubleComplex beta,   // [host or device] const cuDoubleComplex *
				IntPtr C, // [device] cuDoubleComplex *
				int ldc,
				long strideC,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasHgemmStridedBatched(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				int k,
				ref __half alpha,  // [host or device] const __half *
				IntPtr A, // [device] const __half *
				int lda,
				long strideA,   // purposely signed
				IntPtr B, // [device] const __half *
				int ldb,
				long strideB,
				ref __half beta,   // [host or device] const __half *
				IntPtr C, // [device] __half *
				int ldc,
				long strideC,
				int batchCount);

			// ---------------- CUBLAS BLAS-like extension ----------------
			// GEAM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgeam(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] const float *
				int lda,
				ref float beta, // [host or device] const float *
				IntPtr B, // [device] const float *
				int ldb,
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgeam(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] const double *
				int lda,
				ref double beta, // [host or device] const double *
				IntPtr B, // [device] const double *
				int ldb,
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeam(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] const cuComplex *
				int lda,
				ref cuComplex beta, // [host or device] const cuComplex *
				IntPtr B, // [device] const cuComplex *
				int ldb,
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeam(
				cublasHandle_t handle,
				cublasOperation_t transa,
				cublasOperation_t transb,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
				IntPtr B, // [device] const cuDoubleComplex *
				int ldb,
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// Batched LU - GETRF

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgetrfBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] float *Aarray[]
				int lda,
				IntPtr P,     // [device] int *
				IntPtr info,  // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetrfBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] double *Aarray[]
				int lda,
				IntPtr P, // [device] int *
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgetrfBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] cuComplex *Aarray[]
				int lda,
				IntPtr P, // [device] int *
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgetrfBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] cuDoubleComplex *Aarray[]
				int lda,
				IntPtr P, // [device] int *
				IntPtr info, // [device] int *
				int batchSize);

			// Batched inversion based on LU factorization from getrf

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgetriBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] float *Aarray[]
				int lda,
				IntPtr P, // [device] int *
				IntPtr C, // [device] float *Carray[]
				int ldc,
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetriBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] double *Aarray[]
				int lda,
				IntPtr P, // [device] int *
				IntPtr C, // [device] double *Carray[]
				int ldc,
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgetriBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] cuComplex *Aarray[]
				int lda,
				IntPtr P, // [device] int *
				IntPtr C, // [device] cuComplex *Carray[]
				int ldc,
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgetriBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] cuDoubleComplex *Aarray[]
				int lda,
				IntPtr P, // [device] int *
				IntPtr C, // [device] cuDoubleComplex *Carray[]
				int ldc,
				IntPtr info, // [device] int *
				int batchSize);

			// Batched solver based on LU factorization from getrf

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgetrsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] const float *Aarray[]
				int lda,
				IntPtr devIpiv, // [device] const int *
				IntPtr Barray, // [device] float *Barray[]
				int ldb,
				IntPtr info, // [host] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetrsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] const double *Aarray[]
				int lda,
				IntPtr devIpiv, // [device] const int *
				IntPtr Barray, // [device] double *Barray[]
				int ldb,
				IntPtr info, // [host] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasCgetrsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] const cuComplex *Aarray[]
				int lda,
				IntPtr devIpiv, // [device] const int *
				IntPtr Barray,  // [device] cuComplex *Barray[]
				int ldb,
				IntPtr info, // [host] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasZgetrsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] const cuDoubleComplex *Aarray[]
				int lda,
				IntPtr devIpiv, // [device] const int *
				IntPtr Barray,  // [device] cuDoubleComplex *Barray[]
				int ldb,
				IntPtr info, // [host] int *
				int batchSize);

			// TRSM - Batched Triangular Solver

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsmBatched(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref float alpha, // [host or device] const float *
				IntPtr A, // [device] float *A[]
				int lda,
				IntPtr B, // [device] float *B[]
				int ldb,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsmBatched(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref double alpha, // [host or device] const double *
				IntPtr A, // [device] double *A[]
				int lda,
				IntPtr B, // [device] double *B[]
				int ldb,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsmBatched(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref cuComplex alpha, // [host or device] const cuComplex *
				IntPtr A, // [device] cuComplex *A[]
				int lda,
				IntPtr B, // [device] cuComplex *B[]
				int ldb,
				int batchCount);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsmBatched(
				cublasHandle_t handle,
				cublasSideMode_t side,
				cublasFillMode_t uplo,
				cublasOperation_t trans,
				cublasDiagType_t diag,
				int m,
				int n,
				ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
				IntPtr A, // [device] cuDoubleComplex *A[]
				int lda,
				IntPtr B, // [device] cuDoubleComplex *B[]
				int ldb,
				int batchCount);

			// Batched - MATINV

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSmatinvBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] const float *A[]
				int lda,
				IntPtr Ainv, // [device] float *Ainv[]
				int lda_inv,
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDmatinvBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] const double *A[]
				int lda,
				IntPtr Ainv, // [device] double *Ainv[]
				int lda_inv,
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCmatinvBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] const cuComplex *A[]
				int lda,
				IntPtr Ainv, // [device] cuComplex *Ainv[]
				int lda_inv,
				IntPtr info, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZmatinvBatched(
				cublasHandle_t handle,
				int n,
				IntPtr A, // [device] const cuDoubleComplex *A[]
				int lda,
				IntPtr Ainv, // [device] cuDoubleComplex *Ainv[]
				int lda_inv,
				IntPtr info, // [device] int *
				int batchSize);

			// Batch QR Factorization

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgeqrfBatched(
				cublasHandle_t handle,
				int m,
				int n,
				IntPtr Aarray, // [device] float *Aarray[]
				int lda,
				IntPtr TauArray, // [device] float *TauArray[]
				ref int info, // [host] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgeqrfBatched(
				cublasHandle_t handle,
				int m,
				int n,
				IntPtr Aarray, // [device] double *Aarray[]
				int lda,
				IntPtr TauArray, // [device] double *TauArray[]
				ref int info, // [host] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeqrfBatched(
				cublasHandle_t handle,
				int m,
				int n,
				IntPtr Aarray, // [device] cuComplex *Aarray[]
				int lda,
				IntPtr TauArray, // [device] cuComplex *TauArray[]
				ref int info, // [host] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeqrfBatched(
				cublasHandle_t handle,
				int m,
				int n,
				IntPtr Aarray, // [device] cuDoubleComplex *Aarray[]
				int lda,
				IntPtr TauArray, // [device] cuDoubleComplex *TauArray[]
				ref int info, // [host] int *
				int batchSize);

			// Least Square Min only m >= n and Non-transpose supported

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgelsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] float *Aarray[]
				int lda,
				IntPtr Carray, // [device] float *Carray[]
				int ldc,
				ref int info, // [host] int *
				IntPtr devInfoArray, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgelsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] double *Aarray[]
				int lda,
				IntPtr Carray, // [device] double *Carray[]
				int ldc,
				ref int info, // [host] int *
				IntPtr devInfoArray, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgelsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] cuComplex *Aarray[]
				int lda,
				IntPtr Carray, // [device] cuComplex *Carray[]
				int ldc,
				ref int info, // [host] int *
				IntPtr devInfoArray, // [device] int *
				int batchSize);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgelsBatched(
				cublasHandle_t handle,
				cublasOperation_t trans,
				int m,
				int n,
				int nrhs,
				IntPtr Aarray, // [device] cuComplex *Aarray[]
				int lda,
				IntPtr Carray, // [device] cuComplex *Carray[]
				int ldc,
				ref int info, // [host] int *
				IntPtr devInfoArray, // [device] int *
				int batchSize);

			// DGMM

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSdgmm(
				cublasHandle_t handle,
				cublasSideMode_t mode,
				int m,
				int n,
				IntPtr A, // [device] const float *
				int lda,
				IntPtr x, // [device] const float *
				int incx,
				IntPtr C, // [device] float *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDdgmm(
				cublasHandle_t handle,
				cublasSideMode_t mode,
				int m,
				int n,
				IntPtr A, // [device] const double *
				int lda,
				IntPtr x, // [device] const double *
				int incx,
				IntPtr C, // [device] double *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdgmm(
				cublasHandle_t handle,
				cublasSideMode_t mode,
				int m,
				int n,
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr x, // [device] const cuComplex *
				int incx,
				IntPtr C, // [device] cuComplex *
				int ldc);

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
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdgmm(
				cublasHandle_t handle,
				cublasSideMode_t mode,
				int m,
				int n,
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr x, // [device] const cuDoubleComplex *
				int incx,
				IntPtr C, // [device] cuDoubleComplex *
				int ldc);

			// TPTTR : Triangular Pack format to Triangular format

			/// <summary>
			/// This function performs the conversion from the triangular packed format to the triangular format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpttr(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr AP, // [device] const float *
				IntPtr A, // [device] float *
				int lda);

			/// <summary>
			/// This function performs the conversion from the triangular packed format to the triangular format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpttr(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr AP, // [device] const double *
				IntPtr A, // [device] double *
				int lda);

			/// <summary>
			/// This function performs the conversion from the triangular packed format to the triangular format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpttr(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr AP, // [device] const cuComplex *
				IntPtr A, // [device] cuComplex *
				int lda);

			/// <summary>
			/// This function performs the conversion from the triangular packed format to the triangular format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="AP"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpttr(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr AP, // [device] const cuDoubleComplex *
				IntPtr A, // [device] cuDoubleComplex *
				int lda);

			// TRTTP : Triangular format to Triangular Pack format

			/// <summary>
			/// This function performs the conversion from the triangular format to the triangular packed format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="AP"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrttp(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr A, // [device] const float *
				int lda,
				IntPtr AP); // [device] float *

			/// <summary>
			/// This function performs the conversion from the triangular format to the triangular packed format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="AP"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrttp(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr A, // [device] const double *
				int lda,
				IntPtr AP); // [device] double *

			/// <summary>
			/// This function performs the conversion from the triangular format to the triangular packed format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="AP"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrttp(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr A, // [device] const cuComplex *
				int lda,
				IntPtr AP); // [device] cuComplex *

			/// <summary>
			/// This function performs the conversion from the triangular format to the triangular packed format.
			/// </summary>
			/// <param name="handle"></param>
			/// <param name="uplo"></param>
			/// <param name="n"></param>
			/// <param name="A"></param>
			/// <param name="lda"></param>
			/// <param name="AP"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrttp(
				cublasHandle_t handle,
				cublasFillMode_t uplo,
				int n,
				IntPtr A, // [device] const cuDoubleComplex *
				int lda,
				IntPtr AP); // [device] cuDoubleComplex *    
		}
	}
}
