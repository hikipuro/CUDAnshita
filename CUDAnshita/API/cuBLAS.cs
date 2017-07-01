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

			// ----- v1

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasInit();

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasShutdown();

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetError();

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasGetVersion(ref int version);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasAlloc(int n, int elemSize, ref IntPtr devicePtr);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasFree(IntPtr devicePtr);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus cublasSetKernelStream(cudaStream_t stream);

			// ---------------- CUBLAS BLAS1 functions ----------------
			// NRM2
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasSnrm2(int n, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDnrm2(int n, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasScnrm2(int n, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDznrm2(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// DOT
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasSdot(int n, ref float x, int incx, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDdot(int n, ref double x, int incx, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuComplex cublasCdotu(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuComplex cublasCdotc(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuDoubleComplex cublasZdotu(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cuDoubleComplex cublasZdotc(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SCAL
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSscal(int n, float alpha, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDscal(int n, double alpha, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCscal(int n, cuComplex alpha, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZscal(int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsscal(int n, float alpha, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZdscal(int n, double alpha, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// AXPY
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSaxpy(int n, float alpha, ref float x, int incx, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDaxpy(int n, double alpha, ref double x, int incx, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCaxpy(int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZaxpy(int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// COPY
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasScopy(int n, ref float x, int incx, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDcopy(int n, ref double x, int incx, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCcopy(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZcopy(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SWAP
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSswap(int n, ref float x, int incx, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDswap(int n, ref double x, int incx, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCswap(int n, ref cuComplex x, int incx, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZswap(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// AMAX
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIsamax(int n, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIdamax(int n, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIcamax(int n, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIzamax(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// AMIN
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIsamin(int n, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIdamin(int n, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIcamin(int n, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern int cublasIzamin(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// ASUM
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasSasum(int n, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDasum(int n, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern float cublasScasum(int n, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern double cublasDzasum(int n, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// ROT
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrot(int n, ref float x, int incx, ref float y, int incy, float sc, float ss);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrot(int n, ref double x, int incx, ref double y, int incy, double sc, double ss);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCrot(int n, ref cuComplex x, int incx, ref cuComplex y, int incy, float c, cuComplex s);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZrot(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, double sc, cuDoubleComplex cs);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsrot(int n, ref cuComplex x, int incx, ref cuComplex y, int incy, float c, float s);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZdrot(int n, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, double c, double s);

			// -----------------------------------------------------------------------
			// ROTG
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrotg(ref float sa, ref float sb, ref float sc, ref float ss);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrotg(ref double sa, ref double sb, ref double sc, ref double ss);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCrotg(ref cuComplex ca, cuComplex cb, ref float sc, ref cuComplex cs);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZrotg(ref cuDoubleComplex ca, cuDoubleComplex cb, ref double sc, ref cuDoubleComplex cs);

			// -----------------------------------------------------------------------
			// ROTM
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrotm(int n, ref float x, int incx, ref float y, int incy, ref float sparam);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrotm(int n, ref double x, int incx, ref double y, int incy, ref double sparam);

			// -----------------------------------------------------------------------
			// ROTMG
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSrotmg(ref float sd1, ref float sd2, ref float sx1, ref float sy1, ref float sparam);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDrotmg(ref double sd1, ref double sd2, ref double sx1, ref double sy1, ref double sparam);

			// --------------- CUBLAS BLAS2 functions  ----------------
			// GEMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSgemv(char trans, int m, int n, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDgemv(char trans, int m, int n, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgemv(char trans, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgemv(char trans, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// GBMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDgbmv(char trans, int m, int n, int kl, int ku, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgbmv(char trans, int m, int n, int kl, int ku, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgbmv(char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// TRMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrmv(char uplo, char trans, char diag, int n, ref float A, int lda, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrmv(char uplo, char trans, char diag, int n, ref double A, int lda, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrmv(char uplo, char trans, char diag, int n, ref cuComplex A, int lda, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrmv(char uplo, char trans, char diag, int n, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TBMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStbmv(char uplo, char trans, char diag, int n, int k, ref float A, int lda, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtbmv(char uplo, char trans, char diag, int n, int k, ref double A, int lda, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtbmv(char uplo, char trans, char diag, int n, int k, ref cuComplex A, int lda, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtbmv(char uplo, char trans, char diag, int n, int k, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TPMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStpmv(char uplo, char trans, char diag, int n, ref float AP, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtpmv(char uplo, char trans, char diag, int n, ref double AP, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtpmv(char uplo, char trans, char diag, int n, ref cuComplex AP, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtpmv(char uplo, char trans, char diag, int n, ref cuDoubleComplex AP, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TRSV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrsv(char uplo, char trans, char diag, int n, ref float A, int lda, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrsv(char uplo, char trans, char diag, int n, ref double A, int lda, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrsv(char uplo, char trans, char diag, int n, ref cuComplex A, int lda, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrsv(char uplo, char trans, char diag, int n, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TPSV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStpsv(char uplo, char trans, char diag, int n, ref float AP, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtpsv(char uplo, char trans, char diag, int n, ref double AP, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtpsv(char uplo, char trans, char diag, int n, ref cuComplex AP, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtpsv(char uplo, char trans, char diag, int n, ref cuDoubleComplex AP, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// TBSV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStbsv(char uplo, char trans, char diag, int n, int k, ref float A, int lda, ref float x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtbsv(char uplo, char trans, char diag, int n, int k, ref double A, int lda, ref double x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtbsv(char uplo, char trans, char diag, int n, int k, ref cuComplex A, int lda, ref cuComplex x, int incx);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtbsv(char uplo, char trans, char diag, int n, int k, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx);

			// -----------------------------------------------------------------------
			// SYMV/HEMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsymv(char uplo, int n, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsymv(char uplo, int n, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChemv(char uplo, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhemv(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SBMV/HBMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsbmv(char uplo, int n, int k, float alpha, ref float A, int lda, ref float x, int incx, float beta, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsbmv(char uplo, int n, int k, double alpha, ref double A, int lda, ref double x, int incx, double beta, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChbmv(char uplo, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhbmv(char uplo, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// SPMV/HPMV
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSspmv(char uplo, int n, float alpha, ref float AP, ref float x, int incx, float beta, ref float y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDspmv(char uplo, int n, double alpha, ref double AP, ref double x, int incx, double beta, ref double y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChpmv(char uplo, int n, cuComplex alpha, ref cuComplex AP, ref cuComplex x, int incx, cuComplex beta, ref cuComplex y, int incy);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhpmv(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex AP, ref cuDoubleComplex x, int incx, cuDoubleComplex beta, ref cuDoubleComplex y, int incy);

			// -----------------------------------------------------------------------
			// GER
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSger(int m, int n, float alpha, ref float x, int incx, ref float y, int incy, ref float A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDger(int m, int n, double alpha, ref double x, int incx, ref double y, int incy, ref double A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgeru(int m, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgerc(int m, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgeru(int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgerc(int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex A, int lda);

			// -----------------------------------------------------------------------
			// SYR/HER
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyr(char uplo, int n, float alpha, ref float x, int incx, ref float A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyr(char uplo, int n, double alpha, ref double x, int incx, ref double A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCher(char uplo, int n, float alpha, ref cuComplex x, int incx, ref cuComplex A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZher(char uplo, int n, double alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex A, int lda);

			// -----------------------------------------------------------------------
			// SPR/HPR
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSspr(char uplo, int n, float alpha, ref float x, int incx, ref float AP);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDspr(char uplo, int n, double alpha, ref double x, int incx, ref double AP);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChpr(char uplo, int n, float alpha, ref cuComplex x, int incx, ref cuComplex AP);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhpr(char uplo, int n, double alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex AP);

			// -----------------------------------------------------------------------
			// SYR2/HER2
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyr2(char uplo, int n, float alpha, ref float x, int incx, ref float y, int incy, ref float A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyr2(char uplo, int n, double alpha, ref double x, int incx, ref double y, int incy, ref double A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCher2(char uplo, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex A, int lda);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZher2(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex A, int lda);

			// -----------------------------------------------------------------------
			// SPR2/HPR2
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSspr2(char uplo, int n, float alpha, ref float x, int incx, ref float y, int incy, ref float AP);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDspr2(char uplo, int n, double alpha, ref double x, int incx, ref double y, int incy, ref double AP);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChpr2(char uplo, int n, cuComplex alpha, ref cuComplex x, int incx, ref cuComplex y, int incy, ref cuComplex AP);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhpr2(char uplo, int n, cuDoubleComplex alpha, ref cuDoubleComplex x, int incx, ref cuDoubleComplex y, int incy, ref cuDoubleComplex AP);

			// ------------------------BLAS3 Functions -------------------------------
			// GEMM
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, ref float A, int lda, ref float B, int ldb, float beta, ref float C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDgemm(char transa, char transb, int m, int n, int k, double alpha, ref double A, int lda, ref double B, int ldb, double beta, ref double C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCgemm(char transa, char transb, int m, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZgemm(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// SYRK
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyrk(char uplo, char trans, int n, int k, float alpha, ref float A, int lda, float beta, ref float C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyrk(char uplo, char trans, int n, int k, double alpha, ref double A, int lda, double beta, ref double C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsyrk(char uplo, char trans, int n, int k, cuComplex alpha, ref cuComplex A, int lda, cuComplex beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZsyrk(char uplo, char trans, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// HERK
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCherk(char uplo, char trans, int n, int k, float alpha, ref cuComplex A, int lda, float beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZherk(char uplo, char trans, int n, int k, double alpha, ref cuDoubleComplex A, int lda, double beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// SYR2K
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, ref float A, int lda, ref float B, int ldb, float beta, ref float C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, ref double A, int lda, ref double B, int ldb, double beta, ref double C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsyr2k(char uplo, char trans, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZsyr2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -------------------------------------------------------
			// HER2K
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCher2k(char uplo, char trans, int n, int k, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, float beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZher2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, double beta, ref cuDoubleComplex C, int ldc);

			// -----------------------------------------------------------------------
			// SYMM*/
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasSsymm(char side, char uplo, int m, int n, float alpha, ref float A, int lda, ref float B, int ldb, float beta, ref float C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDsymm(char side, char uplo, int m, int n, double alpha, ref double A, int lda, ref double B, int ldb, double beta, ref double C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCsymm(char side, char uplo, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZsymm(char side, char uplo, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -----------------------------------------------------------------------
			// HEMM*/
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChemm(char side, char uplo, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhemm(char side, char uplo, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -----------------------------------------------------------------------
			// TRSM*/
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, ref float A, int lda, ref float B, int ldb);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, ref double A, int lda, ref double B, int ldb);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrsm(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrsm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb);

			// -----------------------------------------------------------------------
			// TRMM*/
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasStrmm(char side, char uplo, char transa, char diag, int m, int n, float alpha, ref float A, int lda, ref float B, int ldb);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasDtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha, ref double A, int lda, ref double B, int ldb);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasCtrmm(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZtrmm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb);

			// ----- v2

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCreate_v2(ref cublasHandle_t handle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, ref int version);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetProperty(libraryPropertyType type, ref int value);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, ref cudaStream_t streamId);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, ref cublasPointerMode_t mode);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, ref cublasAtomicsMode_t mode);
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetVector(int n, int elemSize, IntPtr x,
													int incx, IntPtr devicePtr, int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetVector(int n, int elemSize, IntPtr x,
													int incx, IntPtr y, int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
													IntPtr A, int lda, IntPtr B,
													int ldb);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize,
													IntPtr A, int lda, IntPtr B,
													int ldb);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetVectorAsync(int n, int elemSize,
													IntPtr hostPtr, int incx,
													IntPtr devicePtr, int incy,
													cudaStream_t stream);
			
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetVectorAsync(int n, int elemSize,
													IntPtr devicePtr, int incx,
													IntPtr hostPtr, int incy,
													cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,
													IntPtr A, int lda, IntPtr B,
													int ldb, cudaStream_t stream);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,
													IntPtr A, int lda, IntPtr B,
													int ldb, cudaStream_t stream);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasXerbla(string srName, int info);

			// ---------------- CUBLAS BLAS1 functions ----------------

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasNrm2Ex(cublasHandle_t handle,
													int n,
													IntPtr x,
													cudaDataType xType,
													int incx,
													IntPtr result,
													cudaDataType resultType,
													cudaDataType executionType); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle,
													int n,
													IntPtr x,
													int incx,
													ref float result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle,
													int n,
													double[] x,
													int incx,
													ref double result);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref float result);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref double result);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDotEx(cublasHandle_t handle,
													int n,
													IntPtr x,
													cudaDataType xType, 
													int incx,
													IntPtr y,
													cudaDataType yType,
													int incy,
													IntPtr result,
													cudaDataType resultType,
													cudaDataType executionType);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDotcEx(cublasHandle_t handle,
													int n,
													IntPtr x,
													cudaDataType xType, 
													int incx,
													IntPtr y,
													cudaDataType yType,
													int incy,
													IntPtr result,
													cudaDataType resultType,
													cudaDataType executionType);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSdot_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref float y,
													int incy,
													ref float result);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDdot_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref double y,
													int incy,
													ref double result);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdotu_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref cuComplex result);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdotc_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref cuComplex result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdotu_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref cuDoubleComplex result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdotc_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref cuDoubleComplex result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScalEx(cublasHandle_t handle,
													int n,
													IntPtr alpha,  // host or device pointer
													cudaDataType alphaType,
                                                    IntPtr x,
													cudaDataType xType,
                                                    int incx,
													cudaDataType executionType);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSscal_v2(cublasHandle_t handle,
													int n,
													ref float alpha,  // host or device pointer
													ref float x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDscal_v2(cublasHandle_t handle,
													int n,
													ref double alpha,  // host or device pointer
													ref double x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCscal_v2(cublasHandle_t handle,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex x, 
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsscal_v2(cublasHandle_t handle,
													int n,
													ref float alpha, // host or device pointer
													ref cuComplex x, 
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZscal_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex x, 
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdscal_v2(cublasHandle_t handle,
													int n,
													ref double alpha, // host or device pointer
													ref cuDoubleComplex x, 
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasAxpyEx(cublasHandle_t handle,
													int n,
													IntPtr alpha, // host or device pointer
													cudaDataType alphaType,
													IntPtr x,
													cudaDataType xType,
													int incx,
													IntPtr y,
													cudaDataType yType,
													int incy,
													cudaDataType executiontype);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle,
													int n,
													ref float alpha, // host or device pointer
													ref float x,
													int incx,
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle,
													int n,
													ref double alpha, // host or device pointer
													ref double x,
													int incx,
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex y, 
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y, 
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScopy_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDcopy_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCcopy_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref cuComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZcopy_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSswap_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDswap_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCswap_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZswap_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIsamax_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIdamax_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIcamax_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIzamax_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIsamin_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIdamin_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIcamin_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIzamin_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref int result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSasum_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref float result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDasum_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref double result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScasum_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref float result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDzasum_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref double result); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrot_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref float y,
													int incy,
													ref float c,  // host or device pointer
													ref float s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrot_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref double y,
													int incy,
													ref double c,  // host or device pointer
													ref double s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCrot_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref float c,      // host or device pointer
													ref cuComplex s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsrot_v2(cublasHandle_t handle,
													int n,
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref float c,  // host or device pointer
													ref float s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZrot_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref double c,            // host or device pointer
													ref cuDoubleComplex s);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdrot_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref double c,  // host or device pointer
													ref double s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotg_v2(cublasHandle_t handle,
													ref float a,   // host or device pointer
													ref float b,   // host or device pointer
													ref float c,   // host or device pointer
													ref float s);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotg_v2(cublasHandle_t handle,
													ref double a,  // host or device pointer
													ref double b,  // host or device pointer
													ref double c,  // host or device pointer
													ref double s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCrotg_v2(cublasHandle_t handle,
													ref cuComplex a,  // host or device pointer
													ref cuComplex b,  // host or device pointer
													ref float c,      // host or device pointer
													ref cuComplex s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZrotg_v2(cublasHandle_t handle,
													ref cuDoubleComplex a,  // host or device pointer
													ref cuDoubleComplex b,  // host or device pointer
													ref double c,           // host or device pointer
													ref cuDoubleComplex s); // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotm_v2(cublasHandle_t handle,
													int n,
													ref float x,
													int incx,
													ref float y,
													int incy,
													ref float param);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotm_v2(cublasHandle_t handle,
													int n,
													ref double x,
													int incx,
													ref double y,
													int incy,
													ref double param);  // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle,
													ref float d1,        // host or device pointer
													ref float d2,        // host or device pointer
													ref float x1,        // host or device pointer
													ref float y1,  // host or device pointer
													ref float param);    // host or device pointer

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle,
													ref double d1,        // host or device pointer
													ref double d2,        // host or device pointer
													ref double x1,        // host or device pointer
													ref double y1,  // host or device pointer
													ref double param);    // host or device pointer

			// --------------- CUBLAS BLAS2 functions  ----------------

			// GEMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float x,
													int incx,
													ref float beta,  // host or device pointer
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double x,
													int incx,
													ref double beta, // host or device pointer
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex x,
													int incx,
													ref cuComplex beta, // host or device pointer
													ref cuComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex y,
                                                    int incy);
			// GBMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int kl,
													int ku,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float x,
													int incx,
													ref float beta, // host or device pointer
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int kl,
													int ku,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double x,
													int incx,
													ref double beta, // host or device pointer
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgbmv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int kl,
													int ku,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex x,
													int incx,
													ref cuComplex beta, // host or device pointer
													ref cuComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgbmv_v2(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int kl,
													int ku,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex y,
                                                    int incy);

			// TRMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref float A,
													int lda,
													ref float x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref double A,
													int lda,
													ref double x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuComplex A,
													int lda,
													ref cuComplex x, 
                                                    int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x, 
                                                    int incx);

			// TBMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref float A,
													int lda,
													ref float x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref double A,
													int lda,
													ref double x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref cuComplex A,
													int lda,
													ref cuComplex x, 
                                                    int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x, 
                                                    int incx);

			// TPMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref float AP,
													ref float x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref double AP,
													ref double x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuComplex AP,
													ref cuComplex x, 
                                                    int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuDoubleComplex AP,
													ref cuDoubleComplex x, 
                                                    int incx);

			// TRSV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref float A,
													int lda,
													ref float x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref double A,
													int lda,
													ref double x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuComplex A,
													int lda,
													ref cuComplex x, 
                                                    int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x, 
                                                    int incx);

			// TPSV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref float AP,
													ref float x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref double AP,
													ref double x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuComplex AP,
													ref cuComplex x, 
                                                    int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													ref cuDoubleComplex AP,
													ref cuDoubleComplex x, 
                                                    int incx);
			// TBSV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStbsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref float A,
													int lda,
													ref float x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref double A,
													int lda,
													ref double x,
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref cuComplex A,
													int lda,
													ref cuComplex x, 
                                                    int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x, 
                                                    int incx);

			// SYMV/HEMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsymv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float x,
													int incx,
													ref float beta, // host or device pointer
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsymv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double x,
													int incx,
													ref double beta, // host or device pointer
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsymv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex x,
													int incx,
													ref cuComplex beta, // host or device pointer
													ref cuComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsymv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha,  // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex beta,   // host or device pointer
													ref cuDoubleComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChemv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex x,
													int incx,
													ref cuComplex beta, // host or device pointer
													ref cuComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhemv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha,  // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex beta,   // host or device pointer
													ref cuDoubleComplex y,
                                                    int incy);

			// SBMV/HBMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													int k,
													ref float alpha,   // host or device pointer
													ref float A,
													int lda,
													ref float x,
													int incx,
													ref float beta,  // host or device pointer
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													int k,
													ref double alpha,   // host or device pointer
													ref double A,
													int lda,
													ref double x,
													int incx,
													ref double beta,   // host or device pointer
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex x,
													int incx,
													ref cuComplex beta, // host or device pointer
													ref cuComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex y,
                                                    int incy);

			// SPMV/HPMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha,  // host or device pointer
													ref float AP,
													ref float x,
													int incx,
													ref float beta,   // host or device pointer
													ref float y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // host or device pointer
													ref double AP,
													ref double x,
													int incx,
													ref double beta,  // host or device pointer
													ref double y,
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex AP,
													ref cuComplex x,
													int incx,
													ref cuComplex beta, // host or device pointer
													ref cuComplex y,
                                                    int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex AP,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex y, 
                                                    int incy);

			// GER
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSger_v2(cublasHandle_t handle,
													int m,
													int n,
													ref float alpha, // host or device pointer
													ref float x,
													int incx,
													ref float y,
													int incy,
													ref float A,
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDger_v2(cublasHandle_t handle,
													int m,
													int n,
													ref double alpha, // host or device pointer
													ref double x,
													int incx,
													ref double y,
													int incy,
													ref double A,
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeru_v2(cublasHandle_t handle,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref cuComplex A,
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgerc_v2(cublasHandle_t handle,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref cuComplex A,
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeru_v2(cublasHandle_t handle,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref cuDoubleComplex A,
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgerc_v2(cublasHandle_t handle,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref cuDoubleComplex A,
                                                    int lda);

			// SYR/HER
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // host or device pointer
													ref float x,
													int incx,
													ref float A,
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // host or device pointer
													ref double x,
													int incx,
													ref double A,
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex A, 
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex A, 
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex A, 
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex A, 
                                                    int lda);

			// SPR/HPR
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // host or device pointer
													ref float x,
													int incx,
													ref float AP);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // host or device pointer
													ref double x,
													int incx,
													ref double AP);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex AP);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex AP);                       
    
			// SYR2/HER2                                    
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // host or device pointer
													ref float x,
													int incx,
													ref float y,
													int incy,
													ref float A,
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // host or device pointer
													ref double x,
													int incx,
													ref double y,
													int incy,
													ref double A,
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo, int n,
													ref cuComplex alpha,  // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref cuComplex A, 
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha,  // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref cuDoubleComplex A,
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo, int n,
													ref cuComplex alpha,  // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref cuComplex A, 
                                                    int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha,  // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref cuDoubleComplex A,
                                                    int lda);

			// SPR2/HPR2
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha,  // host or device pointer
													ref float x,
													int incx,
													ref float y,
													int incy,
													ref float AP);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha,  // host or device pointer
													ref double x,
													int incx,
													ref double y,
													int incy,
													ref double AP);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex x,
													int incx,
													ref cuComplex y,
													int incy,
													ref cuComplex AP);
                                     
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex y,
													int incy,
													ref cuDoubleComplex AP); 

			// ---------------- CUBLAS BLAS3 functions ----------------

			// GEMM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float B,
													int ldb,
													ref float beta, // host or device pointer
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double B,
													int ldb,
													ref double beta, // host or device pointer
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm_v2(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3m(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemm_v2(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemm3m(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasHgemm(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref __half alpha, // host or device pointer
													ref __half A,
													int lda,
													ref __half B,
													int ldb,
													ref __half beta, // host or device pointer
													ref __half C,
                                                    int ldc);

			// IO in FP16/FP32, computation in float
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref float alpha, // host or device pointer
													IntPtr A,
													cudaDataType Atype,
                                                    int lda,
													IntPtr B,
													cudaDataType Btype,
                                                    int ldb,
													ref float beta, // host or device pointer
													IntPtr C,
													cudaDataType Ctype,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGemmEx(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													IntPtr alpha, // host or device pointer
													IntPtr A,
													cudaDataType Atype,
                                                    int lda,
													IntPtr B,
													cudaDataType Btype,
                                                    int ldb,
													IntPtr beta, // host or device pointer
													IntPtr C,
													cudaDataType Ctype,
                                                    int ldc,
													cudaDataType computeType,
                                                    cublasGemmAlgo_t algo);

			// IO in Int8 complex/cuComplex, computation in ref cuComplex /
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmEx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasUint8gemmBias(cublasHandle_t handle,
													cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc,
													int m, int n, int k,
													ref byte A, int A_bias, int lda,
													ref byte B, int B_bias, int ldb,
													ref byte C, int C_bias, int ldc,
													int C_mult, int C_shift);

			// SYRK
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float beta, // host or device pointer
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref double alpha,  // host or device pointer
													ref double A,
													int lda,
													ref double beta,  // host or device pointer
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex beta, // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex C, 
                                                    int ldc);

			// IO in Int8 complex/cuComplex, computation in ref ref cuComplex
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrkEx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													IntPtr A,
													cudaDataType Atype, 
                                                    int lda,
													ref cuComplex beta, // host or device pointer
													IntPtr C,
													cudaDataType Ctype, 
                                                    int ldc);

			// IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuComplex alpha,
													IntPtr A,
													cudaDataType Atype, 
                                                    int lda,
													ref cuComplex beta,
													IntPtr C,
													cudaDataType Ctype, 
                                                    int ldc);

			// HERK
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherk_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref float alpha,  // host or device pointer
													ref cuComplex A,
													int lda,
													ref float beta,   // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZherk_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref double alpha,  // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref double beta,  // host or device pointer
													ref cuDoubleComplex C,
                                                    int ldc);

			// IO in Int8 complex/cuComplex, computation in ref ref cuComplex
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherkEx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref float alpha,  // host or device pointer
													IntPtr A,
													cudaDataType Atype,
                                                    int lda,
													ref float beta,   // host or device pointer
													IntPtr C,
													cudaDataType Ctype,
                                                    int ldc);

			// IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherk3mEx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref float alpha,
													IntPtr A, cudaDataType Atype, 
                                                    int lda,
													ref float beta,
													IntPtr C,
													cudaDataType Ctype, 
                                                    int ldc);

			// SYR2K
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float B,
													int ldb,
													ref float beta, // host or device pointer
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double B,
													int ldb,
													ref double beta, // host or device pointer
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuDoubleComplex alpha,  // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex beta,  // host or device pointer
													ref cuDoubleComplex C,
                                                    int ldc);

			// HER2K
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher2k_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref float beta,   // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher2k_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref double beta, // host or device pointer
													ref cuDoubleComplex C,
                                                    int ldc);

			// SYRKX : eXtended SYRK*/
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyrkx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float B,
													int ldb,
													ref float beta, // host or device pointer
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyrkx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double B,
													int ldb,
													ref double beta, // host or device pointer
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrkx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex C, 
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyrkx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex C, 
                                                    int ldc);

			// HERKX : eXtended HERK
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherkx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref float beta, // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZherkx(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref double beta, // host or device pointer
													ref cuDoubleComplex C,
													int ldc);

			// SYMM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsymm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													int m,
													int n,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float B,
													int ldb,
													ref float beta, // host or device pointer
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsymm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													int m,
													int n,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double B,
													int ldb,
													ref double beta, // host or device pointer
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsymm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsymm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex C,
                                                    int ldc);

			// HEMM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChemm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex C, 
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhemm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex C,
                                                    int ldc);

			// TRSM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float B,
													int ldb);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double B,
													int ldb);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
                                                    int ldb);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
                                                    int ldb);

			// TRMM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrmm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float B,
													int ldb,
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double B,
													int ldb,
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex B,
													int ldb,
													ref cuComplex C,
                                                    int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex C,
                                                    int ldc);

			// BATCH GEMM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref float alpha,  // host or device pointer
													ref float[] Aarray,
													int lda,
													ref float[] Barray,
													int ldb,
													ref float beta,   // host or device pointer
													ref float[] Carray,
													int ldc,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref double alpha,  // host or device pointer
													ref double[] Aarray,
													int lda,
													ref double[] Barray,
													int ldb,
													ref double beta,  // host or device pointer
													ref double[] Carray,
													int ldc,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex[] Aarray,
													int lda,
													ref cuComplex[] Barray,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex[] Carray,
                                                    int ldc,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex[] Aarray,
													int lda,
													ref cuComplex[] Barray,
													int ldb,
													ref cuComplex beta, // host or device pointer
													ref cuComplex[] Carray,
                                                    int ldc,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemmBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex[] Aarray,
													int lda,
													ref cuDoubleComplex[] Barray,
													int ldb,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex[] Carray,
                                                    int ldc,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref float alpha,  // host or device pointer
													ref float A,
													int lda,
													long strideA,   // purposely signed
													ref float B,
													int ldb,
													long strideB,
													ref float beta,   // host or device pointer
													ref float C,
													int ldc,
													long strideC,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref double alpha,  // host or device pointer
													ref double A,
													int lda,
													long strideA,   // purposely signed
													ref double B,
													int ldb,
													long strideB,
													ref double beta,   // host or device pointer
													ref double C,
													int ldc,
													long strideC,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuComplex alpha,  // host or device pointer
													ref cuComplex A,
													int lda,
													long strideA,   // purposely signed
													ref cuComplex B,
													int ldb,
													long strideB,
													ref cuComplex beta,   // host or device pointer
													ref cuComplex C,
                                                    int ldc,
													long strideC,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuComplex alpha,  // host or device pointer
													ref cuComplex A,
													int lda,
													long strideA,   // purposely signed
													ref cuComplex B,
													int ldb,
													long strideB,
													ref cuComplex beta,   // host or device pointer
													ref cuComplex C,
                                                    int ldc,
													long strideC,
													int batchCount);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref cuDoubleComplex alpha,  // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													long strideA,   // purposely signed
													ref cuDoubleComplex B,
													int ldb,
													long strideB,
													ref cuDoubleComplex beta,   // host or device poi
													ref cuDoubleComplex C,
                                                    int ldc,
													long strideC,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													int k,
													ref __half alpha,  // host or device pointer
													ref __half A,
													int lda,
													long strideA,   // purposely signed
													ref __half B,
													int ldb,
													long strideB,
													ref __half beta,   // host or device pointer
													ref __half C,
                                                    int ldc,
													long strideC,
													int batchCount);

			// ---------------- CUBLAS BLAS-like extension ----------------
			// GEAM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgeam(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													ref float alpha, // host or device pointer
													ref float A,
													int lda,
													ref float beta, // host or device pointer
													ref float B,
													int ldb,
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgeam(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													ref double alpha, // host or device pointer
													ref double A,
													int lda,
													ref double beta, // host or device pointer
													ref double B,
													int ldb,
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeam(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													ref cuComplex alpha, // host or device pointer
													ref cuComplex A,
													int lda,
													ref cuComplex beta, // host or device pointer
													ref cuComplex B,
													int ldb,
													ref cuComplex C, 
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeam(cublasHandle_t handle,
													cublasOperation_t transa,
													cublasOperation_t transb,
													int m,
													int n,
													ref cuDoubleComplex alpha, // host or device pointer
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex beta, // host or device pointer
													ref cuDoubleComplex B,
													int ldb,
													ref cuDoubleComplex C, 
													int ldc);

			// Batched LU - GETRF*/
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
													int n,
													ref float[] A,                      /*Device pointer*/
													int lda,
													ref int P,                          /*Device Pointer*/
													ref int info,                       /*Device Pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle,
													int n,
													ref double[] A,                     /*Device pointer*/
													int lda,
													ref int P,                          /*Device Pointer*/
													ref int info,                       /*Device Pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle,
													int n,
													ref cuComplex[] A,                 /*Device pointer*/
													int lda,
													ref int P,                         /*Device Pointer*/
													ref int info,                      /*Device Pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex[] A,           /*Device pointer*/
													int lda,
													ref int P,                         /*Device Pointer*/
													ref int info,                      /*Device Pointer*/
													int batchSize);

			// Batched inversion based on LU factorization from getrf
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgetriBatched(cublasHandle_t handle,
													int n,
													ref float[] A,               /*Device pointer*/
													int lda,
													ref int P,                   /*Device pointer*/
													ref float[] C,                     /*Device pointer*/
													int ldc,
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetriBatched(cublasHandle_t handle,
													int n,
													ref double[] A,              /*Device pointer*/
													int lda,
													ref int P,                   /*Device pointer*/
													ref double[] C,                    /*Device pointer*/
													int ldc,
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgetriBatched(cublasHandle_t handle,
													int n,
													ref cuComplex[] A,            /*Device pointer*/
													int lda,
													ref int P,                   /*Device pointer*/
													ref cuComplex[] C,                 /*Device pointer*/
													int ldc,
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgetriBatched(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex[] A,     /*Device pointer*/
													int lda,
													ref int P,                   /*Device pointer*/
													ref cuDoubleComplex[] C,           /*Device pointer*/
													int ldc,
													ref int info,
													int batchSize);

			// Batched solver based on LU factorization from getrf

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasSgetrsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int n,
													int nrhs,
													ref float[] Aarray,
													int lda,
													ref int devIpiv,
													ref float[] Barray,
													int ldb,
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int n,
													int nrhs,
													ref double[] Aarray,
													int lda,
													ref int devIpiv,
													ref double[] Barray,
													int ldb,
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasCgetrsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int n,
													int nrhs,
													ref cuComplex[] Aarray,
													int lda,
													ref int devIpiv,
													ref cuComplex[] Barray, 
                                                    int ldb,
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasZgetrsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int n,
													int nrhs,
													ref cuDoubleComplex[] Aarray,
													int lda,
													ref int devIpiv,
													ref cuDoubleComplex[] Barray, 
                                                    int ldb,
													ref int info,
													int batchSize);

			// TRSM - Batched Triangular Solver
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsmBatched(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref float alpha,           /*Host or Device Pointer*/
													ref float[] A,
													int lda,
													ref float[] B,
													int ldb,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref double alpha,          /*Host or Device Pointer*/
													ref double[] A,
													int lda,
													ref double[] B,
													int ldb,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref cuComplex alpha,       /*Host or Device Pointer*/
													ref cuComplex[] A,
													int lda,
													ref cuComplex[] B, 
                                                    int ldb,
													int batchCount);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle,
													cublasSideMode_t side,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int m,
													int n,
													ref cuDoubleComplex alpha, /*Host or Device Pointer*/
													ref cuDoubleComplex[] A,
													int lda,
													ref cuDoubleComplex[] B, 
                                                    int ldb,
													int batchCount);

			// Batched - MATINV*/
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle,
													int n,
													ref float[] A,                  /*Device pointer*/
													int lda,
													ref float[] Ainv,               /*Device pointer*/
													int lda_inv,
													ref int info,                   /*Device Pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle,
													int n,
													ref double[] A,                 /*Device pointer*/
													int lda,
													ref double[] Ainv,              /*Device pointer*/
													int lda_inv,
													ref int info,                   /*Device Pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle,
													int n,
													ref cuComplex[] A,              /*Device pointer*/
													int lda,
													ref cuComplex[] Ainv,           /*Device pointer*/
                                                    int lda_inv,
													ref int info,                   /*Device Pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex[] A,        /*Device pointer*/
													int lda,
													ref cuDoubleComplex[] Ainv,     /*Device pointer*/
                                                    int lda_inv,
													ref int info,                   /*Device Pointer*/
													int batchSize);

			// Batch QR Factorization
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													ref float[] Aarray,           /*Device pointer*/
													int lda,
													ref float[] TauArray,        // Device pointer*/
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													ref double[] Aarray,           /*Device pointer*/
													int lda,
													ref double[] TauArray,        // Device pointer*/
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													ref cuComplex[] Aarray,           /*Device pointer*/
													int lda,
													ref cuComplex[] TauArray,        // Device pointer*/
													ref int info,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													ref cuDoubleComplex[] Aarray,           /*Device pointer*/
													int lda,
													ref cuDoubleComplex[] TauArray,        // Device pointer*/
													ref int info,
													int batchSize);

			// Least Square Min only m >= n and Non-transpose supported
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgelsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int nrhs,
													ref float[] Aarray, /*Device pointer*/
													int lda,
													ref float[] Carray, // Device pointer*/
													int ldc,
													ref int info,
													ref int devInfoArray, // Device pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgelsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int nrhs,
													ref double[] Aarray, /*Device pointer*/
													int lda,
													ref double[] Carray, // Device pointer*/
													int ldc,
													ref int info,
													ref int devInfoArray, // Device pointer*/
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgelsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int nrhs,
													ref cuComplex[] Aarray, /*Device pointer*/
													int lda,
													ref cuComplex[] Carray, // Device pointer*/
													int ldc,
													ref int info,
													ref int devInfoArray,
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgelsBatched(cublasHandle_t handle,
													cublasOperation_t trans,
													int m,
													int n,
													int nrhs,
													ref cuDoubleComplex[] Aarray, /*Device pointer*/
													int lda,
													ref cuDoubleComplex[] Carray, // Device pointer*/
													int ldc,
													ref int info,
													ref int devInfoArray,
													int batchSize);
			// DGMM
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSdgmm(cublasHandle_t handle,
													cublasSideMode_t mode,
													int m,
													int n,
													ref float A,
													int lda,
													ref float x,
													int incx,
													ref float C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDdgmm(cublasHandle_t handle,
													cublasSideMode_t mode,
													int m,
													int n,
													ref double A,
													int lda,
													ref double x,
													int incx,
													ref double C,
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdgmm(cublasHandle_t handle,
													cublasSideMode_t mode,
													int m,
													int n,
													ref cuComplex A,
													int lda,
													ref cuComplex x,
													int incx,
													ref cuComplex C, 
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdgmm(cublasHandle_t handle,
													cublasSideMode_t mode,
													int m,
													int n,
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex x,
													int incx,
													ref cuDoubleComplex C, 
													int ldc);

			// TPTTR : Triangular Pack format to Triangular format
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float AP,
													ref float A,
													int lda );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double AP,
													ref double A,
													int lda );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex AP,
													ref cuComplex A,  
                                                    int lda );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex AP,
													ref cuDoubleComplex A,  
                                                    int lda );

			// TRTTP : Triangular format to Triangular Pack format
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float A,
													int lda,
													ref float AP );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double A,
													int lda,
													ref double AP);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex A,
													int lda,
													ref cuComplex AP );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex A,
													int lda,
													ref cuDoubleComplex AP );     
		}

		// ----- C# Interface

		public static cublasHandle_t Create_v2() {
			cublasHandle_t handle = IntPtr.Zero;
			CheckStatus(API.cublasCreate_v2(ref handle));
			return handle;
		}

		public static void Destroy_v2(cublasHandle_t handle) {
			CheckStatus(API.cublasDestroy_v2(handle));
		}

		public static int GetVersion_v2(cublasHandle_t handle) {
			int version = 0;
			CheckStatus(API.cublasGetVersion_v2(handle, ref version));
			return version;
		}

		public static int GetProperty(libraryPropertyType type) {
			int value = 0;
			CheckStatus(API.cublasGetProperty(type, ref value));
			return value;
		}

		public static void SetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
			CheckStatus(API.cublasSetStream_v2(handle, streamId));
		}

		public static cudaStream_t GetStream_v2(cublasHandle_t handle) {
			cudaStream_t streamId = IntPtr.Zero;
			CheckStatus(API.cublasGetStream_v2(handle, ref streamId));
			return streamId;
		}

		public static cublasPointerMode_t GetPointerMode_v2(cublasHandle_t handle) {
			cublasPointerMode_t mode = cublasPointerMode_t.CUBLAS_POINTER_MODE_HOST;
			CheckStatus(API.cublasGetPointerMode_v2(handle, ref mode));
			return mode;
		}

		public static void SetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
			CheckStatus(API.cublasSetPointerMode_v2(handle, mode));
		}

		public static cublasAtomicsMode_t GetAtomicsMode(cublasHandle_t handle) {
			cublasAtomicsMode_t mode = cublasAtomicsMode_t.CUBLAS_ATOMICS_NOT_ALLOWED;
			CheckStatus(API.cublasGetAtomicsMode(handle, ref mode));
			return mode;
		}

		public static void SetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
			CheckStatus(API.cublasSetAtomicsMode(handle, mode));
		}

		public static void SetVector(int n, int elemSize, IntPtr x, int incx, IntPtr devicePtr, int incy) {
			CheckStatus(API.cublasSetVector(n, elemSize, x, incx, devicePtr, incy));
		}

		public static void GetVector(int n, int elemSize, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasGetVector(n, elemSize, x, incx, y, incy));
		}

		public static void SetMatrix(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}

		public static void GetMatrix(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}

		public static void SetVectorAsync(int n, int elemSize, IntPtr hostPtr, int incx, IntPtr devicePtr, int incy, cudaStream_t stream) {
			CheckStatus(API.cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream));
		}

		public static void GetVectorAsync(int n, int elemSize, IntPtr devicePtr, int incx, IntPtr hostPtr, int incy, cudaStream_t stream) {
			CheckStatus(API.cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream));
		}

		public static void SetMatrixAsync(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb, cudaStream_t stream) {
			CheckStatus(API.cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
		}

		public static void GetMatrixAsync(int rows, int cols, int elemSize, IntPtr A, int lda, IntPtr B, int ldb, cudaStream_t stream) {
			CheckStatus(API.cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
		}

		public static void Xerbla(string srName, int info) {
			API.cublasXerbla(srName, info);
		}

		public static void Nrm2Ex(cublasHandle_t handle, int n, IntPtr x, cudaDataType xType, int incx, IntPtr result, cudaDataType resultType, cudaDataType executionType) {
			CheckStatus(API.cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType));
		}

		public static float Snrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0f;
			CheckStatus(API.cublasSnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static double Dnrm2_v2(cublasHandle_t handle, int n, double[] x, int incx) {
			double result = 0f;
			CheckStatus(API.cublasDnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

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

		/*
		public void SetMatrix(int rows, int cols, double[] A, int lda, IntPtr B, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(double));
			CheckStatus(API.cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}

		public void SetMatrix(int rows, int cols, float[] A, int lda, IntPtr B, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(float));
			CheckStatus(API.cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
		}
		*/

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

		/*
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
		*/

		/*
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
		*/


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
				throw new CudaException(status.ToString());
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
