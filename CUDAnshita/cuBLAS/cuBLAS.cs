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
			// SYMM
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
			// HEMM
			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasChemm(char side, char uplo, int m, int n, cuComplex alpha, ref cuComplex A, int lda, ref cuComplex B, int ldb, cuComplex beta, ref cuComplex C, int ldc);

			[Obsolete]
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void cublasZhemm(char side, char uplo, int m, int n, cuDoubleComplex alpha, ref cuDoubleComplex A, int lda, ref cuDoubleComplex B, int ldb, cuDoubleComplex beta, ref cuDoubleComplex C, int ldc);

			// -----------------------------------------------------------------------
			// TRSM
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
			// TRMM
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
			public static extern cublasStatus_t cublasSetVector(int n, int elemSize,
													IntPtr x, // [host] const void *
													int incx,
													IntPtr y, // [device] void *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetVector(int n, int elemSize,
													IntPtr x, // [device] const void *
													int incx,
													IntPtr y, // [host] void *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
													IntPtr A, // [host] const void *
													int lda,
													IntPtr B, // [device] void *
													int ldb);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize,
													IntPtr A, // [device] const void *
													int lda,
													IntPtr B, // [host] void *
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
													IntPtr x, // [device] const void *
													cudaDataType xType,
													int incx,
													IntPtr result, // [host or device] void *
													cudaDataType resultType,
													cudaDataType executionType);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const float*
													int incx,
													ref float result); // [host] 

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const float*
													int incx,
													IntPtr result); // [device] 

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const double*
													int incx,
													ref double result);  // [host or device] 

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuComplex*
													int incx,
													ref float result);  // [host or device] 

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuDoubleComplex*
													int incx,
													ref double result);  // [host or device] 

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDotEx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDotcEx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSdot_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const float *
													int incx,
													IntPtr y, // [device] const float *
													int incy,
													IntPtr result);  // [host or device] 

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDdot_v2(cublasHandle_t handle,
													int n,
													IntPtr x,
													int incx,
													IntPtr y,
													int incy,
													IntPtr result);  // [host or device] 

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdotu_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] const cuComplex *
													int incy,
													ref cuComplex result);  // [[host or device] ] cuComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdotc_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] const cuComplex *
													int incy,
													ref cuComplex result); // [[host or device] ] cuComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdotu_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr y, // [device] const cuDoubleComplex *
													int incy,
													ref cuDoubleComplex result); // [[host or device] ] cuDoubleComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdotc_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr y, // [device] const cuDoubleComplex *
													int incy,
													ref cuDoubleComplex result); // [[host or device] ] cuDoubleComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScalEx(cublasHandle_t handle,
													int n,
													IntPtr alpha,  // [host or device] const void *
													cudaDataType alphaType,
                                                    IntPtr x, // [device] void *
													cudaDataType xType,
                                                    int incx,
													cudaDataType executionType);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSscal_v2(cublasHandle_t handle,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x,     // [device] float *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDscal_v2(cublasHandle_t handle,
													int n,
													ref double alpha,  // [host or device] const double *
													IntPtr x, // [device] double *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCscal_v2(cublasHandle_t handle,
													int n,
													ref cuComplex alpha, // [host or device] const cuComplex *
													IntPtr x,  // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsscal_v2(cublasHandle_t handle,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZscal_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
													IntPtr x,  // [device] cuDoubleComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdscal_v2(cublasHandle_t handle,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] cuDoubleComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasAxpyEx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] const float *
													int incx,
													IntPtr y, // [device] float *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] const double *
													int incx,
													IntPtr y, // [device] double *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle,
													int n,
													ref cuComplex alpha, // [host or device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] cuComplex *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle,
													int n,
													ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr y, // [device] cuDoubleComplex *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScopy_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const float *
													int incx,
													IntPtr y, // [device] float *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDcopy_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const double *
													int incx,
													IntPtr y, // [device] double *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCcopy_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] cuComplex *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZcopy_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr y, // [device] cuDoubleComplex *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSswap_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] float *
													int incx,
													IntPtr y, // [device] float *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDswap_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] double *
													int incx,
													IntPtr y, // [device] double *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCswap_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] cuComplex *
													int incx,
													IntPtr y, // [device] cuComplex *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZswap_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] cuDoubleComplex *
													int incx,
													IntPtr y, // [device] cuDoubleComplex *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIsamax_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const float *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIdamax_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const double *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIcamax_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuComplex *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIzamax_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIsamin_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const float *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIdamin_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const double *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIcamin_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuComplex *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasIzamin_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													ref int result); // [host or device] int *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSasum_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const float *
													int incx,
													ref float result); // [host or device] float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDasum_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const double *
													int incx,
													ref double result); // [host or device] double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasScasum_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuComplex *
													int incx,
													ref float result); // [host or device] float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDzasum_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													ref double result); // [host or device] double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrot_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] float *
													int incx,
													IntPtr y, // [device] float *
													int incy,
													float[] c,  // [host or device] const float *
													float[] s); // [host or device] const float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrot_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] double *
													int incx,
													IntPtr y, // [device] double *
													int incy,
													double[] c,  // [host or device] const double *
													double[] s); // [host or device] const double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCrot_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] cuComplex *
													int incx,
													IntPtr y, // [device] cuComplex *
													int incy,
													float[] c,      // [host or device] const float *
													cuComplex[] s); // [host or device] const cuComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsrot_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] cuComplex *
													int incx,
													IntPtr y, // [device] cuComplex *
													int incy,
													float[] c,  // [host or device] const float *
													float[] s); // [host or device] const float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZrot_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] cuDoubleComplex *
													int incx,
													IntPtr y, // [device] cuDoubleComplex *
													int incy,
													double[] c,            // [host or device] const double *
													cuDoubleComplex[] s);  // [host or device] const cuDoubleComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdrot_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] cuDoubleComplex *
													int incx,
													IntPtr y, // [device] cuDoubleComplex *
													int incy,
													double[] c,  // [host or device] const double *
													double[] s); // [host or device] const double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotg_v2(cublasHandle_t handle,
													ref float a,   // [host or device] float *
													ref float b,   // [host or device] float *
													ref float c,   // [host or device] float *
													ref float s);  // [host or device] float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotg_v2(cublasHandle_t handle,
													ref double a,  // [host or device] double *
													ref double b,  // [host or device] double *
													ref double c,  // [host or device] double *
													ref double s); // [host or device] double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCrotg_v2(cublasHandle_t handle,
													ref cuComplex a,  // [host or device] cuComplex *
													ref cuComplex b,  // [host or device] cuComplex *
													ref float c,      // [host or device] float *
													ref cuComplex s); // [host or device] cuComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZrotg_v2(cublasHandle_t handle,
													ref cuDoubleComplex a,  // [host or device] cuDoubleComplex *
													ref cuDoubleComplex b,  // [host or device] cuDoubleComplex *
													ref double c,           // [host or device] double *
													ref cuDoubleComplex s); // [host or device] cuDoubleComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotm_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] float *
													int incx,
													IntPtr y, // [device] float *
													int incy,
													float[] param);  // [host or device] const float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotm_v2(cublasHandle_t handle,
													int n,
													IntPtr x, // [device] double *
													int incx,
													IntPtr y, // [device] double *
													int incy,
													double[] param);  // [host or device] const double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle,
													ref float d1,        // [host or device] float *
													ref float d2,        // [host or device] float *
													ref float x1,        // [host or device] float *
													ref float y1,  // [host or device] const float *
													float[] param);    // [host or device] float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle,
													ref double d1,        // [host or device] double *
													ref double d2,        // [host or device] double *
													ref double x1,        // [host or device] double *
													ref double y1,  // [host or device] const double *
													double[] param);    // [host or device] double *

			// --------------- CUBLAS BLAS2 functions  ----------------

			// GEMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemv_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgbmv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgbmv_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const float *
													int lda,
													IntPtr x, // [device] float *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const double *
													int lda,
													IntPtr x, // [device] double *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const cuComplex *
													int lda,
													IntPtr x, // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const cuComplex *
													int lda,
													IntPtr x, // [device] cuComplex *
													int incx);

			// TBMV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													IntPtr A, // [device] const float *
													int lda,
													IntPtr x, // [device] float *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													IntPtr A, // [device] const double *
													int lda,
													IntPtr x, // [device] double *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													IntPtr A, // [device] const cuComplex *
													int lda,
													IntPtr x, // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const float *
													IntPtr x, // [device] float *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const double *
													IntPtr x, // [device] double *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const cuComplex *
													IntPtr x, // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const cuDoubleComplex *
													IntPtr x, // [device] cuDoubleComplex *
													int incx);

			// TRSV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const float *
													int lda,
													IntPtr x, // [device] float *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const double *
													int lda,
													IntPtr x, // [device] double *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const cuComplex *
													int lda,
													IntPtr x,  // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr A, // [device] const cuDoubleComplex *
													int lda,
													IntPtr x,  // [device] cuDoubleComplex *
													int incx);

			// TPSV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const float *
													IntPtr x, // [device] float *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const double *
													IntPtr x, // [device] double *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const cuComplex *
													IntPtr x, // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													IntPtr AP, // [device] const cuDoubleComplex *
													IntPtr x, // [device] cuDoubleComplex *
													int incx);
			// TBSV
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStbsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													IntPtr A, // [device] const float *
													int lda,
													IntPtr x, // [device] float *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													IntPtr A, // [device] const double *
													int lda,
													IntPtr x, // [device] double *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													cublasOperation_t trans,
													cublasDiagType_t diag,
													int n,
													int k,
													IntPtr A, // [device] const cuComplex *
													int lda,
													IntPtr x, // [device] cuComplex *
													int incx);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsymv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsymv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsymv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsymv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChemv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhemv_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChbmv_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhbmv_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha,  // [host or device] const float *
													IntPtr AP, // [device] const float *
													IntPtr x, // [device] const float *
													int incx,
													ref float beta,   // [host or device] const float *
													IntPtr y, // [device] float *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr AP, // [device] const double *
													IntPtr x, // [device] const double *
													int incx,
													ref double beta,  // [host or device] const double *
													IntPtr y, // [device] double *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpmv_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // [host or device] const cuComplex *
													IntPtr AP, // [device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													ref cuComplex beta, // [host or device] const cuComplex *
													IntPtr y, // [device] cuComplex *
													int incy);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpmv_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSger_v2(cublasHandle_t handle,
													int m,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] const float *
													int incx,
													IntPtr y, // [device] const float *
													int incy,
													IntPtr A, // [device] float *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDger_v2(cublasHandle_t handle,
													int m,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] const double *
													int incx,
													IntPtr y, // [device] const double *
													int incy,
													IntPtr A, // [device] double *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeru_v2(cublasHandle_t handle,
													int m,
													int n,
													ref cuComplex alpha, // [host or device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] const cuComplex *
													int incy,
													IntPtr A, // [device] cuComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgerc_v2(cublasHandle_t handle,
													int m,
													int n,
													ref cuComplex alpha, // [host or device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] const cuComplex *
													int incy,
													IntPtr A, // [device] cuComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeru_v2(cublasHandle_t handle,
													int m,
													int n,
													ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr y, // [device] const cuDoubleComplex *
													int incy,
													IntPtr A, // [device] cuDoubleComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgerc_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] const float *
													int incx,
													IntPtr A, // [device] float *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] const double *
													int incx,
													IntPtr A, // [device] double *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // [host or device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr A, // [device] cuComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr A, // [device] cuDoubleComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr A,  // [device] cuComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr A, // [device] cuDoubleComplex *
													int lda);

			// SPR/HPR
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] const float *
													int incx,
													IntPtr AP); // [device] float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] const double *
													int incx,
													IntPtr AP); // [device] double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr AP); // [device] cuComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpr_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr AP); // [device] cuDoubleComplex *                     

			// SYR2/HER2                                    
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha, // [host or device] const float *
													IntPtr x, // [device] const float *
													int incx,
													IntPtr y, // [device] const float *
													int incy,
													IntPtr A, // [device] float *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha, // [host or device] const double *
													IntPtr x, // [device] const double *
													int incx,
													IntPtr y, // [device] const double *
													int incy,
													IntPtr A, // [device] double *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo, int n,
													ref cuComplex alpha,  // [host or device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] const cuComplex *
													int incy,
													IntPtr A, // [device] cuComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuDoubleComplex alpha,  // [host or device] const cuDoubleComplex *
													IntPtr x, // [device] const cuDoubleComplex *
													int incx,
													IntPtr y, // [device] const cuDoubleComplex *
													int incy,
													IntPtr A, // [device] cuDoubleComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo, int n,
													ref cuComplex alpha,  // [host or device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] const cuComplex *
													int incy,
													IntPtr A, // [device] cuComplex *
													int lda);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher2_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSspr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref float alpha,  // [host or device] const float *
													IntPtr x, // [device] const float *
													int incx,
													IntPtr y, // [device] const float *
													int incy,
													IntPtr AP); // [device] float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDspr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref double alpha,  // [host or device] const double *
													IntPtr x, // [device] const double *
													int incx,
													IntPtr y, // [device] const double *
													int incy,
													IntPtr AP); // [device] double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChpr2_v2(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													ref cuComplex alpha, // [host or device] const cuComplex *
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr y, // [device] const cuComplex *
													int incy,
													IntPtr AP); // [device] cuComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhpr2_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3m(cublasHandle_t handle,
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
													ref cuDoubleComplex alpha, // [host or device] const cuDoubleComplex *
													IntPtr A, // [device] const cuDoubleComplex *
													int lda,
													IntPtr B, // [device] const cuDoubleComplex *
													int ldb,
													ref cuDoubleComplex beta, // [host or device] const cuDoubleComplex *
													IntPtr C, // [device] cuDoubleComplex *
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemm3m(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasHgemm(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasGemmEx(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmEx(cublasHandle_t handle,
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
													ref float alpha, // [host or device] const float *
													IntPtr A, // [device] const float *
													int lda,
													ref float beta, // [host or device] const float *
													IntPtr C, // [device] float *
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrkEx(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherk_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZherk_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherkEx(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherk3mEx(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCher2k_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZher2k_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsyrkx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsyrkx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsyrkx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsyrkx(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCherkx(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZherkx(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSsymm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDsymm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCsymm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZsymm_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasChemm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZhemm_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsm_v2(cublasHandle_t handle,
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


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrmm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemmBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle,
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


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgeam(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgeam(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeam(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeam(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] float *Aarray[]
													int lda,
													IntPtr P,     // [device] int *
													IntPtr info,  // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] double *Aarray[]
													int lda,
													IntPtr P, // [device] int *
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] cuComplex *Aarray[]
													int lda,
													IntPtr P, // [device] int *
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] cuDoubleComplex *Aarray[]
													int lda,
													IntPtr P, // [device] int *
													IntPtr info, // [device] int *
													int batchSize);

			// Batched inversion based on LU factorization from getrf
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgetriBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] float *Aarray[]
													int lda,
													IntPtr P, // [device] int *
													IntPtr C, // [device] float *Carray[]
													int ldc,
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetriBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] double *Aarray[]
													int lda,
													IntPtr P, // [device] int *
													IntPtr C, // [device] double *Carray[]
													int ldc,
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgetriBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] cuComplex *Aarray[]
													int lda,
													IntPtr P, // [device] int *
													IntPtr C, // [device] cuComplex *Carray[]
													int ldc,
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgetriBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] cuDoubleComplex *Aarray[]
													int lda,
													IntPtr P, // [device] int *
													IntPtr C, // [device] cuDoubleComplex *Carray[]
													int ldc,
													IntPtr info, // [device] int *
													int batchSize);

			// Batched solver based on LU factorization from getrf

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasSgetrsBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasCgetrsBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t  cublasZgetrsBatched(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrsmBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] const float *A[]
													int lda,
													IntPtr Ainv, // [device] float *Ainv[]
													int lda_inv,
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] const double *A[]
													int lda,
													IntPtr Ainv, // [device] double *Ainv[]
													int lda_inv,
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] const cuComplex *A[]
													int lda,
													IntPtr Ainv, // [device] cuComplex *Ainv[]
													int lda_inv,
													IntPtr info, // [device] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle,
													int n,
													IntPtr A, // [device] const cuDoubleComplex *A[]
													int lda,
													IntPtr Ainv, // [device] cuDoubleComplex *Ainv[]
													int lda_inv,
													IntPtr info, // [device] int *
													int batchSize);

			// Batch QR Factorization
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													IntPtr Aarray, // [device] float *Aarray[]
													int lda,
													IntPtr TauArray, // [device] float *TauArray[]
													ref int info, // [host] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													IntPtr Aarray, // [device] double *Aarray[]
													int lda,
													IntPtr TauArray, // [device] double *TauArray[]
													ref int info, // [host] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													IntPtr Aarray, // [device] cuComplex *Aarray[]
													int lda,
													IntPtr TauArray, // [device] cuComplex *TauArray[]
													ref int info, // [host] int *
													int batchSize);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle,
													int m,
													int n,
													IntPtr Aarray, // [device] cuDoubleComplex *Aarray[]
													int lda,
													IntPtr TauArray, // [device] cuDoubleComplex *TauArray[]
													ref int info, // [host] int *
													int batchSize);

			// Least Square Min only m >= n and Non-transpose supported
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSgelsBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDgelsBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCgelsBatched(cublasHandle_t handle,
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

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZgelsBatched(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasSdgmm(cublasHandle_t handle,
													cublasSideMode_t mode,
													int m,
													int n,
													IntPtr A, // [device] const float *
													int lda,
													IntPtr x, // [device] const float *
													int incx,
													IntPtr C, // [device] float *
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDdgmm(cublasHandle_t handle,
													cublasSideMode_t mode,
													int m,
													int n,
													IntPtr A, // [device] const double *
													int lda,
													IntPtr x, // [device] const double *
													int incx,
													IntPtr C, // [device] double *
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCdgmm(cublasHandle_t handle,
													cublasSideMode_t mode,
													int m,
													int n,
													IntPtr A, // [device] const cuComplex *
													int lda,
													IntPtr x, // [device] const cuComplex *
													int incx,
													IntPtr C, // [device] cuComplex *
													int ldc);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZdgmm(cublasHandle_t handle,
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
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr AP, // [device] const float *
													IntPtr A, // [device] float *
													int lda );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr AP, // [device] const double *
													IntPtr A, // [device] double *
													int lda );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr AP, // [device] const cuComplex *
													IntPtr A, // [device] cuComplex *
													int lda );

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtpttr(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr AP, // [device] const cuDoubleComplex *
													IntPtr A, // [device] cuDoubleComplex *
													int lda );

			// TRTTP : Triangular format to Triangular Pack format
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasStrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr A, // [device] const float *
													int lda,
													IntPtr AP ); // [device] float *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasDtrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr A, // [device] const double *
													int lda,
													IntPtr AP); // [device] double *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasCtrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr A, // [device] const cuComplex *
													int lda,
													IntPtr AP ); // [device] cuComplex *

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern cublasStatus_t cublasZtrttp(cublasHandle_t handle,
													cublasFillMode_t uplo,
													int n,
													IntPtr A, // [device] const cuDoubleComplex *
													int lda,
													IntPtr AP ); // [device] cuDoubleComplex *    
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

		public static void SetVector<T>(T[] x, int incx, IntPtr y, int incy) {
			int n = x.Length;
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * n;
			IntPtr xPointer = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(x, 0, xPointer, x.Length);
			SetVector(n, elemSize, xPointer, incx, y, incy);
			Marshal.FreeHGlobal(xPointer);
		}

		public static void SetVector<T>(T[] x, IntPtr y) {
			SetVector<T>(x, 1, y, 1);
		}

		/// <summary>
		/// 
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

		public static T[] GetVector<T>(int n, IntPtr x) {
			return GetVector<T>(n, x, 1, 1);
		}

		/// <summary>
		/// 
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

		public static void SetMatrix<T>(int rows, int cols, T[] A, int lda, IntPtr B, int ldb) {
			int elemSize = Marshal.SizeOf(typeof(T));
			int byteSize = elemSize * rows * cols;
			IntPtr APointer = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(A, 0, APointer, A.Length);
			SetMatrix(rows, cols, elemSize, APointer, lda, B, ldb);
			Marshal.FreeHGlobal(APointer);
		}

		public static void SetMatrix<T>(int rows, int cols, T[] A, IntPtr B) {
			SetMatrix<T>(rows, cols, A, rows, B, rows);
		}

		/// <summary>
		/// 
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

		public static T[] GetMatrix<T>(int rows, int cols, IntPtr A) {
			return GetMatrix<T>(rows, cols, A, rows, rows);
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

		/// <summary>
		/// 
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

		public static float Snrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0f;
			CheckStatus(API.cublasSnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static void Snrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr result) {
			cublasPointerMode mode = GetPointerMode_v2(handle);
			SetPointerMode_v2(handle, cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE);
			CheckStatus(API.cublasSnrm2_v2(handle, n, x, incx, result));
			SetPointerMode_v2(handle, mode);
		}

		public static double Dnrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0f;
			CheckStatus(API.cublasDnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static float Scnrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0f;
			CheckStatus(API.cublasScnrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static double Dznrm2_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0f;
			CheckStatus(API.cublasDznrm2_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static IntPtr DotEx(cublasHandle_t handle, int n, IntPtr x, cudaDataType xType, int incx, IntPtr y, cudaDataType yType, int incy, cudaDataType resultType, cudaDataType executionType) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasDotEx(
				handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType
			));
			return result;
		}

		public static IntPtr DotcEx(cublasHandle_t handle, int n, IntPtr x, cudaDataType xType, int incx, IntPtr y, cudaDataType yType, int incy, cudaDataType resultType, cudaDataType executionType) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasDotcEx(
				handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType
			));
			return result;
		}

		public static IntPtr Sdot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasSdot_v2(
				handle, n, x, incx, y, incy, result
			));
			return result;
		}

		public static IntPtr Ddot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			IntPtr result = IntPtr.Zero;
			CheckStatus(API.cublasDdot_v2(
				handle, n, x, incx, y, incy, result
			));
			return result;
		}

		public static cuComplex Cdotu_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuComplex result = new cuComplex();
			CheckStatus(API.cublasCdotu_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}
		public static cuComplex Cdotc_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuComplex result = new cuComplex();
			CheckStatus(API.cublasCdotc_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}
		public static cuDoubleComplex Zdotu_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuDoubleComplex result = new cuDoubleComplex();
			CheckStatus(API.cublasZdotu_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}
		public static cuDoubleComplex Zdotc_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			cuDoubleComplex result = new cuDoubleComplex();
			CheckStatus(API.cublasZdotc_v2(
				handle, n, x, incx, y, incy, ref result
			));
			return result;
		}

		public static void ScalEx(cublasHandle_t handle, int n, IntPtr alpha, cudaDataType alphaType, IntPtr x, cudaDataType xType, int incx, cudaDataType executionType) {
			CheckStatus(API.cublasScalEx(
				handle, n, alpha, alphaType, x, xType, incx, executionType
			));
		}

		public static void Sscal_v2(cublasHandle_t handle, int n, float alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasSscal_v2(handle, n, ref alpha, x, incx));
		}
		public static void Dscal_v2(cublasHandle_t handle, int n, double alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasDscal_v2(handle, n, ref alpha, x, incx));
		}
		public static void Cscal_v2(cublasHandle_t handle, int n, cuComplex alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasCscal_v2(handle, n, ref alpha, x, incx));
		}
		public static void Csscal_v2(cublasHandle_t handle, int n, float alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasCsscal_v2(handle, n, ref alpha, x, incx));
		}
		public static void Zscal_v2(cublasHandle_t handle, int n, cuDoubleComplex alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasZscal_v2(handle, n, ref alpha, x, incx));
		}
		public static void Zdscal_v2(cublasHandle_t handle, int n, double alpha, IntPtr x, int incx) {
			CheckStatus(API.cublasZdscal_v2(handle, n, ref alpha, x, incx));
		}

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

		public static void Saxpy_v2(cublasHandle_t handle, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasSaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}
		public static void Daxpy_v2(cublasHandle_t handle, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasDaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}
		public static void Caxpy_v2(cublasHandle_t handle, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasCaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}
		public static void Zaxpy_v2(cublasHandle_t handle, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasZaxpy_v2(handle, n, ref alpha, x, incx, y, incy));
		}

		public static void Scopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasScopy_v2(handle, n, x, incx, y, incy));
		}
		public static void Dcopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasDcopy_v2(handle, n, x, incx, y, incy));
		}
		public static void Ccopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasCcopy_v2(handle, n, x, incx, y, incy));
		}
		public static void Zcopy_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasZcopy_v2(handle, n, x, incx, y, incy));
		}

		public static void Sswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasSswap_v2(handle, n, x, incx, y, incy));
		}
		public static void Dswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasDswap_v2(handle, n, x, incx, y, incy));
		}
		public static void Cswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasCswap_v2(handle, n, x, incx, y, incy));
		}
		public static void Zswap_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy) {
			CheckStatus(API.cublasZswap_v2(handle, n, x, incx, y, incy));
		}

		public static int Isamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIsamax_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static int Idamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIdamax_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static int Icamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIcamax_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static int Izamax_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIzamax_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static int Isamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIsamin_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static int Idamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIdamin_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static int Icamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIcamin_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static int Izamin_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			int result = 0;
			CheckStatus(API.cublasIzamin_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static float Sasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0;
			CheckStatus(API.cublasSasum_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static double Dasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0;
			CheckStatus(API.cublasDasum_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static float Scasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			float result = 0;
			CheckStatus(API.cublasScasum_v2(handle, n, x, incx, ref result));
			return result;
		}
		public static double Dzasum_v2(cublasHandle_t handle, int n, IntPtr x, int incx) {
			double result = 0;
			CheckStatus(API.cublasDzasum_v2(handle, n, x, incx, ref result));
			return result;
		}

		public static void Srot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] c, float[] s) {
			CheckStatus(API.cublasSrot_v2(handle, n, x, incx, y, incy, c, s));
		}
		public static void Drot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] c, double[] s) {
			CheckStatus(API.cublasDrot_v2(handle, n, x, incx, y, incy, c, s));
		}
		public static void Crot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] c, cuComplex[] s) {
			CheckStatus(API.cublasCrot_v2(handle, n, x, incx, y, incy, c, s));
		}
		public static void Csrot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] c, float[] s) {
			CheckStatus(API.cublasCsrot_v2(handle, n, x, incx, y, incy, c, s));
		}
		public static void Zrot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] c, cuDoubleComplex[] s) {
			CheckStatus(API.cublasZrot_v2(handle, n, x, incx, y, incy, c, s));
		}
		public static void Zdrot_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] c, double[] s) {
			CheckStatus(API.cublasZdrot_v2(handle, n, x, incx, y, incy, c, s));
		}

		public static void Srotg_v2(cublasHandle_t handle, float a, float b, float c, float s) {
			CheckStatus(API.cublasSrotg_v2(handle, ref a, ref b, ref c, ref s));
		}
		public static void Drotg_v2(cublasHandle_t handle, double a, double b, double c, double s) {
			CheckStatus(API.cublasDrotg_v2(handle, ref a, ref b, ref c, ref s));
		}
		public static void Crotg_v2(cublasHandle_t handle, cuComplex a, cuComplex b, float c, cuComplex s) {
			CheckStatus(API.cublasCrotg_v2(handle, ref a, ref b, ref c, ref s));
		}
		public static void Zrotg_v2(cublasHandle_t handle, cuDoubleComplex a, cuDoubleComplex b, double c, cuDoubleComplex s) {
			CheckStatus(API.cublasZrotg_v2(handle, ref a, ref b, ref c, ref s));
		}

		public static void Srotm_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, float[] param) {
			CheckStatus(API.cublasSrotm_v2(handle, n, x, incx, y, incy, param));
		}
		public static void Drotm_v2(cublasHandle_t handle, int n, IntPtr x, int incx, IntPtr y, int incy, double[] param) {
			CheckStatus(API.cublasDrotm_v2(handle, n, x, incx, y, incy, param));
		}

		public static void Srotmg_v2(cublasHandle_t handle, float d1, float d2, float x1, float y1, float[] param) {
			CheckStatus(API.cublasSrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param));
		}
		public static void Drotmg_v2(cublasHandle_t handle, double d1, double d2, double x1, double y1, double[] param) {
			CheckStatus(API.cublasDrotmg_v2(handle, ref d1, ref d2, ref x1, ref y1, param));
		}

		public static void Sgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, float alpha, IntPtr A, int lda, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Dgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, double alpha, IntPtr A, int lda, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Cgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasCgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Zgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZgemv_v2(handle, trans, m, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

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

		public static void Strmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}
		public static void Dtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}
		public static void Ctrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}
		public static void Ztrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		public static void Stbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}
		public static void Dtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}
		public static void Ctbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}
		public static void Ztbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		public static void Stpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}
		public static void Dtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}
		public static void Ctpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}
		public static void Ztpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		public static void Strsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}
		public static void Dtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}
		public static void Ctrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}
		public static void Ztrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx));
		}

		public static void Stpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}
		public static void Dtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}
		public static void Ctpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}
		public static void Ztpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, IntPtr AP, IntPtr x, int incx) {
			CheckStatus(API.cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx));
		}

		public static void Stbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}
		public static void Dtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}
		public static void Ctbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}
		public static void Ztbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, IntPtr A, int lda, IntPtr x, int incx) {
			CheckStatus(API.cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx));
		}

		public static void Ssymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr A, int lda, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Dsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr A, int lda, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Csymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasCsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Zsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZsymv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		public static void Chemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasChemv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Zhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZhemv_v2(handle, uplo, n, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		public static void Ssbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, float alpha, IntPtr A, int lda, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSsbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Dsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, double alpha, IntPtr A, int lda, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDsbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Chbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasChbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}
		public static void Zhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZhbmv_v2(handle, uplo, n, k, ref alpha, A, lda, x, incx, ref beta, y, incy));
		}

		public static void Sspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr AP, IntPtr x, int incx, float beta, IntPtr y, int incy) {
			CheckStatus(API.cublasSspmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}
		public static void Dspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr AP, IntPtr x, int incx, double beta, IntPtr y, int incy) {
			CheckStatus(API.cublasDspmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}
		public static void Chpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr AP, IntPtr x, int incx, cuComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasChpmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}
		public static void Zhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr AP, IntPtr x, int incx, cuDoubleComplex beta, IntPtr y, int incy) {
			CheckStatus(API.cublasZhpmv_v2(handle, uplo, n, ref alpha, AP, x, incx, ref beta, y, incy));
		}

		public static void Sger_v2(cublasHandle_t handle, int m, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasSger_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Dger_v2(cublasHandle_t handle, int m, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasDger_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Cgeru_v2(cublasHandle_t handle, int m, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCgeru_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Cgerc_v2(cublasHandle_t handle, int m, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCgerc_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Zgeru_v2(cublasHandle_t handle, int m, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZgeru_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Zgerc_v2(cublasHandle_t handle, int m, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZgerc_v2(handle, m, n, ref alpha, x, incx, y, incy, A, lda));
		}

		public static void Ssyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasSsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}
		public static void Dsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasDsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}
		public static void Csyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasCsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}
		public static void Zsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasZsyr_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		public static void Cher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasCher_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}
		public static void Zher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr A, int lda) {
			CheckStatus(API.cublasZher_v2(handle, uplo, n, ref alpha, x, incx, A, lda));
		}

		public static void Sspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasSspr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}
		public static void Dspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasDspr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}

		public static void Chpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasChpr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}
		public static void Zhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr AP) {
			CheckStatus(API.cublasZhpr_v2(handle, uplo, n, ref alpha, x, incx, AP));
		}

		public static void Ssyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasSsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Dsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasDsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Csyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Zsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZsyr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		public static void Cher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasCher2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}
		public static void Zher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr A, int lda) {
			CheckStatus(API.cublasZher2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, A, lda));
		}

		public static void Sspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, float alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasSspr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}
		public static void Dspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, double alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasDspr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}

		public static void Chpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasChpr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}
		public static void Zhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex alpha, IntPtr x, int incx, IntPtr y, int incy, IntPtr AP) {
			CheckStatus(API.cublasZhpr2_v2(handle, uplo, n, ref alpha, x, incx, y, incy, AP));
		}

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

		public static void Ssyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}
		public static void Dsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}
		public static void Csyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}
		public static void Zsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsyrk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}
		public static void CsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, cudaDataType Atype, int lda, cuComplex beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCsyrkEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}
		public static void Csyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, cudaDataType Atype, int lda, cuComplex beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCsyrk3mEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}

		public static void Cherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCherk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}
		public static void Zherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZherk_v2(handle, uplo, trans, n, k, ref alpha, A, lda, ref beta, C, ldc));
		}
		public static void CherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, cudaDataType Atype, int lda, float beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCherkEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}
		public static void Cherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, cudaDataType Atype, int lda, float beta, IntPtr C, cudaDataType Ctype, int ldc) {
			CheckStatus(API.cublasCherk3mEx(handle, uplo, trans, n, k, ref alpha, A, Atype, lda, ref beta, C, Ctype, ldc));
		}

		public static void Ssyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Dsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Csyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Zsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsyr2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		public static void Cher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCher2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Zher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZher2k_v2(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		public static void Ssyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Dsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Csyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Zsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsyrkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		public static void Cherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCherkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Zherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZherkx(handle, uplo, trans, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		public static void Ssymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, float alpha, IntPtr A, int lda, IntPtr B, int ldb, float beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasSsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Dsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb, double beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasDsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Csymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasCsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Zsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZsymm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Chemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasChemm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}
		public static void Zhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, cuDoubleComplex beta, IntPtr C, int ldc) {
			CheckStatus(API.cublasZhemm_v2(handle, side, uplo, m, n, ref alpha, A, lda, B, ldb, ref beta, C, ldc));
		}

		public static void Strsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, float alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}
		public static void Dtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}
		public static void Ctrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}
		public static void Ztrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb) {
			CheckStatus(API.cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb));
		}

		public static void Strmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, float alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}
		public static void Dtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, double alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}
		public static void Ctrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}
		public static void Ztrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, cuDoubleComplex alpha, IntPtr A, int lda, IntPtr B, int ldb, IntPtr C, int ldc) {
			CheckStatus(API.cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, ref alpha, A, lda, B, ldb, C, ldc));
		}

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

		public static void SgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}
		public static void DgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}
		public static void CgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}
		public static void ZgetrfBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZgetrfBatched(handle, n, A, lda, P, info, batchSize));
		}

		public static void SgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}
		public static void DgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}
		public static void CgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}
		public static void ZgetriBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr P, IntPtr C, int ldc, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize));
		}

		public static void SgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}
		public static void DgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}
		public static void CgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}
		public static void ZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, IntPtr Aarray, int lda, IntPtr devIpiv, IntPtr Barray, int ldb, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize));
		}

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

		public static void SmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasSmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}
		public static void DmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasDmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}
		public static void CmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasCmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}
		public static void ZmatinvBatched(cublasHandle_t handle, int n, IntPtr A, int lda, IntPtr Ainv, int lda_inv, IntPtr info, int batchSize) {
			CheckStatus(API.cublasZmatinvBatched(
				handle, n,
				A, lda,
				Ainv, lda_inv,
				info,
				batchSize
			));
		}

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

		public static void Sdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasSdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}
		public static void Ddgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasDdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}
		public static void Cdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasCdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}
		public static void Zdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc) {
			CheckStatus(API.cublasZdgmm(
				handle, mode,
				m, n,
				A, lda,
				x, incx,
				C, ldc
			));
		}

		public static void Stpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasStpttr(handle, uplo, n, AP, A, lda));
		}
		public static void Dtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasDtpttr(handle, uplo, n, AP, A, lda));
		}
		public static void Ctpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasCtpttr(handle, uplo, n, AP, A, lda));
		}
		public static void Ztpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr AP, IntPtr A, int lda) {
			CheckStatus(API.cublasZtpttr(handle, uplo, n, AP, A, lda));
		}

		public static void Strttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasStrttp(handle, uplo, n, A, lda, AP));
		}
		public static void Dtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasDtrttp(handle, uplo, n, A, lda, AP));
		}
		public static void Ctrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasCtrttp(handle, uplo, n, A, lda, AP));
		}
		public static void Ztrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, IntPtr A, int lda, IntPtr AP) {
			CheckStatus(API.cublasZtrttp(handle, uplo, n, A, lda, AP));
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
	/// </summary>
	/// <remarks>
	/// It is important to point out that if several scalar values are present in the function call, all of them must conform to the same single pointer mode.
	/// The pointer mode can be set and retrieved using cublasSetPointerMode() and cublasGetPointerMode() routines, respectively.
	/// </remarks>
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
