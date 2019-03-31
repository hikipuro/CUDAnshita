using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using fftw_plan = IntPtr;
	using fftwf_plan = IntPtr;
	using fftw_complex = double2;
	using fftwf_complex = float2;
	using fftwf_iodim = fftw_iodim;
	using size_t = Int64;

	/// <summary>
	/// NVIDIA CUDA FFTW library (CUFFTW)
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/cufft/">http://docs.nvidia.com/cuda/cufft/</a>
	/// </remarks>
	public class cuFFTW {
		/// <summary>
		/// cuFFTW DLL functions.
		/// </summary>
		public class API {
			//const string DLL_PATH = "cufftw64_80.dll";
			const string DLL_PATH = "cufftw64_10.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_1d(int n,
									ref fftw_complex _in,
									ref fftw_complex _out,
									int sign,
									uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_2d(int n0,
												int n1,
												ref fftw_complex _in,
												ref fftw_complex _out,
												int sign,
												uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_3d(int n0,
												int n1,
												int n2,
												ref fftw_complex _in,
												ref fftw_complex _out,
												int sign,
												uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft(int rank,
								 ref int n,
								 ref fftw_complex _in,
								 ref fftw_complex _out,
								 int sign,
								 uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_r2c_1d(int n,
										ref double _in,
										ref fftw_complex _out,
										uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_r2c_2d(int n0,
													int n1,
													ref double _in,
													ref fftw_complex _out,
													uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_r2c_3d(int n0,
													int n1,
													int n2,
													ref double _in,
													ref fftw_complex _out,
													uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_r2c(int rank,
									 ref int n,
									 ref double _in,
                                     ref fftw_complex _out, 
                                     uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_c2r_1d(int n,
													ref fftw_complex _in,
													ref double _out,
													uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_c2r_2d(int n0,
													int n1,
													ref fftw_complex _in,
													ref double _out,
													uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_c2r_3d(int n0,
													int n1,
													int n2,
													ref fftw_complex _in,
													ref double _out,
													uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_dft_c2r(int rank,
									 ref int n,
									 ref fftw_complex _in,
									 ref double _out, 
                                     uint flags);


			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_many_dft(int rank,
									  ref int n,
									  int batch,
									  ref fftw_complex _in,
									  ref int inembed, int istride, int idist,
									  ref fftw_complex _out,
									  ref int onembed, int ostride, int odist,
									  int sign, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_many_dft_r2c(int rank,
										  ref int n,
										  int batch,
										  ref double _in,
                                          ref int inembed, int istride, int idist,
										  ref fftw_complex _out,
										  ref int onembed, int ostride, int odist,
										  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_many_dft_c2r(int rank,
										  ref int n,
										  int batch,
										  ref fftw_complex _in,
										  ref int inembed, int istride, int idist,
										  ref double _out,
                                          ref int onembed, int ostride, int odist,
										  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_guru_dft(int rank, ref fftw_iodim dims,
									  int batch_rank, ref fftw_iodim batch_dims,
									  ref fftw_complex _in, ref fftw_complex _out,
									  int sign, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_guru_dft_r2c(int rank, ref fftw_iodim dims,
										  int batch_rank, ref fftw_iodim batch_dims,
										  ref double _in, ref fftw_complex _out, 
                                          uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftw_plan fftw_plan_guru_dft_c2r(int rank, ref fftw_iodim dims,
										  int batch_rank, ref fftw_iodim batch_dims,
										  ref fftw_complex _in, ref double _out, 
                                          uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftw_execute(fftw_plan plan);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftw_execute_dft(fftw_plan plan,
										   ref fftw_complex idata,
										   ref fftw_complex odata);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftw_execute_dft_r2c(fftw_plan plan,
								   ref double idata,
								   ref fftw_complex odata);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftw_execute_dft_c2r(fftw_plan plan,
								   ref fftw_complex idata,
                                   ref double odata);


			// CUFFTW defines and supports the following single precision APIs

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_1d(int n,
												  ref fftwf_complex _in,
												  ref fftwf_complex _out,
												  int sign,
												  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_2d(int n0,
												  int n1,
												  ref fftwf_complex _in,
												  ref fftwf_complex _out,
												  int sign,
												  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_3d(int n0,
												  int n1,
												  int n2,
												  ref fftwf_complex _in,
												  ref fftwf_complex _out,
												  int sign,
												  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft(int rank,
								   ref int n,
								   ref fftwf_complex _in,
								   ref fftwf_complex _out,
								   int sign,
								   uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_r2c_1d(int n,
										  ref float _in,
										  ref fftwf_complex _out,
										  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_r2c_2d(int n0,
													  int n1,
													  ref float _in,
													  ref fftwf_complex _out,
													  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_r2c_3d(int n0,
													  int n1,
													  int n2,
													  ref float _in,
													  ref fftwf_complex _out,
													  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_r2c(int rank,
									   ref int n,
									   ref float _in,
                                       ref fftwf_complex _out, 
                                       uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_c2r_1d(int n,
													  ref fftwf_complex _in,
													  ref float _out,
													  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_c2r_2d(int n0,
													  int n1,
													  ref fftwf_complex _in,
													  ref float _out,
													  uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_c2r_3d(int n0,
													int n1,
													int n2,
													ref fftwf_complex _in,
													ref float _out,
													uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_dft_c2r(int rank,
									   ref int n,
									   ref fftwf_complex _in,
									   ref float _out, 
                                       uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_many_dft(int rank,
										ref int n,
										int batch,
										ref fftwf_complex _in,
										ref int inembed, int istride, int idist,
										ref fftwf_complex _out,
										ref int onembed, int ostride, int odist,
										int sign, uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_many_dft_r2c(int rank,
											ref int n,
											int batch,
											ref float _in,
                                            ref int inembed, int istride, int idist,
											ref fftwf_complex _out,
											ref int onembed, int ostride, int odist,
											uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_many_dft_c2r(int rank,
											ref int n,
											int batch,
											ref fftwf_complex _in,
											ref int inembed, int istride, int idist,
											ref float _out,
                                            ref int onembed, int ostride, int odist,
											uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_guru_dft(int rank, ref fftwf_iodim dims,
										int batch_rank, ref fftwf_iodim batch_dims,
										ref fftwf_complex  _in, ref fftwf_complex  _out,
										int sign, uint flags);
                                        
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_guru_dft_r2c(int rank, ref fftwf_iodim dims,

											int batch_rank, ref fftwf_iodim batch_dims,

											ref float _in, ref fftwf_complex  _out, 
                                            uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern fftwf_plan fftwf_plan_guru_dft_c2r(int rank, ref fftwf_iodim dims,

											int batch_rank, ref fftwf_iodim batch_dims,
											ref fftwf_complex  _in, ref float _out, 
                                            uint flags);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftwf_execute(fftw_plan plan);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftwf_execute_dft(fftwf_plan plan,
											ref fftwf_complex idata,
                                ref fftwf_complex  odata);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftwf_execute_dft_r2c(fftwf_plan plan,

									ref float idata,
									ref fftwf_complex odata);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern void fftwf_execute_dft_c2r(fftwf_plan plan,
									ref fftwf_complex idata,
                                    ref float odata);

		}

		/// <summary>(cuFFTW) transform direction.</summary>
		public const int FFTW_FORWARD = -1;
		/// <summary>(cuFFTW) transform direction.</summary>
		public const int FFTW_INVERSE = 1;
		/// <summary>(cuFFTW) transform direction.</summary>
		public const int FFTW_BACKWARD = 1;

		/// <summary>(cuFFTW) Planner flags.</summary>
		public const int FFTW_ESTIMATE = 0x01;
		/// <summary>(cuFFTW) Planner flags.</summary>
		public const int FFTW_MEASURE = 0x02;
		/// <summary>(cuFFTW) Planner flags.</summary>
		public const int FFTW_PATIENT = 0x03;
		/// <summary>(cuFFTW) Planner flags.</summary>
		public const int FFTW_EXHAUSTIVE = 0x04;
		/// <summary>(cuFFTW) Planner flags.</summary>
		public const int FFTW_WISDOM_ONLY = 0x05;

		/// <summary>(cuFFTW) Algorithm restriction flags.</summary>
		public const int FFTW_DESTROY_INPUT = 0x08;
		/// <summary>(cuFFTW) Algorithm restriction flags.</summary>
		public const int FFTW_PRESERVE_INPUT = 0x0C;
		/// <summary>(cuFFTW) Algorithm restriction flags.</summary>
		public const int FFTW_UNALIGNED = 0x10;

	}

	/// <summary>
	/// (cuFFTW) 
	/// </summary>
	public struct fftw_iodim {
		public int n;
		public int _is;
		public int os;
	}

	/// <summary>
	/// (cuFFTW) 
	/// </summary>
	public struct fftw_iodim64 {
		public int n;
		public size_t _is;
		public size_t os;
	}
}
