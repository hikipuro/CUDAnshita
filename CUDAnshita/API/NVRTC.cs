using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CUDAnshita.API {
	/// <summary>
	/// http://docs.nvidia.com/cuda/nvrtc/
	/// </summary>
	public class NVRTC {
		const string DLL_PATH = "nvrtc64_75.dll";
		const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
		const CharSet CHAR_SET = CharSet.Ansi;

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern IntPtr nvrtcGetErrorString(nvrtcResult result);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcVersion(ref int major, ref int minor);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
		static extern nvrtcResult nvrtcCompileProgram(
			IntPtr prog,
			int numOptions,
			[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
			string[] options);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
		static extern nvrtcResult nvrtcCreateProgram(
			ref IntPtr prog,
			string src,
			string name,
			int numHeaders,
			[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
			string[] headers,
			[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
			string[] includeNames);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
		static extern nvrtcResult nvrtcGetPTX(IntPtr prog, StringBuilder ptx);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcGetPTXSize(IntPtr prog, ref long ptxSizeRet);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
		static extern nvrtcResult nvrtcGetProgramLog(IntPtr prog, StringBuilder log);

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, ref long logSizeRet);

		public static string GetVersion() {
			int major = 0;
			int minor = 0;
			nvrtcResult result = nvrtcVersion(ref major, ref minor);
			if (result != nvrtcResult.NVRTC_SUCCESS) {
				throw new Exception(result.ToString());
			}
			return String.Format("{0}.{1}", major, minor);
		}

		public static string GetErrorString(nvrtcResult result) {
			IntPtr ptr = nvrtcGetErrorString(result);
			return Marshal.PtrToStringAnsi(ptr);
		}

		public class Program : IDisposable {
			public const string OPTION_TARGET_20 = "--gpu-architecture=compute_20";
			public const string OPTION_TARGET_30 = "--gpu-architecture=compute_30";
			public const string OPTION_TARGET_35 = "--gpu-architecture=compute_35";
			public const string OPTION_TARGET_50 = "--gpu-architecture=compute_50";
			public const string OPTION_TARGET_52 = "--gpu-architecture=compute_52";
			public const string OPTION_TARGET_53 = "--gpu-architecture=compute_53";

			public const string OPTION_DEVICE_C = "--device-c";
			public const string OPTION_DEVICE_W = "--device-w";
			public const string OPTION_DEVICE_CODE_TRUE = "--relocatable-device-code=true";
			public const string OPTION_DEVICE_CODE_FALSE = "--relocatable-device-code=false";

			public const string OPTION_DEVICE_DEBUG = "--device-debug";
			public const string OPTION_LINE_INFO = "--generate-line-info";

			public const string OPTION_MAX_REG_COUNT_ = "--maxrregcount=";
			public const string OPTION_FTZ_TRUE = "--ftz=true";
			public const string OPTION_FTZ_FALSE = "--ftz=false";
			public const string OPTION_PREC_SQRT_TRUE = "--prec-sqrt=true";
			public const string OPTION_PREC_SQRT_FALSE = "--prec-sqrt=false";
			public const string OPTION_PREC_DIV_TRUE = "--prec-div=true";
			public const string OPTION_PREC_DIV_FALSE = "--prec-div=false";
			public const string OPTION_FMAD_TRUE = "--fmad=true";
			public const string OPTION_FMAD_FALSE = "--fmad=false";
			public const string OPTION_USE_FAST_MATH = "--use_fast_math";

			public const string OPTION_DEFINE_MACRO_ = "--define-macro=";
			public const string OPTION_UNDEFINE_MACRO_ = "--undefine-macro=";
			public const string OPTION_INCLUDE_PATH_ = "--include-path=";
			public const string OPTION_PRE_INCLUDE_ = "--pre-include=";

			public const string OPTION_STD_CXX11 = "--std=c++11";
			public const string OPTION_BUILTIN_MOVE_FORWARD_TRUE = "--builtin-move-forward=true";
			public const string OPTION_BUILTIN_MOVE_FORWARD_FALSE = "--builtin-move-forward=false";
			public const string OPTION_BUILTIN_INITIALIZER_LIST_TRUE = "--builtin-initializer-list=true";
			public const string OPTION_BUILTIN_INITIALIZER_LIST_FALSE = "--builtin-initializer-list=false";

			public const string OPTION_DISABLE_WARNINGS = "--disable-warnings";
			public const string OPTION_RESTRICT = "--restrict";
			public const string OPTION_DEVICE_AS_DEFAULT_EXECUTION_SPACE = "--device-as-default-execution-space";

			IntPtr program = IntPtr.Zero;

			public void Create(string src, string name, string[] headers, string[] includeNames) {
				int numHeaders = 0;
				if (headers != null) {
					numHeaders = headers.Length;
				}
				nvrtcResult result = nvrtcCreateProgram(ref program, src, name, numHeaders, headers, includeNames);
				if (result != nvrtcResult.NVRTC_SUCCESS) {
					throw new Exception(result.ToString());
				}
			}

			public void Dispose() {
				if (program == null || program == IntPtr.Zero) {
					return;
				}
				nvrtcDestroyProgram(ref program);
			}

			public void Compile(params string[] options) {
				if (program == null || program == IntPtr.Zero) {
					return;
				}
				int numOptions = 0;
				if (options != null) {
					numOptions = options.Length;
				}
				nvrtcResult result = nvrtcCompileProgram(program, numOptions, options);
				if (result != nvrtcResult.NVRTC_SUCCESS) {
					throw new Exception(result.ToString());
				}
			}

			public string GetLog() {
				if (program == null || program == IntPtr.Zero) {
					return string.Empty;
				}
				long logSize = 0;
				nvrtcResult result = NVRTC.nvrtcGetProgramLogSize(program, ref logSize);
				if (result != nvrtcResult.NVRTC_SUCCESS) {
					throw new Exception(result.ToString());
				}

				StringBuilder log = new StringBuilder((int)logSize);
				result = NVRTC.nvrtcGetProgramLog(program, log);
				if (result != nvrtcResult.NVRTC_SUCCESS) {
					throw new Exception(result.ToString());
				}
				return log.ToString();
			}

			public string GetPTX() {
				if (program == null || program == IntPtr.Zero) {
					return string.Empty;
				}
				long ptxSize = 0;
				nvrtcResult result = NVRTC.nvrtcGetPTXSize(program, ref ptxSize);
				if (result != nvrtcResult.NVRTC_SUCCESS) {
					throw new Exception(result.ToString());
				}

				StringBuilder ptx = new StringBuilder((int)ptxSize);
				result = NVRTC.nvrtcGetPTX(program, ptx);
				if (result != nvrtcResult.NVRTC_SUCCESS) {
					throw new Exception(result.ToString());
				}
				return ptx.ToString();
			}
		}
	}
}
