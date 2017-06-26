using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CUDAnshita.API {
	/// <summary>
	/// NVRTC is a runtime compilation library for CUDA C++.
	/// http://docs.nvidia.com/cuda/nvrtc/
	/// </summary>
	public class NVRTC {
		const string DLL_PATH = "nvrtc64_80.dll";
		const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
		const CharSet CHAR_SET = CharSet.Ansi;

		/// <summary>
		/// nvrtcGetErrorString is a helper function that returns a string describing the given nvrtcResult code,
		/// e.g., NVRTC_SUCCESS to "NVRTC_SUCCESS". For unrecognized enumeration values, it returns "NVRTC_ERROR unknown".
		/// </summary>
		/// <param name="result">CUDA Runtime Compilation API result code.</param>
		/// <returns>Message string for the given nvrtcResult code.</returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern IntPtr nvrtcGetErrorString(nvrtcResult result);

		/// <summary>
		/// nvrtcVersion sets the output parameters major and minor with the CUDA Runtime Compilation version number.
		/// </summary>
		/// <param name="major">CUDA Runtime Compilation major version number.</param>
		/// <param name="minor">CUDA Runtime Compilation minor version number.</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_INVALID_INPUT
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcVersion(ref int major, ref int minor);

		/// <summary>
		/// nvrtcAddNameExpression notes the given name expression denoting a __global__ function or function template instantiation.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="name_expression">constant expression denoting a __global__ function or function template instantiation.</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcAddNameExpression(IntPtr prog, string name_expression);

		/// <summary>
		/// nvrtcCompileProgram compiles the given program.
		/// </summary>
		/// <remarks>
		/// It supports compile options listed in Supported Compile Options.
		/// </remarks>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="numOptions"></param>
		/// <param name="options"></param>
		/// <returns></returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
		static extern nvrtcResult nvrtcCompileProgram(
			IntPtr prog,
			int numOptions,
			[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
			string[] options);

		/// <summary>
		/// nvrtcCreateProgram creates an instance of nvrtcProgram with the given input parameters, and sets the output parameter prog with it.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="src">CUDA program source.</param>
		/// <param name="name">CUDA program name. name can be NULL; "default_program" is used when name is NULL.</param>
		/// <param name="numHeaders">Number of headers used. numHeaders must be greater than or equal to 0.</param>
		/// <param name="headers">Sources of the headers. headers can be NULL when numHeaders is 0.</param>
		/// <param name="includeNames">Name of each header by which they can be included in the CUDA program source. includeNames can be NULL when numHeaders is 0.</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_OUT_OF_MEMORY
		/// - NVRTC_ERROR_PROGRAM_CREATION_FAILURE
		/// - NVRTC_ERROR_INVALID_INPUT
		/// - NVRTC_ERROR_INVALID_PROGRAM
		/// </returns>
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

		/// <summary>
		/// nvrtcDestroyProgram destroys the given program.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_INVALID_PROGRAM
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

		/// <summary>
		/// nvrtcGetLoweredName extracts the lowered (mangled) name for a __global__ function or function template instantiation,
		/// and updates *lowered_name to point to it. The memory containing the name is released when the NVRTC program is destroyed
		/// by nvrtcDestroyProgram. The identical name expression must have been previously provided to nvrtcAddNameExpression.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="name_expression">constant expression denoting a __global__ function or function template instantiation.</param>
		/// <param name="lowered_name">initialized by the function to point to a C string containing the lowered (mangled) name corresponding to the provided name expression.</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
		/// - NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcGetLoweredName(
			IntPtr prog,
			string name_expression,
			[MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
			string[] lowered_name);

		/// <summary>
		/// nvrtcGetPTX stores the PTX generated by the previous compilation of prog in the memory pointed by ptx.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="ptx">Compiled result.</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_INVALID_INPUT
		/// - NVRTC_ERROR_INVALID_PROGRAM
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
		static extern nvrtcResult nvrtcGetPTX(IntPtr prog, StringBuilder ptx);

		/// <summary>
		/// nvrtcGetPTXSize sets ptxSizeRet with the size of the PTX generated by the previous compilation of prog (including the trailing NULL).
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="ptxSizeRet">Size of the generated PTX (including the trailing NULL).</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_INVALID_INPUT
		/// - NVRTC_ERROR_INVALID_PROGRAM
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcGetPTXSize(IntPtr prog, ref long ptxSizeRet);

		/// <summary>
		/// nvrtcGetProgramLog stores the log generated by the previous compilation of prog in the memory pointed by log.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="log">Compilation log.</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_INVALID_INPUT
		/// - NVRTC_ERROR_INVALID_PROGRAM
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
		static extern nvrtcResult nvrtcGetProgramLog(IntPtr prog, StringBuilder log);

		/// <summary>
		/// nvrtcGetProgramLogSize sets logSizeRet with the size of the log generated by the previous compilation of prog
		/// (including the trailing NULL).
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="logSizeRet">Size of the compilation log (including the trailing NULL).</param>
		/// <returns>
		/// - NVRTC_SUCCESS
		/// - NVRTC_ERROR_INVALID_INPUT
		/// - NVRTC_ERROR_INVALID_PROGRAM
		/// </returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, ref long logSizeRet);

		/// <summary>
		/// CUDA Runtime Compilation version number.
		/// </summary>
		/// <returns></returns>
		public static string GetVersion() {
			int major = 0;
			int minor = 0;
			nvrtcResult result = nvrtcVersion(ref major, ref minor);
			if (result != nvrtcResult.NVRTC_SUCCESS) {
				throw new Exception(result.ToString());
			}
			return string.Format("{0}.{1}", major, minor);
		}

		/// <summary>
		/// returns a string describing the given nvrtcResult code,
		/// e.g., NVRTC_SUCCESS to "NVRTC_SUCCESS". For unrecognized enumeration values, it returns "NVRTC_ERROR unknown".
		/// </summary>
		/// <param name="result"></param>
		/// <returns></returns>
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
