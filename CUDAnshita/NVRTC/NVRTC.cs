using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CUDAnshita {
	/// <summary>
	/// NVRTC is a runtime compilation library for CUDA C++.
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/nvrtc/">http://docs.nvidia.com/cuda/nvrtc/</a>
	/// </remarks>
	public class NVRTC {
		/// <summary>
		/// NVRTC DLL functions.
		/// </summary>
		public class API {
			//const string DLL_PATH = "nvrtc64_80.dll";
			const string DLL_PATH = "nvrtc64_101_0.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			/// <summary>
			/// nvrtcGetErrorString is a helper function that returns a string describing the given nvrtcResult code,
			/// e.g., NVRTC_SUCCESS to "NVRTC_SUCCESS". For unrecognized enumeration values, it returns "NVRTC_ERROR unknown".
			/// </summary>
			/// <param name="result">CUDA Runtime Compilation API result code.</param>
			/// <returns>Message string for the given nvrtcResult code.</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern IntPtr nvrtcGetErrorString(nvrtcResult result);

			/// <summary>
			/// nvrtcVersion sets the output parameters major and minor with the CUDA Runtime Compilation version number.
			/// </summary>
			/// <param name="major">CUDA Runtime Compilation major version number.</param>
			/// <param name="minor">CUDA Runtime Compilation minor version number.</param>
			/// <returns>NVRTC_SUCCESS, NVRTC_ERROR_INVALID_INPUT</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern nvrtcResult nvrtcVersion(ref int major, ref int minor);

			/// <summary>
			/// nvrtcAddNameExpression notes the given name expression denoting a __global__ function or function template instantiation.
			/// </summary>
			/// <param name="prog">CUDA Runtime Compilation program.</param>
			/// <param name="name_expression">constant expression denoting a __global__ function or function template instantiation.</param>
			/// <returns>NVRTC_SUCCESS, NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION</returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern nvrtcResult nvrtcAddNameExpression(IntPtr prog, string name_expression);

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
			public static extern nvrtcResult nvrtcCompileProgram(
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
			/// NVRTC_SUCCESS,
			/// NVRTC_ERROR_OUT_OF_MEMORY,
			/// NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
			/// NVRTC_ERROR_INVALID_INPUT,
			/// NVRTC_ERROR_INVALID_PROGRAM
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
			public static extern nvrtcResult nvrtcCreateProgram(
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
			/// NVRTC_SUCCESS,
			/// NVRTC_ERROR_INVALID_PROGRAM
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

			/// <summary>
			/// nvrtcGetLoweredName extracts the lowered (mangled) name for a __global__ function or function template instantiation,
			/// and updates *lowered_name to point to it. The memory containing the name is released when the NVRTC program is destroyed
			/// by nvrtcDestroyProgram. The identical name expression must have been previously provided to nvrtcAddNameExpression.
			/// </summary>
			/// <param name="prog">CUDA Runtime Compilation program.</param>
			/// <param name="name_expression">constant expression denoting a __global__ function or function template instantiation.</param>
			/// <param name="lowered_name">initialized by the function to point to a C string containing the lowered (mangled) name corresponding to the provided name expression.</param>
			/// <returns>
			/// NVRTC_SUCCESS,
			/// NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
			/// NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern nvrtcResult nvrtcGetLoweredName(
				IntPtr prog,
				string name_expression,
				ref StringBuilder lowered_name);

			/// <summary>
			/// nvrtcGetPTX stores the PTX generated by the previous compilation of prog in the memory pointed by ptx.
			/// </summary>
			/// <param name="prog">CUDA Runtime Compilation program.</param>
			/// <param name="ptx">Compiled result.</param>
			/// <returns>
			/// NVRTC_SUCCESS,
			/// NVRTC_ERROR_INVALID_INPUT,
			/// NVRTC_ERROR_INVALID_PROGRAM
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
			public static extern nvrtcResult nvrtcGetPTX(IntPtr prog, StringBuilder ptx);

			/// <summary>
			/// nvrtcGetPTXSize sets ptxSizeRet with the size of the PTX generated by the previous compilation of prog (including the trailing NULL).
			/// </summary>
			/// <param name="prog">CUDA Runtime Compilation program.</param>
			/// <param name="ptxSizeRet">Size of the generated PTX (including the trailing NULL).</param>
			/// <returns>
			/// NVRTC_SUCCESS,
			/// NVRTC_ERROR_INVALID_INPUT,
			/// NVRTC_ERROR_INVALID_PROGRAM
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern nvrtcResult nvrtcGetPTXSize(IntPtr prog, ref long ptxSizeRet);

			/// <summary>
			/// nvrtcGetProgramLog stores the log generated by the previous compilation of prog in the memory pointed by log.
			/// </summary>
			/// <param name="prog">CUDA Runtime Compilation program.</param>
			/// <param name="log">Compilation log.</param>
			/// <returns>
			/// NVRTC_SUCCESS,
			/// NVRTC_ERROR_INVALID_INPUT,
			/// NVRTC_ERROR_INVALID_PROGRAM
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION, CharSet = CHAR_SET)]
			public static extern nvrtcResult nvrtcGetProgramLog(IntPtr prog, StringBuilder log);

			/// <summary>
			/// nvrtcGetProgramLogSize sets logSizeRet with the size of the log generated by the previous compilation of prog
			/// (including the trailing NULL).
			/// </summary>
			/// <param name="prog">CUDA Runtime Compilation program.</param>
			/// <param name="logSizeRet">Size of the compilation log (including the trailing NULL).</param>
			/// <returns>
			/// NVRTC_SUCCESS,
			/// NVRTC_ERROR_INVALID_INPUT,
			/// NVRTC_ERROR_INVALID_PROGRAM
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, ref long logSizeRet);
		}

		// ----- C# Interface

		/// <summary>
		/// nvrtcGetErrorString is a helper function that returns a string describing the given nvrtcResult code,
		/// e.g., NVRTC_SUCCESS to "NVRTC_SUCCESS". For unrecognized enumeration values, it returns "NVRTC_ERROR unknown".
		/// </summary>
		/// <param name="result">CUDA Runtime Compilation API result code.</param>
		/// <returns>Message string for the given nvrtcResult code.</returns>
		public static string GetErrorString(nvrtcResult result) {
			IntPtr ptr = API.nvrtcGetErrorString(result);
			return Marshal.PtrToStringAnsi(ptr);
		}

		/// <summary>
		/// nvrtcVersion sets the output parameters major and minor with the CUDA Runtime Compilation version number.
		/// </summary>
		/// <returns>version string.</returns>
		public static string Version() {
			int major = 0;
			int minor = 0;
			CheckResult(API.nvrtcVersion(ref major, ref minor));
			return string.Format("{0}.{1}", major, minor);
		}

		/// <summary>
		/// nvrtcAddNameExpression notes the given name expression denoting the address
		/// of a __global__ function or __device__/__constant__ variable.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="name_expression">constant expression denoting the address of a __global__ function or __device__/__constant__ variable.</param>
		public static void AddNameExpression(IntPtr prog, string name_expression) {
			CheckResult(API.nvrtcAddNameExpression(prog, name_expression));
		}

		/// <summary>
		/// nvrtcCompileProgram compiles the given program.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="numOptions"></param>
		/// <param name="options"></param>
		public static void CompileProgram(IntPtr prog, int numOptions, string[] options) {
			CheckResult(API.nvrtcCompileProgram(prog, numOptions, options));
		}

		/// <summary>
		/// nvrtcCreateProgram creates an instance of nvrtcProgram with the given input parameters,
		/// and sets the output parameter prog with it.
		/// </summary>
		/// <param name="src">CUDA program source.</param>
		/// <param name="name">CUDA program name. name can be NULL; "default_program" is used when name is NULL.</param>
		/// <param name="numHeaders">Number of headers used. numHeaders must be greater than or equal to 0.</param>
		/// <param name="headers">Sources of the headers. headers can be NULL when numHeaders is 0.</param>
		/// <param name="includeNames">Name of each header by which they can be included in the CUDA program source. includeNames can be NULL when numHeaders is 0.</param>
		/// <returns>CUDA Runtime Compilation program.</returns>
		public static IntPtr CreateProgram(string src, string name, int numHeaders, string[] headers, string[] includeNames) {
			IntPtr prog = IntPtr.Zero;
			CheckResult(API.nvrtcCreateProgram(ref prog, src, name, numHeaders, headers, includeNames));
			return prog;
		}

		/// <summary>
		/// nvrtcDestroyProgram destroys the given program.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		public static void DestroyProgram(IntPtr prog) {
			CheckResult(API.nvrtcDestroyProgram(ref prog));
		}

		/// <summary>
		/// nvrtcGetLoweredName extracts the lowered (mangled) name for a __global__ function
		/// or __device__/__constant__ variable, and updates *lowered_name to point to it.
		/// The memory containing the name is released when the NVRTC program is destroyed by nvrtcDestroyProgram.
		/// The identical name expression must have been previously provided to nvrtcAddNameExpression.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <param name="name_expression">constant expression denoting the address of a __global__ function or __device__/__constant__ variable.</param>
		/// <returns>initialized by the function to point to a C string containing the lowered (mangled) name corresponding to the provided name expression.</returns>
		public static string GetLoweredName(IntPtr prog, string name_expression) {
			StringBuilder lowered_name = new StringBuilder();
			CheckResult(API.nvrtcGetLoweredName(prog, name_expression, ref lowered_name));
			return lowered_name.ToString();
		}

		/// <summary>
		/// nvrtcGetPTX stores the PTX generated by the previous compilation of prog in the memory pointed by ptx.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <returns>Compiled result.</returns>
		public static string GetPTX(IntPtr prog) {
			long size = GetPTXSize(prog);
			StringBuilder ptx = new StringBuilder((int)size);
			CheckResult(API.nvrtcGetPTX(prog, ptx));
			return ptx.ToString();
		}

		/// <summary>
		/// nvrtcGetPTXSize sets ptxSizeRet with the size of the PTX generated by the previous compilation of prog
		/// (including the trailing NULL).
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <returns>Size of the generated PTX (including the trailing NULL).</returns>
		public static long GetPTXSize(IntPtr prog) {
			long ptxSizeRet = 0;
			CheckResult(API.nvrtcGetPTXSize(prog, ref ptxSizeRet));
			return ptxSizeRet;
		}

		/// <summary>
		/// nvrtcGetProgramLog stores the log generated by the previous compilation of prog in the memory pointed by log.
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <returns>Compilation log.</returns>
		public static string GetProgramLog(IntPtr prog) {
			long size = GetProgramLogSize(prog);
			StringBuilder log = new StringBuilder((int)size);
			CheckResult(API.nvrtcGetProgramLog(prog, log));
			return log.ToString();
		}

		/// <summary>
		/// nvrtcGetProgramLogSize sets logSizeRet with the size of the log generated by the previous compilation of prog
		/// (including the trailing NULL).
		/// </summary>
		/// <param name="prog">CUDA Runtime Compilation program.</param>
		/// <returns>Size of the compilation log (including the trailing NULL).</returns>
		public static long GetProgramLogSize(IntPtr prog) {
			long logSizeRet = 0;
			CheckResult(API.nvrtcGetProgramLogSize(prog, ref logSizeRet));
			return logSizeRet;
		}

		static void CheckResult(nvrtcResult result) {
			if (result != nvrtcResult.NVRTC_SUCCESS) {
				throw new CudaException(result.ToString());
			}
		}
	}

	/// <summary>
	/// The enumerated type nvrtcResult defines API call result codes.
	/// NVRTC API functions return nvrtcResult to indicate the call result.
	/// </summary>
	public enum nvrtcResult {
		NVRTC_SUCCESS = 0,
		NVRTC_ERROR_OUT_OF_MEMORY = 1,
		NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
		NVRTC_ERROR_INVALID_INPUT = 3,
		NVRTC_ERROR_INVALID_PROGRAM = 4,
		NVRTC_ERROR_INVALID_OPTION = 5,
		NVRTC_ERROR_COMPILATION = 6,
		NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
		NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
		NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
		NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
		NVRTC_ERROR_INTERNAL_ERROR = 11
	}
}
