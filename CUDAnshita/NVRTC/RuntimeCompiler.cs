using System;
using System.Collections.Generic;
using System.IO;

namespace CUDAnshita {
	/// <summary>
	/// NVRTC runtime compilation API.
	/// </summary>
	public class RuntimeCompiler {
		public const string OPTION_TARGET_20 = "--gpu-architecture=compute_20";
		public const string OPTION_TARGET_30 = "--gpu-architecture=compute_30";
		public const string OPTION_TARGET_32 = "--gpu-architecture=compute_32";
		public const string OPTION_TARGET_35 = "--gpu-architecture=compute_35";
		public const string OPTION_TARGET_37 = "--gpu-architecture=compute_37";
		public const string OPTION_TARGET_50 = "--gpu-architecture=compute_50";
		public const string OPTION_TARGET_52 = "--gpu-architecture=compute_52";
		public const string OPTION_TARGET_53 = "--gpu-architecture=compute_53";
		public const string OPTION_TARGET_60 = "--gpu-architecture=compute_60";
		public const string OPTION_TARGET_61 = "--gpu-architecture=compute_61";
		public const string OPTION_TARGET_62 = "--gpu-architecture=compute_62";
		public const string OPTION_TARGET_70 = "--gpu-architecture=compute_70";
		public const string OPTION_TARGET_72 = "--gpu-architecture=compute_72";
		public const string OPTION_TARGET_75 = "--gpu-architecture=compute_75";

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

		Dictionary<string, string> headerList;
		List<string> optionList;
		string log = string.Empty;

		public string Log {
			get { return log; }
		}

		public RuntimeCompiler() {
			headerList = new Dictionary<string, string>();
			optionList = new List<string>();
		}

		~RuntimeCompiler() {
		}

		public static string OPTION_TARGET(string value) {
			return "--gpu-architecture=" + value;
		}

		public static string OPTION_MAX_REG_COUNT(string value) {
			return OPTION_MAX_REG_COUNT_ + value;
		}

		public static string OPTION_DEFINE_MACRO(string value) {
			return OPTION_DEFINE_MACRO_ + value;
		}

		public static string OPTION_UNDEFINE_MACRO(string value) {
			return OPTION_UNDEFINE_MACRO_ + value;
		}

		public static string OPTION_INCLUDE_PATH(string value) {
			return OPTION_INCLUDE_PATH_ + value;
		}

		public static string OPTION_PRE_INCLUDE(string value) {
			return OPTION_PRE_INCLUDE_ + value;
		}

		public void AddHeader(string name, string path) {
			headerList.Add(name, path);
		}

		public void AddOption(string option) {
			optionList.Add(option);
		}

		public void AddOptions(params string[] options) {
			optionList.AddRange(options);
		}

		public string Compile(string name, string src) {
			IntPtr program = CreateProgram(src, name);

			int numOptions = optionList.Count;
			nvrtcResult result = NVRTC.API.nvrtcCompileProgram(
				program, numOptions, optionList.ToArray()
			);
			if (result != nvrtcResult.NVRTC_SUCCESS) {
				log = GetLog(program);
				return null;
			}
			log = GetLog(program);

			string ptx = GetPTX(program);
			NVRTC.DestroyProgram(program);

			return ptx;
		}

		public string Compile(string path) {
			if (File.Exists(path) == false) {
				log = string.Format("src file not found. ({0})", path);
				return null;
			}
			string name = Path.GetFileName(path);
			string src = File.ReadAllText(path);
			return Compile(name, src);
		}

		IntPtr CreateProgram(string src, string name) {
			int numHeaders = headerList.Count;
			if (numHeaders == 0) {
				return NVRTC.CreateProgram(src, name, numHeaders, null, null);
			}
			string[] headers = new string[numHeaders];
			string[] includeNames = new string[numHeaders];
			headerList.Values.CopyTo(headers, 0);
			headerList.Keys.CopyTo(includeNames, 0);
			return NVRTC.CreateProgram(src, name, numHeaders, headers, includeNames);
		}

		string GetLog(IntPtr program) {
			if (program == null || program == IntPtr.Zero) {
				return string.Empty;
			}
			return NVRTC.GetProgramLog(program);
		}

		string GetPTX(IntPtr program) {
			if (program == null || program == IntPtr.Zero) {
				return string.Empty;
			}
			return NVRTC.GetPTX(program);
		}
	}
}
