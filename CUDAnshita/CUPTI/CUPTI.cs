using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CUDAnshita {
	public class CUPTI {
		/// <summary>
		/// CUPTI functions.
		/// </summary>
		public class API {
			const string DLL_PATH = "cupti64_102.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			/// <summary>
			/// Get the descriptive string for a CUptiResult.
			/// </summary>
			/// <param name="result"></param>
			/// <param name="str"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUptiResult cuptiGetResultString(
				CUptiResult result,
				ref IntPtr str      // const char**
			);

			/// <summary>
			/// Get the CUPTI API version.
			/// </summary>
			/// <param name="version"></param>
			/// <returns></returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern CUptiResult cuptiGetVersion(
				[In] ref int version // uint32_t*
			);
		}

		/// <summary>
		/// Get the descriptive string for a CUptiResult.
		/// </summary>
		/// <param name="result"></param>
		/// <returns></returns>
		public static string GetResultString(CUptiResult result) {
			IntPtr ptr = IntPtr.Zero;
			CheckResult(API.cuptiGetResultString(result, ref ptr));
			return Marshal.PtrToStringAnsi(ptr);
		}

		/// <summary>
		/// Get the CUPTI API version.
		/// </summary>
		/// <returns></returns>
		public static int GetVersion() {
			int version = 0;
			CheckResult(API.cuptiGetVersion(ref version));
			return version;
		}

		static void CheckResult(CUptiResult result) {
			if (result != CUptiResult.CUPTI_SUCCESS) {
				throw new CudaException(result.ToString());
			}
		}
	}
}
