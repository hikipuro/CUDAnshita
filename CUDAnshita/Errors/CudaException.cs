using System;

namespace CUDAnshita {
	/// <summary>
	/// C# interface exception.
	/// </summary>
	public class CudaException : Exception {
		public CudaException() {
		}
		public CudaException(string message) : base(message) {
		}
		public CudaException(string message, Exception innerException) : base(message, innerException) {
		}

		public static void Check(cudaError error, string message) {
			if (error != cudaError.cudaSuccess) {
				message = string.Format("{0}: {1}", error.ToString(), message);
				Exception exception = new CudaException(message);
				exception.Data.Add("cudaError", error);
				throw exception;
			}
		}

		public static void Check(CUresult error, string message) {
			if (error != CUresult.CUDA_SUCCESS) {
				message = string.Format("{0}: {1}", error.ToString(), message);
				Exception exception = new CudaException(message);
				exception.Data.Add("CUresult", error);
				throw exception;
			}
		}
	}
}
