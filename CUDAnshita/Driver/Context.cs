using CUDAnshita.Errors;
using System;

namespace CUDAnshita {
	public class Context : IDisposable {
		IntPtr context = IntPtr.Zero;

		internal Context(int deviceHandle) {
			CUresult result;
			result = NvCuda.API.cuCtxCreate(ref context, 0, deviceHandle);
			CudaException.Check(result, "コンテキストの作成に失敗しました。");
		}

		public void Dispose() {
			if (context != IntPtr.Zero) {
				NvCuda.API.cuCtxDestroy(context);
			}
		}

		public void Synchronize() {
			CUresult result;
			result = NvCuda.API.cuCtxSynchronize();
			CudaException.Check(result, "スレッドの同期に失敗しました。");
		}
	}
}
