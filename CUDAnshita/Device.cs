using CUDAnshita.API;
using CUDAnshita.Errors;
using System;

namespace CUDAnshita {
	public class Device : IDisposable {
		static bool initialized = false;

		int device = 0;

		public static int GetCount() {
			int count = 0;
			cudaError result = NvCuda.cuDeviceGetCount(ref count);
			CudaException.Check(result, "デバイス数の取得に失敗しました。");
			return count;
		}

		public Device(int deviceNumber) {
			cudaError result;

			if (initialized == false) {
				initialized = true;
				result = NvCuda.cuInit(0);
				CudaException.Check(result, "デバイスの初期化に失敗しました。");
			}

			result = NvCuda.cuDeviceGet(ref device, deviceNumber);
			CudaException.Check(result, "デバイスの取得に失敗しました。");
		}

		public void Dispose() {
		}

		public Context CreateContext() {
			return new Context(device);
		}

	}
}
