using CUDAnshita.API;
using CUDAnshita.Errors;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	public class Module : IDisposable {
		IntPtr module = IntPtr.Zero;
		uint gridX = 0;
		uint gridY = 0;
		uint gridZ = 0;
		uint blockX = 0;
		uint blockY = 0;
		uint blockZ = 0;

		public Module() {
		}

		public void Dispose() {
			if (module != IntPtr.Zero) {
				NvCuda.cuModuleUnload(module);
			}
		}

		public void SetBlockCount(int x, int y, int z) {
			gridX = (uint)x;
			gridY = (uint)y;
			gridZ = (uint)z;
		}

		public void SetThreadCount(int x, int y, int z) {
			blockX = (uint)x;
			blockY = (uint)y;
			blockZ = (uint)z;
		}

		public void LoadData(string image) {
			Dispose();
			cudaError result;
			IntPtr ptxImage = Marshal.StringToHGlobalAnsi(image);
			result = NvCuda.cuModuleLoadData(ref module, ptxImage);
			CudaException.Check(result, "モジュールデータのロードに失敗しました。");
		}

		public void LoadDataEx(string image, uint numOptions, CUjit_option options, IntPtr optionValues) {
			Dispose();
			cudaError result;
			IntPtr ptxImage = Marshal.StringToHGlobalAnsi(image);
			result = NvCuda.cuModuleLoadDataEx(
				ref module, ptxImage, numOptions, options, optionValues
			);
			CudaException.Check(result, "モジュールデータのロードに失敗しました。");
		}

		public void Excecute(string funcName, params object[] args) {
			cudaError result;

			IntPtr kernel = IntPtr.Zero;
			result = NvCuda.cuModuleGetFunction(ref kernel, module, funcName);
			CudaException.Check(result, "関数の取得に失敗しました。");

			IntPtr ptrArgs = Marshal.AllocHGlobal(IntPtr.Size * args.Length);
			List<GCHandle> handles = new List<GCHandle>();
			List<long> ptrList = new List<long>();
			foreach (object obj in args) {
				GCHandle handle = _GetPinnedGCHandle(obj);
				handles.Add(handle);
				ptrList.Add(handle.AddrOfPinnedObject().ToInt64());
			}
			Marshal.Copy(ptrList.ToArray(), 0, ptrArgs, args.Length);

			result = NvCuda.cuLaunchKernel(
				kernel,
				gridX, gridY, gridZ,			// grid dim
				blockX, blockY, blockZ,	// block dim
				0, IntPtr.Zero,				// shared mem and stream
				ptrArgs, IntPtr.Zero		// arguments
			);

			CudaException.Check(result, "関数の実行が失敗しました。");

			foreach (GCHandle handle in handles) {
				handle.Free();
			}
		}

		private GCHandle _GetPinnedGCHandle(object obj) {
			return GCHandle.Alloc(obj, GCHandleType.Pinned);
		}
	}
}
