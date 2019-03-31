using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	/// <summary>
	/// (Driver API) 
	/// </summary>
	public class Module : IDisposable {
		class Args : IDisposable {
			IntPtr ptr = IntPtr.Zero;
			List<GCHandle> handles = null;
			public IntPtr Pointer {
				get { return ptr; }
			}
			public Args() {
				handles = new List<GCHandle>();
			}
			~Args() {
				Dispose();
			}
			public void Dispose() {
				if (ptr != IntPtr.Zero) {
					Marshal.FreeHGlobal(ptr);
					ptr = IntPtr.Zero;
				}
				if (handles != null) {
					foreach (GCHandle handle in handles) {
						handle.Free();
					}
					handles.Clear();
					handles = null;
				}
			}
			public static Args Create(object[] args) {
				int length = args.Length;
				Args result = new Args();
				result.ptr = Marshal.AllocHGlobal(IntPtr.Size * length);
				result.handles = new List<GCHandle>();
				List<long> ptrList = new List<long>();
				foreach (object obj in args) {
					GCHandle handle = GetPinnedGCHandle(obj);
					result.handles.Add(handle);
					ptrList.Add(handle.AddrOfPinnedObject().ToInt64());
				}
				Marshal.Copy(ptrList.ToArray(), 0, result.ptr, length);
				return result;
			}
			static GCHandle GetPinnedGCHandle(object obj) {
				return GCHandle.Alloc(obj, GCHandleType.Pinned);
			}
		}

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
				Driver.ModuleUnload(module);
				module = IntPtr.Zero;
			}
		}

		public IntPtr GetFunction(string name) {
			return Driver.ModuleGetFunction(module, name);
		}

		public IntPtr GetGlobal(string name) {
			return Driver.ModuleGetGlobal(module, name);
		}

		public IntPtr GetSurfRef(string name) {
			return Driver.ModuleGetSurfRef(module, name);
		}

		public IntPtr GetTexRef(string name) {
			return Driver.ModuleGetTexRef(module, name);
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

		public void Load(string path) {
			module = Driver.ModuleLoad(path);
		}

		public void LoadData(string image) {
			//Dispose();
			IntPtr ptxImage = Marshal.StringToHGlobalAnsi(image);
			module = Driver.ModuleLoadData(ptxImage);
		}

		public void LoadDataEx(string image, uint numOptions, CUjit_option options, IntPtr optionValues) {
			//Dispose();
			IntPtr ptxImage = Marshal.StringToHGlobalAnsi(image);
			module = Driver.ModuleLoadDataEx(
				ptxImage, numOptions, options, optionValues
			);
		}

		public void LoadFatBinary(IntPtr fatCubin) {
			module = Driver.ModuleLoadFatBinary(fatCubin);
		}

		public void Excecute(string funcName, params object[] args) {
			IntPtr kernel = Driver.ModuleGetFunction(module, funcName);
			Args arguments = Args.Create(args);
			Driver.LaunchKernel(
				kernel,
				gridX, gridY, gridZ,		// grid dim
				blockX, blockY, blockZ,		// block dim
				0, IntPtr.Zero,             // shared mem and stream
				arguments.Pointer, IntPtr.Zero		// arguments
			);
			arguments.Dispose();
		}
	}
}
