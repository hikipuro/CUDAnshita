using CUDAnshita.API;
using CUDAnshita.Errors;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	public class DeviceMemory : IDisposable {
		Dictionary<string, IntPtr> list;

		public DeviceMemory() {
			list = new Dictionary<string, IntPtr>();
		}

		public IntPtr this[string name] {
			get { return list[name]; }
		}

		public void Alloc<T>(string name, int count) {
			IntPtr ptr = _Alloc(Marshal.SizeOf(typeof(T)) * count);
			list.Add(name, ptr);
		}

		public void Add<T>(string name, T[] data) {
			int size = Marshal.SizeOf(typeof(T)) * data.Length;
			IntPtr ptr = _Alloc(size);
			_CopyHtoD<T>(ptr, data);
			list.Add(name, ptr);
		}

		public void Add<T>(string name, List<T> data) {
			int size = Marshal.SizeOf(typeof(T)) * data.Count;
			IntPtr ptr = _Alloc(size);
			_CopyHtoD<T>(ptr, data.ToArray());
			list.Add(name, ptr);
		}

		public void Remove(string name) {
			IntPtr ptr = list[name];
			NvCuda.cuMemFree(ptr);
			list.Remove(name);
		}

		public void Clear() {
			string[] keys = new string[list.Count];
			list.Keys.CopyTo(keys, 0);
			foreach (string name in keys) {
				Remove(name);
			}
			list.Clear();
		}

		public IntPtr GetPointer(string name) {
			return list[name];
		}

		public T[] Read<T>(string name, int count) {
			IntPtr ptr = list[name];
			T[] data = _CopyDtoH<T>(ptr, count);
			return data;
		}

		public void Write<T>(string name, T[] data) {
			IntPtr ptr = list[name];
			_CopyHtoD<T>(ptr, data);
		}

		public void Dispose() {
			Clear();
		}

		private IntPtr _Alloc(int byteSize) {
			IntPtr ptr = IntPtr.Zero;
			cudaError result = NvCuda.cuMemAlloc(ref ptr, byteSize);
			CudaException.Check(result, "デバイスメモリの割り当てに失敗しました。");
			return ptr;
		}

		private void _CopyHtoD<T>(IntPtr dest, T[] data) {
			int byteSize = Marshal.SizeOf(typeof(T)) * data.Length;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(data, 0, ptr, data.Length);

			cudaError result = NvCuda.cuMemcpyHtoD(dest, ptr, byteSize);
			CudaException.Check(result, "メインメモリからデバイスメモリへのコピーに失敗しました。");
			Marshal.FreeHGlobal(ptr);
		}

		private T[] _CopyDtoH<T>(IntPtr src, int count) {
			int byteSize = Marshal.SizeOf(typeof(T)) * count;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);
			cudaError result = NvCuda.cuMemcpyDtoH(ptr, src, byteSize);
			CudaException.Check(result, "デバイスメモリからメインメモリへのコピーに失敗しました。");

			T[] dest = new T[count];
			MarshalUtil.Copy<T>(ptr, dest, 0, count);
			Marshal.FreeHGlobal(ptr);
			return dest;
		}

	}
}
