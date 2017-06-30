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
			NvCuda.MemFree(ptr);
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
			return NvCuda.MemAlloc(byteSize);
		}

		private void _CopyHtoD<T>(IntPtr dest, T[] data) {
			int byteSize = Marshal.SizeOf(typeof(T)) * data.Length;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);
			MarshalUtil.Copy<T>(data, 0, ptr, data.Length);

			NvCuda.MemcpyHtoD(dest, ptr, byteSize);
			Marshal.FreeHGlobal(ptr);
		}

		private T[] _CopyDtoH<T>(IntPtr src, int count) {
			int byteSize = Marshal.SizeOf(typeof(T)) * count;
			IntPtr ptr = Marshal.AllocHGlobal(byteSize);
			NvCuda.MemcpyDtoH(ptr, src, byteSize);

			T[] dest = new T[count];
			MarshalUtil.Copy<T>(ptr, dest, 0, count);
			Marshal.FreeHGlobal(ptr);
			return dest;
		}

	}
}
