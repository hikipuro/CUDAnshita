using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	/// <summary>
	/// (Runtime API) 
	/// </summary>
	public class RuntimeDeviceMemory : IDisposable {
		Dictionary<string, IntPtr> list;

		public RuntimeDeviceMemory() {
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
			Runtime.Free(ptr);
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
			return Runtime.Malloc(byteSize);
		}

		private void _CopyHtoD<T>(IntPtr dest, T[] data) {
			Runtime.MemcpyH2D<T>(dest, data, data.Length);
		}

		private T[] _CopyDtoH<T>(IntPtr src, int count) {
			return Runtime.MemcpyD2H<T>(src, count);
		}
	}
}
