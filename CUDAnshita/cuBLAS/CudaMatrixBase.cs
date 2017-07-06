using System;

namespace CUDAnshita {
	public class CudaMatrixBase : IDisposable {
		public delegate void ForEachAction<T1, T2>(T1 arg1, T2 arg2);
		public delegate void ForEachAction<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3);
		public delegate TResult EveryFunc<in T1, out TResult>(T1 arg1);

		protected int _Cols;
		protected int _Rows;

		protected IntPtr _Device = IntPtr.Zero;
		protected bool _DirtyHost = false;
		protected bool _DirtyDevice = false;

		public int Cols {
			get { return _Cols; }
		}

		public int Rows {
			get { return _Rows; }
		}

		public int Count {
			get { return _Cols * _Rows; }
		}

		~CudaMatrixBase() {
			//Console.WriteLine("~CudaMatrixBase");
			Dispose();
		}

		public void Dispose() {
			if (_Device != IntPtr.Zero) {
				Runtime.Free(_Device);
				_Device = IntPtr.Zero;
			}
		}
	}
}
