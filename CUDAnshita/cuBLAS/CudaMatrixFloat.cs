using System;

namespace CUDAnshita {
	public class CudaMatrixFloat {
		public delegate void ForEachAction<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3);

		const int ItemSize = sizeof(float);
		int _Cols;
		int _Rows;
		float[] _Host;
		IntPtr _Device = IntPtr.Zero;
		bool _Dirty = false;

		public int Cols {
			get { return _Cols; }
		}

		public int Rows {
			get { return _Rows; }
		}

		public int Count {
			get { return _Cols * _Rows; }
		}

		public float[] Data {
			get { return _Host; }
		}

		public float this[int i] {
			get { return _Host[i]; }
			set {
				_Dirty = true;
				_Host[i] = value;
			}
		}

		public float this[int x, int y] {
			get { return _Host[y * _Cols + x]; }
			set {
				_Dirty = true;
				_Host[y * _Cols + x] = value;
			}
		}

		public CudaMatrixFloat(int rows, int cols) {
			_Rows = rows;
			_Cols = cols;

			_Host = new float[Count];
			_Device = Runtime.Malloc(ItemSize * Count);
		}

		~CudaMatrixFloat() {
			if (_Device != IntPtr.Zero) {
				Runtime.Free(_Device);
				_Device = IntPtr.Zero;
			}
		}

		public static CudaMatrixFloat FromByteArray(byte[] bytes) {
			if (bytes == null || bytes.Length < 12) {
				return null;
			}

			int intSize = sizeof(int);
			int[] matrixSize = new int[2];
			float[] data = new float[(bytes.Length - intSize * 2) / ItemSize];
			Buffer.BlockCopy(bytes, 0, matrixSize, 0, intSize * 2);
			Buffer.BlockCopy(bytes, intSize * 2, data, 0, bytes.Length - intSize * 2);

			CudaMatrixFloat matrix = new CudaMatrixFloat(matrixSize[0], matrixSize[1]);
			matrix._Host = data;
			matrix._Dirty = true;
			return matrix;
		}

		public byte[] ToByteArray() {
			int intSize = sizeof(int);
			int size = intSize * 2 + _Host.Length * ItemSize;
			byte[] bytes = new byte[size];
			Buffer.BlockCopy(new int[] { _Rows, _Cols }, 0, bytes, 0, intSize * 2);
			Buffer.BlockCopy(_Host, 0, bytes, intSize * 2, bytes.Length - intSize * 2);
			return bytes;
		}

		public CudaMatrixFloat Add(CudaMatrixFloat matrix) {
			int rows = Math.Max(_Rows, matrix._Rows);
			int cols = Math.Max(_Cols, matrix._Cols);
			CudaMatrixFloat result = new CudaMatrixFloat(rows, cols);

			UpdateDeviceMemory();
			matrix.UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sgeam(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, _Cols,
				1f,
				_Device, _Rows,
				1f,
				matrix._Device, _Rows,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public CudaMatrixFloat Mul(CudaMatrixFloat matrix) {
			int rows = Math.Max(_Rows, matrix._Rows);
			int cols = Math.Max(_Cols, matrix._Cols);
			CudaMatrixFloat result = new CudaMatrixFloat(_Rows, matrix._Cols);

			UpdateDeviceMemory();
			matrix.UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sdgmm(
				handle,
				cublasSideMode.CUBLAS_SIDE_LEFT,
				_Rows, _Cols,
				_Device, _Rows,
				matrix._Device, _Rows,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public CudaMatrixFloat Dot(CudaMatrixFloat matrix) {
			CudaMatrixFloat result = new CudaMatrixFloat(_Rows, matrix._Cols);

			UpdateDeviceMemory();
			matrix.UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sgemm_v2(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, matrix._Cols, _Cols,
				1f,
				_Device, _Rows,
				matrix._Device, matrix._Rows,
				1f,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public void ForEach(ForEachAction<int, int, float> action) {
			for (int y = 0; y < _Rows; y++) {
				for (int x = 0; x < _Cols; x++) {
					action(x, y, this[x, y]);
				}
			}
		}

		void UpdateHostMemory() {
			_Host = cuBLAS.GetMatrix<float>(_Rows, _Cols, _Device);
		}

		void UpdateDeviceMemory() {
			if (_Dirty == false) {
				return;
			}
			_Dirty = false;
			cuBLAS.SetMatrix<float>(_Rows, _Cols, _Host, _Device);
		}
	}
}
