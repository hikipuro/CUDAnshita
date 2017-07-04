using System;
using System.Text;

namespace CUDAnshita {
	public class CudaMatrixFloat : CudaMatrixBase {
		const int ItemSize = sizeof(float);
		float[] _Host;

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
			get { return _Host[GetIndex(x, y)]; }
			set {
				_Dirty = true;
				_Host[GetIndex(x, y)] = value;
			}
		}

		public static CudaMatrixFloat operator -(CudaMatrixFloat matrix) {
			return matrix.Mul(-1f);
		}

		public static CudaMatrixFloat operator +(CudaMatrixFloat matrix, float value) {
			return matrix.Add(value);
		}

		public static CudaMatrixFloat operator +(float value, CudaMatrixFloat matrix) {
			return matrix.Add(value);
		}

		public static CudaMatrixFloat operator -(CudaMatrixFloat matrix, float value) {
			return matrix.Sub(value);
		}

		public static CudaMatrixFloat operator -(float value, CudaMatrixFloat matrix) {
			CudaMatrixFloat result = new CudaMatrixFloat(matrix._Rows, matrix._Cols);
			result.ForEach((x, y, item) => {
				int i = result.GetIndex(x, y);
				result._Host[i] = value - matrix[x, y];
			});
			result._Dirty = true;
			result.UpdateDeviceMemory();
			return result;
		}

		public static CudaMatrixFloat operator *(CudaMatrixFloat matrix, float value) {
			return matrix.Mul(value);
		}

		public static CudaMatrixFloat operator *(float value, CudaMatrixFloat matrix) {
			return matrix.Mul(value);
		}

		public static CudaMatrixFloat operator /(CudaMatrixFloat matrix, float value) {
			return matrix.Div(value);
		}

		public static CudaMatrixFloat operator /(float value, CudaMatrixFloat matrix) {
			CudaMatrixFloat result = new CudaMatrixFloat(matrix._Rows, matrix._Cols);
			result.ForEach((x, y, item) => {
				int i = result.GetIndex(x, y);
				result._Host[i] = value / matrix[x, y];
			});
			result._Dirty = true;
			result.UpdateDeviceMemory();
			return result;
		}

		public static CudaMatrixFloat operator %(CudaMatrixFloat matrix, float value) {
			CudaMatrixFloat result = new CudaMatrixFloat(matrix._Rows, matrix._Cols);
			result.ForEach((x, y, item) => {
				int i = result.GetIndex(x, y);
				result._Host[i] = matrix[x, y] % value;
			});
			result._Dirty = true;
			result.UpdateDeviceMemory();
			return result;
		}

		public CudaMatrixFloat(int rows, int cols) : this(rows, cols, true) {
		}

		public CudaMatrixFloat(int rows, int cols, bool hostAlloc) {
			_Rows = rows;
			_Cols = cols;

			if (hostAlloc) {
				_Host = new float[Count];
			}
			_Device = Runtime.Malloc(ItemSize * Count);
			Runtime.Memset(_Device, 0, ItemSize * Count);
		}

		public static CudaMatrixFloat FromByteArray(byte[] bytes) {
			if (bytes == null || bytes.Length < 12) {
				return null;
			}

			int intSize = sizeof(int);
			int intSize2 = intSize * 2;
			int[] matrixSize = new int[2];
			float[] data = new float[(bytes.Length - intSize2) / ItemSize];
			Buffer.BlockCopy(bytes, 0, matrixSize, 0, intSize2);
			Buffer.BlockCopy(bytes, intSize2, data, 0, bytes.Length - intSize2);

			int rows = matrixSize[0];
			int cols = matrixSize[1];
			CudaMatrixFloat matrix = new CudaMatrixFloat(rows, cols, false);
			matrix._Host = data;
			matrix._Dirty = true;
			matrix.UpdateDeviceMemory();
			return matrix;
		}

		public byte[] ToByteArray() {
			int intSize = sizeof(int);
			int intSize2 = intSize * 2;
			int size = intSize2 + _Host.Length * ItemSize;
			byte[] bytes = new byte[size];
			Buffer.BlockCopy(new int[] { _Rows, _Cols }, 0, bytes, 0, intSize2);
			Buffer.BlockCopy(_Host, 0, bytes, intSize2, bytes.Length - intSize2);
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

		public CudaMatrixFloat Sub(CudaMatrixFloat matrix) {
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
				-1f,
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

		public CudaMatrixFloat Add(float value) {
			CudaMatrixFloat result = new CudaMatrixFloat(_Rows, _Cols);
			ForEach((x, y, item) => {
				int i = GetIndex(x, y);
				result._Host[i] = item + value;
			});
			result._Dirty = true;
			result.UpdateDeviceMemory();
			return result;
		}

		public CudaMatrixFloat Sub(float value) {
			CudaMatrixFloat result = new CudaMatrixFloat(_Rows, _Cols);
			ForEach((x, y, item) => {
				int i = GetIndex(x, y);
				result._Host[i] = item - value;
			});
			result._Dirty = true;
			result.UpdateDeviceMemory();
			return result;
		}

		public CudaMatrixFloat Mul(float value) {
			CudaMatrixFloat result = new CudaMatrixFloat(_Rows, _Cols);
			UpdateDeviceMemory();

			IntPtr handle = cuBLAS.Create_v2();
			cuBLAS.Sgeam(
				handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				_Rows, _Cols,
				value,
				_Device, _Rows,
				0f,
				IntPtr.Zero, _Rows,
				result._Device, _Rows
			);
			cuBLAS.Destroy_v2(handle);

			result.UpdateHostMemory();
			return result;
		}

		public CudaMatrixFloat Div(float value) {
			CudaMatrixFloat result = new CudaMatrixFloat(_Rows, _Cols);
			ForEach((x, y, item) => {
				int i = GetIndex(x, y);
				result._Host[i] = item / value;
			});
			result._Dirty = true;
			result.UpdateDeviceMemory();
			return result;
		}

		public float Sum() {
			UpdateDeviceMemory();
			IntPtr handle = cuBLAS.Create_v2();
			float result = cuBLAS.Sasum_v2(
				handle,
				Count,
				_Device, 1
			);
			cuBLAS.Destroy_v2(handle);
			return result;
		}

		public int MaxIndex() {
			UpdateDeviceMemory();
			IntPtr handle = cuBLAS.Create_v2();
			int index = cuBLAS.Isamax_v2(
				handle,
				Count,
				_Device, 1
			);
			cuBLAS.Destroy_v2(handle);
			return index - 1;
		}

		public int MinIndex() {
			UpdateDeviceMemory();
			IntPtr handle = cuBLAS.Create_v2();
			int index = cuBLAS.Isamin_v2(
				handle,
				Count,
				_Device, 1
			);
			cuBLAS.Destroy_v2(handle);
			return index - 1;
		}

		public void ForEach(ForEachAction<int, int, float> action) {
			for (int y = 0; y < _Rows; y++) {
				for (int x = 0; x < _Cols; x++) {
					action(x, y, this[x, y]);
				}
			}
		}

		public CudaMatrixFloat Every(EveryFunc<float, float> func) {
			var result = new CudaMatrixFloat(_Rows, _Cols);
			for (int y = 0; y < _Rows; y++) {
				for (int x = 0; x < _Cols; x++) {
					result[x, y] = func(this[x, y]);
				}
			}
			return result;
		}

		public void UpdateHostMemory() {
			//Runtime.DeviceSynchronize();
			_Host = cuBLAS.GetMatrix<float>(_Rows, _Cols, _Device);
		}

		public void UpdateDeviceMemory() {
			if (_Dirty == false) {
				return;
			}
			cuBLAS.SetMatrix<float>(_Rows, _Cols, _Host, _Device);
			_Dirty = false;
		}

		public override string ToString() {
			var text = new StringBuilder();
			text.Append("[");

			int width = _Cols;
			int height = _Rows;
			for (var y = 0; y < height; y++) {
				if (height > 1) {
					text.AppendLine();
					text.Append("\t[");
				}
				for (var x = 0; x < width; x++) {
					text.Append(this[x, y].ToString());
					if (x + 1 != _Cols) {
						text.Append(", ");
					}
				}
				if (height > 1) {
					text.Append("]");
				}
			}
			if (height > 1) {
				text.AppendLine();
			}
			text.Append("]");
			return text.ToString();
		}

		int GetIndex(int x, int y) {
			return y *_Cols + x;
		}
	}
}
